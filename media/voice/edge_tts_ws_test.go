package voice

import (
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// edgeFrameCapture 收集 mock 服务端收到的文本帧（speech.config + ssml）与握手 query，供断言。
type edgeFrameCapture struct {
	mu     sync.Mutex
	frames []string
	query  string
}

func (c *edgeFrameCapture) add(s string) {
	c.mu.Lock()
	c.frames = append(c.frames, s)
	c.mu.Unlock()
}

func (c *edgeFrameCapture) setQuery(q string) {
	c.mu.Lock()
	c.query = q
	c.mu.Unlock()
}

func (c *edgeFrameCapture) rawQuery() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.query
}

func (c *edgeFrameCapture) ssml() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, f := range c.frames {
		if strings.Contains(f, "Path:ssml") {
			return f
		}
	}
	return ""
}

// mockEdgeWSServer 模拟 Edge Read Aloud 协议：读 config + ssml 两帧并捕获，回一帧二进制音频
// （[2字节头长][Path:audio 头][音频负载]）再回一帧 turn.end 文本，最后关闭。
func mockEdgeWSServer(t *testing.T, audioPayload []byte) (*httptest.Server, *edgeFrameCapture) {
	t.Helper()
	cap := &edgeFrameCapture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.setQuery(r.URL.RawQuery)
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusInternalError, "")
		ctx := r.Context()

		for range 2 {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			cap.add(string(data))
		}

		header := []byte("X-RequestId:test\r\nContent-Type:audio/mpeg\r\nPath:audio\r\n\r\n")
		frame := make([]byte, 2+len(header)+len(audioPayload))
		binary.BigEndian.PutUint16(frame[:2], uint16(len(header)))
		copy(frame[2:], header)
		copy(frame[2+len(header):], audioPayload)
		if err := conn.Write(ctx, websocket.MessageBinary, frame); err != nil {
			return
		}

		_ = conn.Write(ctx, websocket.MessageText,
			[]byte("X-RequestId:test\r\nContent-Type:application/json; charset=utf-8\r\nPath:turn.end\r\n\r\n{}"))
		conn.Close(websocket.StatusNormalClosure, "")
	}))
	return srv, cap
}

func wsURL(httpURL string) string {
	return "ws" + strings.TrimPrefix(httpURL, "http")
}

// 复现 v0.4.7 修复：旧实现 POST Azure 付费端点 → 405 朗读失效。
// 新实现走免费 WebSocket 协议，能从二进制音频帧累积出 MP3；并校验上送的 SSML
// （音色 / speed→rate / XML 转义防注入）。
func TestEdgeTTS_Synthesize_WebSocketReturnsAudio(t *testing.T) {
	want := []byte{0xFF, 0xFB, 0x90, 0x00, 'M', 'P', '3'} // 伪 MP3 负载
	srv, cap := mockEdgeWSServer(t, want)
	defer srv.Close()

	tts := NewEdgeTTS(
		EdgeTTSWithEndpoint(wsURL(srv.URL)),
		EdgeTTSWithHTTPClient(srv.Client()),
		EdgeTTSWithVoice("zh-CN-YunjianNeural"),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := tts.Synthesize(ctx, "你好<inject>", SynthesizeOptions{Speed: 1.5})
	if err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	if res.Format != FormatMP3 {
		t.Errorf("格式应为 MP3，实际 %v", res.Format)
	}
	if string(res.Audio) != string(want) {
		t.Errorf("音频负载不匹配：got=%v want=%v", res.Audio, want)
	}
	if res.Size != len(want) {
		t.Errorf("Size 应为 %d，实际 %d", len(want), res.Size)
	}

	ssml := cap.ssml()
	if !strings.Contains(ssml, "zh-CN-YunjianNeural") {
		t.Errorf("SSML 未含音色: %q", ssml)
	}
	if !strings.Contains(ssml, "+50%") {
		t.Errorf("SSML rate 计算错误（speed 1.5 应 +50%%）: %q", ssml)
	}
	if !strings.Contains(ssml, "&lt;inject&gt;") || strings.Contains(ssml, "<inject>") {
		t.Errorf("SSML 注入字符未转义: %q", ssml)
	}
}

// 服务端只发 turn.end、不发任何音频 → 应报错而非返回空音频。
func TestEdgeTTS_Synthesize_NoAudioIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusInternalError, "")
		ctx := r.Context()
		for range 2 {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
		_ = conn.Write(ctx, websocket.MessageText, []byte("Path:turn.end\r\n\r\n{}"))
		conn.Close(websocket.StatusNormalClosure, "")
	}))
	defer srv.Close()

	tts := NewEdgeTTS(EdgeTTSWithEndpoint(wsURL(srv.URL)), EdgeTTSWithHTTPClient(srv.Client()))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if _, err := tts.Synthesize(ctx, "x", SynthesizeOptions{}); err == nil {
		t.Error("无音频应返回 error")
	}
}

// #4：GEC 版本可覆盖（微软封禁旧版时热修）——覆盖值应出现在握手 query。
func TestEdgeTTS_GECVersionOverride(t *testing.T) {
	srv, cap := mockEdgeWSServer(t, []byte{0xFF, 0xFB, 0x00})
	defer srv.Close()
	tts := NewEdgeTTS(
		EdgeTTSWithEndpoint(wsURL(srv.URL)),
		EdgeTTSWithHTTPClient(srv.Client()),
		EdgeTTSWithGECVersion("1-999.0.0.0"),
	)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := tts.Synthesize(ctx, "x", SynthesizeOptions{}); err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	if !strings.Contains(cap.rawQuery(), "Sec-MS-GEC-Version=1-999.0.0.0") {
		t.Errorf("握手 query 未带覆盖后的版本: %q", cap.rawQuery())
	}
}

// #5：超时随文本长度伸缩，封顶 edgeMaxTimeout——修复固定 60s 长文本必超时。
func TestEdgeTTS_EffectiveTimeout_ScalesWithLength(t *testing.T) {
	tts := NewEdgeTTS()
	base := tts.effectiveTimeout("")
	if base != tts.timeout {
		t.Errorf("空文本应为基础超时 %s，实际 %s", tts.timeout, base)
	}
	mid := tts.effectiveTimeout(strings.Repeat("字", 1000))
	if mid <= base {
		t.Errorf("长文本超时应大于基础：%s vs %s", mid, base)
	}
	capped := tts.effectiveTimeout(strings.Repeat("字", 1_000_000))
	if capped != edgeMaxTimeout {
		t.Errorf("超长文本应封顶 %s，实际 %s", edgeMaxTimeout, capped)
	}
}

// #3：握手 403 且检测到显著时钟偏移 → 按服务端时间重算 token 重试一次后成功。
func TestEdgeTTS_Synthesize_RetriesOn403ClockSkew(t *testing.T) {
	want := []byte{0xFF, 0xFB, 0x12, 0x34}
	var calls int32
	clientNow := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			// 首次：返回 403 + 比客户端快 10 分钟的 Date 头（制造显著偏移）。
			w.Header().Set("Date", clientNow.Add(10*time.Minute).Format(http.TimeFormat))
			w.WriteHeader(http.StatusForbidden)
			return
		}
		// 重试：正常 WS 升级 + 回音频。
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusInternalError, "")
		ctx := r.Context()
		for range 2 {
			if _, _, err := conn.Read(ctx); err != nil {
				return
			}
		}
		header := []byte("Content-Type:audio/mpeg\r\nPath:audio\r\n\r\n")
		frame := make([]byte, 2+len(header)+len(want))
		binary.BigEndian.PutUint16(frame[:2], uint16(len(header)))
		copy(frame[2:], header)
		copy(frame[2+len(header):], want)
		_ = conn.Write(ctx, websocket.MessageBinary, frame)
		_ = conn.Write(ctx, websocket.MessageText, []byte("Path:turn.end\r\n\r\n{}"))
		conn.Close(websocket.StatusNormalClosure, "")
	}))
	defer srv.Close()

	tts := NewEdgeTTS(EdgeTTSWithEndpoint(wsURL(srv.URL)), EdgeTTSWithHTTPClient(srv.Client()))
	tts.now = func() time.Time { return clientNow }
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := tts.Synthesize(ctx, "hi", SynthesizeOptions{})
	if err != nil {
		t.Fatalf("时钟偏移重试后应成功，实际: %v", err)
	}
	if string(res.Audio) != string(want) {
		t.Errorf("重试后音频不匹配：got=%v want=%v", res.Audio, want)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Errorf("应恰好握手 2 次（首次 403 + 重试成功），实际 %d", calls)
	}
}

func TestEdgeSecMSGEC_DeterministicHexUpper(t *testing.T) {
	// 同一 5 分钟窗口内 token 稳定；输出为 64 位大写十六进制。
	base := time.Date(2026, 6, 26, 10, 1, 0, 0, time.UTC)
	a := edgeSecMSGEC(base)
	b := edgeSecMSGEC(base.Add(90 * time.Second)) // 同 5 分钟窗口
	if a != b {
		t.Errorf("同窗口 token 应一致：%s vs %s", a, b)
	}
	if len(a) != 64 {
		t.Fatalf("token 应为 64 位十六进制，实际 %d 位", len(a))
	}
	if a != strings.ToUpper(a) {
		t.Error("token 应为大写十六进制")
	}
	c := edgeSecMSGEC(base.Add(6 * time.Minute)) // 跨窗口
	if a == c {
		t.Error("跨 5 分钟窗口 token 应变化")
	}
}

func TestParseEdgeAudioFrame(t *testing.T) {
	header := []byte("Path:audio\r\n\r\n")
	payload := []byte{1, 2, 3, 4}
	frame := make([]byte, 2+len(header)+len(payload))
	binary.BigEndian.PutUint16(frame[:2], uint16(len(header)))
	copy(frame[2:], header)
	copy(frame[2+len(header):], payload)

	got, ok := parseEdgeAudioFrame(frame)
	if !ok || string(got) != string(payload) {
		t.Errorf("应解析出音频负载：ok=%v got=%v", ok, got)
	}

	// 非 audio 帧（如 word-boundary 的 audio.metadata）应忽略。
	meta := []byte("Path:audio.metadata\r\n\r\n")
	mframe := make([]byte, 2+len(meta))
	binary.BigEndian.PutUint16(mframe[:2], uint16(len(meta)))
	copy(mframe[2:], meta)
	if _, ok := parseEdgeAudioFrame(mframe); ok {
		t.Error("非 Path:audio 帧应被忽略")
	}

	if _, ok := parseEdgeAudioFrame([]byte{0x01}); ok {
		t.Error("过短帧应返回 false")
	}
}
