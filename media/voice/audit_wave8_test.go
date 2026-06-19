package voice

// audit_wave8_test.go 是 Wave8 审计测试：针对 media/voice 包公开 API 的全场景覆盖。
//
// 覆盖维度：
//   - 正常路径：请求构造 / 响应解析（OpenAI STT/TTS、MiniMax TTS 用 httptest 注入）
//   - 边界：空 / nil / 超长 / 0 / 负 / Unicode / 默认值回退
//   - 错误路径：非 200、解析失败、上游业务错误码、hex 解码失败
//   - 状态一致性：异步 Submit→Poll 终态、失败态传播
//   - 并发竞态：WakeWordDetector 读写并发（-race 下验证）
//   - 安全：错误体脱敏、SSML/XML 注入转义
//
// 注入点说明：
//   - OpenAISTT/OpenAITTS 提供 WithBaseURL + WithHTTPClient → 可用 httptest 完整测请求/响应
//   - MiniMaxTTS 提供 WithMiniMaxBaseURL + WithMiniMaxHTTPClient → 同上
//   - AzureSTT/AzureTTS 提供 AzureSTTWithBaseURL/AzureTTSWithBaseURL + *WithHTTPClient（Wave8 修复）
//   - EdgeTTS 提供 EdgeTTSWithEndpoint + EdgeTTSWithHTTPClient（Wave8 修复）
//     → 上述 Provider 的云请求/响应解析均可用 httptest 在不打真实网络的前提下单测。

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/media"
)

// newTestClient 返回一个普通 *http.Client（无特殊超时），用于注入到 Provider，
// 配合 httptest.Server 的 baseURL 使请求落到本地 mock 而非真实网络。
func newTestClient() *http.Client {
	return &http.Client{Timeout: 5 * time.Second}
}

// ---------------------------------------------------------------------------
// OpenAISTT：请求构造 + 响应解析 + 错误路径（httptest 注入）
// ---------------------------------------------------------------------------

func TestOpenAISTT_Transcribe_RequestAndResponse(t *testing.T) {
	var gotMethod, gotPath, gotAuth, gotContentType string
	var gotModel, gotLanguage, gotPrompt, gotRespFormat string
	var gotFileName string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")

		// 解析 multipart 表单，验证字段构造正确
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			t.Errorf("解析 multipart 失败: %v", err)
		}
		gotModel = r.FormValue("model")
		gotLanguage = r.FormValue("language")
		gotPrompt = r.FormValue("prompt")
		gotRespFormat = r.FormValue("response_format")
		if r.MultipartForm != nil && len(r.MultipartForm.File["file"]) > 0 {
			gotFileName = r.MultipartForm.File["file"][0].Filename
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"你好世界","language":"zh","duration":3.5}`)
	}))
	defer srv.Close()

	stt := NewOpenAISTT("sk-test-key", "whisper-1",
		STTWithBaseURL(srv.URL),
		STTWithHTTPClient(newTestClient()),
	)
	res, err := stt.Transcribe(context.Background(), []byte("RIFFfakeaudio"),
		TranscribeOptions{Language: "zh", Format: FormatMP3, Prompt: "术语表"})
	if err != nil {
		t.Fatalf("Transcribe 失败: %v", err)
	}

	// 请求构造断言
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if gotPath != "/audio/transcriptions" {
		t.Errorf("path = %q, want /audio/transcriptions", gotPath)
	}
	if gotAuth != "Bearer sk-test-key" {
		t.Errorf("Authorization = %q, want Bearer sk-test-key", gotAuth)
	}
	if !strings.HasPrefix(gotContentType, "multipart/form-data") {
		t.Errorf("Content-Type = %q, want multipart/form-data", gotContentType)
	}
	if gotModel != "whisper-1" {
		t.Errorf("model field = %q, want whisper-1", gotModel)
	}
	if gotLanguage != "zh" {
		t.Errorf("language field = %q, want zh", gotLanguage)
	}
	if gotPrompt != "术语表" {
		t.Errorf("prompt field = %q, want 术语表", gotPrompt)
	}
	if gotRespFormat != "verbose_json" {
		t.Errorf("response_format = %q, want verbose_json", gotRespFormat)
	}
	if gotFileName != "audio.mp3" {
		t.Errorf("file name = %q, want audio.mp3 (随 Format 变化)", gotFileName)
	}

	// 响应解析断言
	if res.Text != "你好世界" {
		t.Errorf("Text = %q, want 你好世界", res.Text)
	}
	if res.Language != "zh" {
		t.Errorf("Language = %q, want zh", res.Language)
	}
	if res.Duration != 3.5 {
		t.Errorf("Duration = %v, want 3.5", res.Duration)
	}
}

func TestOpenAISTT_Transcribe_DefaultExtWhenNoFormat(t *testing.T) {
	var gotFileName string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(10 << 20)
		if r.MultipartForm != nil && len(r.MultipartForm.File["file"]) > 0 {
			gotFileName = r.MultipartForm.File["file"][0].Filename
		}
		_, _ = io.WriteString(w, `{"text":"ok"}`)
	}))
	defer srv.Close()

	stt := NewOpenAISTT("sk-x", "whisper-1", STTWithBaseURL(srv.URL), STTWithHTTPClient(newTestClient()))
	if _, err := stt.Transcribe(context.Background(), []byte("x"), TranscribeOptions{}); err != nil {
		t.Fatalf("Transcribe 失败: %v", err)
	}
	// 未指定 Format 时扩展名默认 wav
	if gotFileName != "audio.wav" {
		t.Errorf("默认文件名 = %q, want audio.wav", gotFileName)
	}
}

func TestOpenAISTT_Transcribe_EmptyAudio(t *testing.T) {
	stt := NewOpenAISTT("sk-x", "whisper-1")
	if _, err := stt.Transcribe(context.Background(), nil, TranscribeOptions{}); err == nil {
		t.Error("空音频应返回错误")
	}
	if _, err := stt.Transcribe(context.Background(), []byte{}, TranscribeOptions{}); err == nil {
		t.Error("零长音频应返回错误")
	}
}

func TestOpenAISTT_Transcribe_Non200Sanitized(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		// 错误体里塞入敏感信息：URL、key、bearer，验证脱敏
		_, _ = io.WriteString(w, `{"api_key":"sk-abcdefghij1234567890zz","detail":"see https://internal.example.com/x","auth":"Bearer abcdef1234567890"}`)
	}))
	defer srv.Close()

	stt := NewOpenAISTT("sk-x", "whisper-1", STTWithBaseURL(srv.URL), STTWithHTTPClient(newTestClient()))
	_, err := stt.Transcribe(context.Background(), []byte("x"), TranscribeOptions{})
	if err == nil {
		t.Fatal("非 200 应返回错误")
	}
	msg := err.Error()
	if !strings.Contains(msg, "401") {
		t.Errorf("错误应包含状态码 401, got %q", msg)
	}
	// 脱敏断言：原始 URL / key / bearer 不应出现
	if strings.Contains(msg, "https://internal.example.com") {
		t.Errorf("内部 URL 未脱敏: %q", msg)
	}
	if strings.Contains(msg, "sk-abcdefghij1234567890zz") {
		t.Errorf("api key 未脱敏: %q", msg)
	}
	if strings.Contains(msg, "Bearer abcdef1234567890") {
		t.Errorf("bearer token 未脱敏: %q", msg)
	}
}

func TestOpenAISTT_Transcribe_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{not-json`)
	}))
	defer srv.Close()

	stt := NewOpenAISTT("sk-x", "whisper-1", STTWithBaseURL(srv.URL), STTWithHTTPClient(newTestClient()))
	if _, err := stt.Transcribe(context.Background(), []byte("x"), TranscribeOptions{}); err == nil {
		t.Error("非法 JSON 响应应返回解析错误")
	}
}

func TestOpenAISTT_DefaultModel(t *testing.T) {
	stt := NewOpenAISTT("sk-x", "")
	// model 为空时默认 whisper-1；通过请求体间接验证
	if stt.model != "whisper-1" {
		t.Errorf("空 model 应默认 whisper-1, got %q", stt.model)
	}
}

// ---------------------------------------------------------------------------
// OpenAITTS：请求构造 + 响应（二进制音频）+ 默认值 + 错误
// ---------------------------------------------------------------------------

func TestOpenAITTS_Synthesize_RequestAndResponse(t *testing.T) {
	var gotBody map[string]any
	var gotPath, gotAuth string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("FAKE-MP3-BYTES"))
	}))
	defer srv.Close()

	tts := NewOpenAITTS("sk-tts", "tts-1-hd",
		TTSWithBaseURL(srv.URL), TTSWithHTTPClient(newTestClient()))
	res, err := tts.Synthesize(context.Background(), "你好，世界",
		SynthesizeOptions{Voice: "nova", Format: FormatOGG, Speed: 1.5})
	if err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}

	if gotPath != "/audio/speech" {
		t.Errorf("path = %q, want /audio/speech", gotPath)
	}
	if gotAuth != "Bearer sk-tts" {
		t.Errorf("auth = %q, want Bearer sk-tts", gotAuth)
	}
	if gotBody["model"] != "tts-1-hd" {
		t.Errorf("model = %v, want tts-1-hd", gotBody["model"])
	}
	if gotBody["input"] != "你好，世界" {
		t.Errorf("input = %v", gotBody["input"])
	}
	if gotBody["voice"] != "nova" {
		t.Errorf("voice = %v, want nova", gotBody["voice"])
	}
	if gotBody["response_format"] != "ogg" {
		t.Errorf("response_format = %v, want ogg", gotBody["response_format"])
	}
	if gotBody["speed"] != 1.5 {
		t.Errorf("speed = %v, want 1.5", gotBody["speed"])
	}

	if string(res.Audio) != "FAKE-MP3-BYTES" {
		t.Errorf("Audio = %q", res.Audio)
	}
	// 关键一致性：返回的 Format 应是请求里的 ogg，而不是固定 mp3
	if res.Format != FormatOGG {
		t.Errorf("Format = %q, want ogg", res.Format)
	}
	if res.Size != len("FAKE-MP3-BYTES") {
		t.Errorf("Size = %d, want %d", res.Size, len("FAKE-MP3-BYTES"))
	}
}

func TestOpenAITTS_Synthesize_Defaults(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = w.Write([]byte("a"))
	}))
	defer srv.Close()

	tts := NewOpenAITTS("sk-x", "", TTSWithBaseURL(srv.URL), TTSWithHTTPClient(newTestClient()))
	res, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	// 默认值断言：voice=alloy, format=mp3, speed=1.0, model=tts-1
	if gotBody["voice"] != "alloy" {
		t.Errorf("默认 voice = %v, want alloy", gotBody["voice"])
	}
	if gotBody["response_format"] != "mp3" {
		t.Errorf("默认 format = %v, want mp3", gotBody["response_format"])
	}
	if gotBody["speed"] != 1.0 {
		t.Errorf("默认 speed = %v, want 1.0", gotBody["speed"])
	}
	if gotBody["model"] != "tts-1" {
		t.Errorf("默认 model = %v, want tts-1", gotBody["model"])
	}
	if res.Format != FormatMP3 {
		t.Errorf("默认 Format = %q, want mp3", res.Format)
	}
}

func TestOpenAITTS_Synthesize_NegativeSpeedFallback(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = w.Write([]byte("a"))
	}))
	defer srv.Close()

	tts := NewOpenAITTS("sk-x", "tts-1", TTSWithBaseURL(srv.URL), TTSWithHTTPClient(newTestClient()))
	// 负 speed 应回退到 1.0
	if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{Speed: -3}); err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	if gotBody["speed"] != 1.0 {
		t.Errorf("负 speed 应回退 1.0, got %v", gotBody["speed"])
	}
}

func TestOpenAITTS_Synthesize_EmptyText(t *testing.T) {
	tts := NewOpenAITTS("sk-x", "tts-1")
	if _, err := tts.Synthesize(context.Background(), "", SynthesizeOptions{}); err == nil {
		t.Error("空文本应返回错误")
	}
}

func TestOpenAITTS_Synthesize_Non200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, "rate limited")
	}))
	defer srv.Close()

	tts := NewOpenAITTS("sk-x", "tts-1", TTSWithBaseURL(srv.URL), TTSWithHTTPClient(newTestClient()))
	_, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Errorf("应返回含 429 的错误, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// MiniMaxTTS：请求构造 + hex 解码 + base_resp 业务错误码 + 边界
// ---------------------------------------------------------------------------

func TestMiniMaxTTS_Synthesize_Success(t *testing.T) {
	var gotURL, gotAuth string
	var gotBody map[string]any
	wantAudio := []byte{0xde, 0xad, 0xbe, 0xef}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotURL = r.URL.String()
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		resp := map[string]any{
			"data":      map[string]any{"audio": hex.EncodeToString(wantAudio)},
			"base_resp": map[string]any{"status_code": 0, "status_msg": "success"},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("api-key", "group-123",
		WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()))
	res, err := tts.Synthesize(context.Background(), "测试文本",
		SynthesizeOptions{Voice: "female-tianmei", Speed: 1.2})
	if err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}

	// URL 带 GroupId query
	if !strings.Contains(gotURL, "/v1/t2a_v2?GroupId=group-123") {
		t.Errorf("URL = %q, 应含 /v1/t2a_v2?GroupId=group-123", gotURL)
	}
	if gotAuth != "Bearer api-key" {
		t.Errorf("auth = %q, want Bearer api-key", gotAuth)
	}
	if gotBody["text"] != "测试文本" {
		t.Errorf("text = %v", gotBody["text"])
	}
	vs, _ := gotBody["voice_setting"].(map[string]any)
	if vs == nil || vs["voice_id"] != "female-tianmei" {
		t.Errorf("voice_id = %v, want female-tianmei", vs)
	}
	if vs["speed"] != 1.2 {
		t.Errorf("speed = %v, want 1.2", vs["speed"])
	}

	// hex 解码正确
	if string(res.Audio) != string(wantAudio) {
		t.Errorf("Audio = %x, want %x", res.Audio, wantAudio)
	}
	if res.Format != FormatMP3 {
		t.Errorf("Format = %q, want mp3", res.Format)
	}
	if res.Size != len(wantAudio) {
		t.Errorf("Size = %d, want %d", res.Size, len(wantAudio))
	}
}

func TestMiniMaxTTS_Synthesize_DefaultVoiceWhenEmpty(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"audio": hex.EncodeToString([]byte("x"))},
			"base_resp": map[string]any{"status_code": 0},
		})
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("k", "g", WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()))
	// Voice 为空 / 仅空白 → 默认 male-qn-qingse
	if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{Voice: "   "}); err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	vs, _ := gotBody["voice_setting"].(map[string]any)
	if vs["voice_id"] != "male-qn-qingse" {
		t.Errorf("默认 voice_id = %v, want male-qn-qingse", vs["voice_id"])
	}
}

func TestMiniMaxTTS_Synthesize_MissingCreds(t *testing.T) {
	cases := []struct {
		name, key, group string
	}{
		{"both empty", "", ""},
		{"empty key", "", "g"},
		{"empty group", "k", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tts := NewMiniMaxTTS(c.key, c.group)
			if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{}); err == nil {
				t.Error("缺少凭据应返回错误")
			}
		})
	}
}

func TestMiniMaxTTS_Synthesize_BlankText(t *testing.T) {
	tts := NewMiniMaxTTS("k", "g")
	// 纯空白文本应被 TrimSpace 后判空
	if _, err := tts.Synthesize(context.Background(), "   \n\t ", SynthesizeOptions{}); err == nil {
		t.Error("纯空白文本应返回错误")
	}
}

func TestMiniMaxTTS_Synthesize_BusinessError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// HTTP 200 但 base_resp.status_code != 0 → 业务错误
		_ = json.NewEncoder(w).Encode(map[string]any{
			"base_resp": map[string]any{"status_code": 1004, "status_msg": "余额不足"},
		})
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("k", "g", WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()))
	_, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err == nil {
		t.Fatal("业务错误码应返回错误")
	}
	if !strings.Contains(err.Error(), "1004") || !strings.Contains(err.Error(), "余额不足") {
		t.Errorf("错误应含业务码与消息, got %v", err)
	}
}

func TestMiniMaxTTS_Synthesize_EmptyAudio(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// status_code=0 但 audio 为空 → 应报 empty audio
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"audio": ""},
			"base_resp": map[string]any{"status_code": 0},
		})
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("k", "g", WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()))
	if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{}); err == nil {
		t.Error("空音频应返回错误")
	}
}

func TestMiniMaxTTS_Synthesize_BadHex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// audio 非合法 hex → 解码失败
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"audio": "zzzz-not-hex"},
			"base_resp": map[string]any{"status_code": 0},
		})
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("k", "g", WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()))
	if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{}); err == nil {
		t.Error("非法 hex 应返回解码错误")
	}
}

func TestMiniMaxTTS_Synthesize_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, "forbidden")
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("k", "g", WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()))
	_, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Errorf("应返回含 403 的错误, got %v", err)
	}
}

func TestMiniMaxTTS_Synthesize_BaseURLTrailingSlash(t *testing.T) {
	var gotURL string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotURL = r.URL.Path
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"audio": hex.EncodeToString([]byte("x"))},
			"base_resp": map[string]any{"status_code": 0},
		})
	}))
	defer srv.Close()

	// baseURL 带尾斜杠，应被 TrimRight 处理，不出现双斜杠
	tts := NewMiniMaxTTS("k", "g", WithMiniMaxBaseURL(srv.URL+"/"), WithMiniMaxHTTPClient(newTestClient()))
	if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{}); err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	if strings.Contains(gotURL, "//v1") {
		t.Errorf("尾斜杠未处理，path 出现双斜杠: %q", gotURL)
	}
	if gotURL != "/v1/t2a_v2" {
		t.Errorf("path = %q, want /v1/t2a_v2", gotURL)
	}
}

// ---------------------------------------------------------------------------
// MiniMax model 选项 + Voices/Formats 元数据
// ---------------------------------------------------------------------------

func TestMiniMaxTTS_ModelOption(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"audio": hex.EncodeToString([]byte("x"))},
			"base_resp": map[string]any{"status_code": 0},
		})
	}))
	defer srv.Close()

	tts := NewMiniMaxTTS("k", "g",
		WithMiniMaxBaseURL(srv.URL), WithMiniMaxHTTPClient(newTestClient()),
		WithMiniMaxModel("speech-02-hd"))
	if _, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{}); err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}
	if gotBody["model"] != "speech-02-hd" {
		t.Errorf("model = %v, want speech-02-hd", gotBody["model"])
	}
}

func TestMiniMaxTTS_Metadata(t *testing.T) {
	tts := NewMiniMaxTTS("k", "g")
	if tts.Name() != "minimax-tts" {
		t.Errorf("Name = %q", tts.Name())
	}
	if len(tts.Voices()) == 0 {
		t.Error("Voices 不应为空")
	}
	if got := tts.SupportedFormats(); len(got) != 1 || got[0] != FormatMP3 {
		t.Errorf("SupportedFormats = %v, want [mp3]", got)
	}
}

// ---------------------------------------------------------------------------
// 异步音频：终态/失败态/Submit 错误/ctx 取消
// ---------------------------------------------------------------------------

// scriptedAudio 是脚本化异步 Provider：按 states 顺序逐次返回。
type scriptedAudio struct {
	submitErr error
	states    []AudioTaskStatus
	pollErrAt int // 在第几次 Poll 返回错误（0 表示不出错），从 1 计数
	idx       int
}

func (s *scriptedAudio) Name() string { return "scripted-audio" }
func (s *scriptedAudio) Submit(_ context.Context, _ AudioRequest) (string, error) {
	if s.submitErr != nil {
		return "", s.submitErr
	}
	return "task-x", nil
}
func (s *scriptedAudio) Poll(_ context.Context, _ string) (AudioTaskStatus, error) {
	s.idx++
	if s.pollErrAt != 0 && s.idx == s.pollErrAt {
		return AudioTaskStatus{}, errors.New("poll boom")
	}
	if s.idx-1 < len(s.states) {
		return s.states[s.idx-1], nil
	}
	return s.states[len(s.states)-1], nil
}

func TestSubmitAndWaitAudio_FailedTerminalPropagates(t *testing.T) {
	// 任务到达 failed 终态：WaitFor 视为 done，无 err，但最终状态应为 failed
	p := &scriptedAudio{states: []AudioTaskStatus{
		{TaskID: "task-x", State: media.TaskRunning},
		{TaskID: "task-x", State: media.TaskFailed, Error: "上游崩了"},
	}}
	st, err := SubmitAndWaitAudio(context.Background(), p, AudioRequest{Text: "x"}, time.Millisecond)
	if err != nil {
		t.Fatalf("失败终态不应返回 waitErr（done=true）, got %v", err)
	}
	if st.State != media.TaskFailed {
		t.Errorf("State = %q, want failed", st.State)
	}
	if st.Error != "上游崩了" {
		t.Errorf("Error = %q, want 上游崩了", st.Error)
	}
}

func TestSubmitAndWaitAudio_PollError(t *testing.T) {
	p := &scriptedAudio{
		states:    []AudioTaskStatus{{State: media.TaskRunning}},
		pollErrAt: 1,
	}
	_, err := SubmitAndWaitAudio(context.Background(), p, AudioRequest{Text: "x"}, time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "poll boom") {
		t.Errorf("Poll 错误应传播, got %v", err)
	}
}

func TestSubmitAndWaitAudio_SubmitError(t *testing.T) {
	p := &scriptedAudio{submitErr: errors.New("submit fail")}
	st, err := SubmitAndWaitAudio(context.Background(), p, AudioRequest{Text: "x"}, time.Millisecond)
	if err == nil {
		t.Fatal("Submit 错误应传播")
	}
	// Submit 失败时应返回空状态
	if st.State != "" {
		t.Errorf("Submit 失败应返回空状态, got %q", st.State)
	}
}

func TestSubmitAndWaitAudio_CtxCanceled(t *testing.T) {
	// 永不终态 → 依赖 ctx 取消退出
	p := &scriptedAudio{states: []AudioTaskStatus{{State: media.TaskRunning}}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 立即取消
	_, err := SubmitAndWaitAudio(ctx, p, AudioRequest{Text: "x"}, 50*time.Millisecond)
	// 第一次 poll 先执行（running，非终态），随后进入等待时遇到已取消 ctx
	if err == nil {
		t.Error("ctx 取消应返回错误")
	}
}

// ---------------------------------------------------------------------------
// WakeWordDetector：并发竞态 + 边界
// ---------------------------------------------------------------------------

func TestWakeWordDetector_ConcurrentRace(t *testing.T) {
	d := NewWakeWordDetector([]string{"河蟹"}, func(word, text string) {})
	var wg sync.WaitGroup
	stop := make(chan struct{})

	// 读 goroutine
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					d.Detect("你好河蟹")
					_ = d.Words()
				}
			}
		}()
	}
	// 写 goroutine
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					d.AddWord("w")
					d.RemoveWord("w")
					d.SetEnabled(n%2 == 0)
				}
			}
		}(i)
	}

	time.Sleep(80 * time.Millisecond)
	close(stop)
	wg.Wait()
	// 仅需在 -race 下无数据竞争即通过；此处断言 Words 返回的是拷贝（修改不影响内部）
	w := d.Words()
	if len(w) > 0 {
		w[0] = "tampered"
	}
	for _, x := range d.Words() {
		if x == "tampered" {
			t.Error("Words() 应返回内部切片的拷贝，外部修改不应影响内部状态")
		}
	}
}

func TestWakeWordDetector_EmptyWords(t *testing.T) {
	d := NewWakeWordDetector(nil, nil)
	if got := d.Detect("任意文本"); got != "" {
		t.Errorf("空唤醒词列表应不匹配, got %q", got)
	}
	if got := d.Detect(""); got != "" {
		t.Errorf("空文本应不匹配, got %q", got)
	}
}

func TestWakeWordDetector_RemoveNonexistent(t *testing.T) {
	d := NewWakeWordDetector([]string{"a", "b"}, nil)
	d.RemoveWord("不存在")
	if len(d.Words()) != 2 {
		t.Errorf("移除不存在的词不应改变列表, got %d", len(d.Words()))
	}
}

func TestWakeWordDetector_FirstMatchWins(t *testing.T) {
	// 文本同时含两个词，应返回 words 列表中靠前的那个
	d := NewWakeWordDetector([]string{"小蟹", "河蟹"}, nil)
	got := d.Detect("河蟹和小蟹都在")
	if got != "小蟹" {
		t.Errorf("应返回列表中第一个匹配的词 小蟹, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// SSML/XML 注入转义（escapeXML 经 Azure/Edge Synthesize 的本地分支不可直接到达，
// 但 escapeXML 是包内函数，可直接测；这里测其安全语义）
// ---------------------------------------------------------------------------

func TestEscapeXML_InjectionChars(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{`<script>`, `&lt;script&gt;`},
		{`a & b`, `a &amp; b`},
		{`'quote'`, `&apos;quote&apos;`},
		{`"dq"`, `&quot;dq&quot;`},
		{`</voice><voice name='evil'>`, `&lt;/voice&gt;&lt;voice name=&apos;evil&apos;&gt;`},
		{`正常中文`, `正常中文`},
	}
	for _, c := range cases {
		if got := escapeXML(c.in); got != c.want {
			t.Errorf("escapeXML(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Service：超长文本 / Unicode 透传，配合 mock Provider
// ---------------------------------------------------------------------------

type passthroughTTS struct{ gotText string }

func (p *passthroughTTS) Name() string { return "passthrough" }
func (p *passthroughTTS) Synthesize(_ context.Context, text string, _ SynthesizeOptions) (*SynthesizeResult, error) {
	p.gotText = text
	return &SynthesizeResult{Audio: []byte(text), Format: FormatMP3, Size: len(text)}, nil
}
func (p *passthroughTTS) Voices() []VoiceInfo             { return nil }
func (p *passthroughTTS) SupportedFormats() []AudioFormat { return []AudioFormat{FormatMP3} }

func TestService_Synthesize_LongUnicode(t *testing.T) {
	p := &passthroughTTS{}
	svc := NewService(nil, p)
	long := strings.Repeat("你好🎉emoji混合ABC123 ", 5000) // 超长 + Unicode + emoji
	res, err := svc.Synthesize(context.Background(), long, SynthesizeOptions{})
	if err != nil {
		t.Fatalf("超长 Unicode 合成失败: %v", err)
	}
	if p.gotText != long {
		t.Error("文本未原样透传到 Provider")
	}
	if res.Size != len(long) {
		t.Errorf("Size = %d, want %d", res.Size, len(long))
	}
}

// ---------------------------------------------------------------------------
// ChainedTTS：BuildMultiTTS 构造逻辑 + nil 处理
// ---------------------------------------------------------------------------

func TestBuildMultiTTS_RecognizedAndSkipped(t *testing.T) {
	cfg := []MultiTTSConfig{
		{Provider: "edge-tts"},                               // 无需 key
		{Provider: "edge"},                                   // 别名
		{Provider: "openai", APIKey: "sk-x"},                 // 有 key → 接受
		{Provider: "openai"},                                 // 无 key → 跳过
		{Provider: "minimax", APIKey: "k", GroupID: "g"},     // 全 → 接受
		{Provider: "minimax", APIKey: "k"},                   // 缺 group → 跳过
		{Provider: "azure", APIKey: "k", Region: "eastasia"}, // 全 → 接受
		{Provider: "azure", APIKey: "k"},                     // 缺 region → 跳过
		{Provider: "unknown-xyz"},                            // 未识别 → 跳过
	}
	chain := BuildMultiTTS(cfg)
	if chain == nil {
		t.Fatal("应构造出非空 chain")
	}
	// 期望接受：edge, edge, openai(key), minimax(full), azure(full) = 5 个
	if got := len(chain.providers); got != 5 {
		t.Errorf("识别出的 provider 数 = %d, want 5", got)
	}
}

func TestBuildMultiTTS_AllSkippedReturnsNil(t *testing.T) {
	cfg := []MultiTTSConfig{
		{Provider: "openai"},               // 无 key
		{Provider: "minimax", APIKey: "k"}, // 缺 group
		{Provider: "unknown"},
	}
	if chain := BuildMultiTTS(cfg); chain != nil {
		t.Errorf("全部跳过应返回 nil, got %+v", chain)
	}
}

func TestBuildMultiTTS_EmptyReturnsNil(t *testing.T) {
	if chain := BuildMultiTTS(nil); chain != nil {
		t.Error("空配置应返回 nil")
	}
	if chain := BuildMultiTTS([]MultiTTSConfig{}); chain != nil {
		t.Error("空 slice 应返回 nil")
	}
}

func TestChainedTTS_NameEmpty(t *testing.T) {
	c := NewChainedTTS()
	if c.Name() != "chained-tts(empty)" {
		t.Errorf("空 chain Name = %q", c.Name())
	}
}

func TestChainedTTS_NameWithProviders(t *testing.T) {
	c := NewChainedTTS(&fakeTTS{name: "p1"}, &fakeTTS{name: "p2"})
	name := c.Name()
	if !strings.Contains(name, "p1") || !strings.Contains(name, "p2") {
		t.Errorf("Name 应含各 provider 名, got %q", name)
	}
}

// ---------------------------------------------------------------------------
// AudioFormat 常量 sanity
// ---------------------------------------------------------------------------

func TestAudioFormat_Constants(t *testing.T) {
	pairs := map[AudioFormat]string{
		FormatWAV: "wav", FormatMP3: "mp3", FormatOGG: "ogg",
		FormatFLAC: "flac", FormatPCM: "pcm",
	}
	for f, s := range pairs {
		if string(f) != s {
			t.Errorf("AudioFormat %v != %q", f, s)
		}
	}
}

// ---------------------------------------------------------------------------
// 回归: AzureSTT/AzureTTS 无 HTTP client 与 baseURL 注入点，云请求构造/响应解析无法单测（bug #1）
//
// 修复前：AzureSTT/AzureTTS 把 endpoint 域名（由 region 拼成完整 https 域名）与
// http.Client 硬编码进构造器，没有 WithBaseURL/WithHTTPClient 注入点，导致网络交互
// 无法在不打真实网络的前提下 mock —— 请求构造（method/path/header/SSML body）与响应解析
// 全部不可单测。修复后新增 AzureSTTWithBaseURL/AzureSTTWithHTTPClient 与
// AzureTTSWithBaseURL/AzureTTSWithHTTPClient additive option，端点优先用注入的 baseURL。
// 本测试用 httptest 完整断言请求构造与响应解析，永久钉死该注入契约，不得删除或弱化。
// ---------------------------------------------------------------------------

func TestAzureSTT_Transcribe_RequestAndResponse_Injected(t *testing.T) {
	var gotMethod, gotPath, gotKey, gotContentType, gotQuery string
	var gotAudio []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotKey = r.Header.Get("Ocp-Apim-Subscription-Key")
		gotContentType = r.Header.Get("Content-Type")
		gotAudio, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"RecognitionStatus":"Success","DisplayText":"你好世界","Duration":35000000,"NBest":[{"Confidence":0.92,"Display":"你好世界"}]}`)
	}))
	defer srv.Close()

	stt := NewAzureSTT("azure-key", "eastasia",
		AzureSTTWithBaseURL(srv.URL),
		AzureSTTWithHTTPClient(newTestClient()),
	)
	res, err := stt.Transcribe(context.Background(), []byte("RIFFfakeaudio"),
		TranscribeOptions{Language: "zh-CN", Format: FormatMP3})
	if err != nil {
		t.Fatalf("Transcribe 失败: %v", err)
	}

	// 请求构造断言
	if gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", gotMethod)
	}
	if gotPath != "/speech/recognition/conversation/cognitiveservices/v1" {
		t.Errorf("path = %q", gotPath)
	}
	if !strings.Contains(gotQuery, "language=zh-CN") || !strings.Contains(gotQuery, "format=detailed") {
		t.Errorf("query = %q, 应含 language=zh-CN & format=detailed", gotQuery)
	}
	if gotKey != "azure-key" {
		t.Errorf("Ocp-Apim-Subscription-Key = %q, want azure-key", gotKey)
	}
	if gotContentType != "audio/mpeg" {
		t.Errorf("Content-Type = %q, want audio/mpeg (随 Format=mp3)", gotContentType)
	}
	if string(gotAudio) != "RIFFfakeaudio" {
		t.Errorf("上传音频 = %q, want RIFFfakeaudio", gotAudio)
	}

	// 响应解析断言
	if res.Text != "你好世界" {
		t.Errorf("Text = %q, want 你好世界", res.Text)
	}
	if res.Language != "zh-CN" {
		t.Errorf("Language = %q, want zh-CN", res.Language)
	}
	if res.Duration != 3.5 { // 35000000 ticks / 1e7 = 3.5s
		t.Errorf("Duration = %v, want 3.5", res.Duration)
	}
	if res.Confidence != 0.92 {
		t.Errorf("Confidence = %v, want 0.92", res.Confidence)
	}
}

func TestAzureSTT_Transcribe_Non200_Injected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, "invalid subscription key")
	}))
	defer srv.Close()

	stt := NewAzureSTT("bad", "eastasia", AzureSTTWithBaseURL(srv.URL), AzureSTTWithHTTPClient(newTestClient()))
	_, err := stt.Transcribe(context.Background(), []byte("x"), TranscribeOptions{})
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Errorf("非 200 应返回含 401 的错误, got %v", err)
	}
}

func TestAzureTTS_Synthesize_RequestAndResponse_Injected(t *testing.T) {
	var gotPath, gotKey, gotContentType, gotOutFormat, gotBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("Ocp-Apim-Subscription-Key")
		gotContentType = r.Header.Get("Content-Type")
		gotOutFormat = r.Header.Get("X-Microsoft-OutputFormat")
		raw, _ := io.ReadAll(r.Body)
		gotBody = string(raw)
		_, _ = w.Write([]byte("FAKE-AZURE-MP3"))
	}))
	defer srv.Close()

	tts := NewAzureTTS("azure-key", "eastasia",
		AzureTTSWithBaseURL(srv.URL), AzureTTSWithHTTPClient(newTestClient()))
	// 含 SSML 注入字符，验证 escapeXML 在真实请求体里生效
	res, err := tts.Synthesize(context.Background(), "你好<evil>",
		SynthesizeOptions{Voice: "zh-CN-YunxiNeural", Format: FormatMP3})
	if err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}

	if gotPath != "/cognitiveservices/v1" {
		t.Errorf("path = %q, want /cognitiveservices/v1", gotPath)
	}
	if gotKey != "azure-key" {
		t.Errorf("subscription key = %q", gotKey)
	}
	if gotContentType != "application/ssml+xml" {
		t.Errorf("Content-Type = %q, want application/ssml+xml", gotContentType)
	}
	if gotOutFormat != "audio-16khz-128kbitrate-mono-mp3" {
		t.Errorf("X-Microsoft-OutputFormat = %q", gotOutFormat)
	}
	// SSML body 应包含转义后的注入字符 & 选定的 voice
	if !strings.Contains(gotBody, "zh-CN-YunxiNeural") {
		t.Errorf("SSML 未含 voice: %q", gotBody)
	}
	if !strings.Contains(gotBody, "&lt;evil&gt;") || strings.Contains(gotBody, "<evil>") {
		t.Errorf("SSML 注入字符未转义: %q", gotBody)
	}
	if string(res.Audio) != "FAKE-AZURE-MP3" || res.Size != len("FAKE-AZURE-MP3") {
		t.Errorf("响应音频解析错误: Audio=%q Size=%d", res.Audio, res.Size)
	}
	if res.Format != FormatMP3 {
		t.Errorf("Format = %q, want mp3", res.Format)
	}
}

func TestAzureTTS_Synthesize_Non200_Injected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, "throttled")
	}))
	defer srv.Close()

	tts := NewAzureTTS("k", "eastasia", AzureTTSWithBaseURL(srv.URL), AzureTTSWithHTTPClient(newTestClient()))
	_, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Errorf("应返回含 429 的错误, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// 回归: EdgeTTS 端点硬编码且无 client/baseURL 注入点，合成请求/响应解析无法单测（bug #2）
//
// 修复前：EdgeTTS 把 endpoint（https://eastus.api.speech.microsoft.com/...）与 http.Client
// 硬编码进实现，没有注入点，合成请求构造与响应解析无法 mock。修复后新增
// EdgeTTSWithEndpoint/EdgeTTSWithHTTPClient additive option。本测试用 httptest 断言
// 请求构造（SSML rate/voice 转义、header）与响应解析，永久钉死，不得删除或弱化。
// ---------------------------------------------------------------------------

func TestEdgeTTS_Synthesize_RequestAndResponse_Injected(t *testing.T) {
	var gotContentType, gotOutFormat, gotBody string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentType = r.Header.Get("Content-Type")
		gotOutFormat = r.Header.Get("X-Microsoft-OutputFormat")
		raw, _ := io.ReadAll(r.Body)
		gotBody = string(raw)
		_, _ = w.Write([]byte("FAKE-EDGE-MP3"))
	}))
	defer srv.Close()

	tts := NewEdgeTTS(
		EdgeTTSWithEndpoint(srv.URL),
		EdgeTTSWithHTTPClient(newTestClient()),
		EdgeTTSWithVoice("zh-CN-YunjianNeural"),
	)
	res, err := tts.Synthesize(context.Background(), "你好<inject>",
		SynthesizeOptions{Speed: 1.5})
	if err != nil {
		t.Fatalf("Synthesize 失败: %v", err)
	}

	if gotContentType != "application/ssml+xml" {
		t.Errorf("Content-Type = %q, want application/ssml+xml", gotContentType)
	}
	if gotOutFormat != "audio-16khz-128kbitrate-mono-mp3" {
		t.Errorf("X-Microsoft-OutputFormat = %q", gotOutFormat)
	}
	// SSML body 应含默认 voice、speed=1.5 对应 rate=+50%、且注入字符被转义
	if !strings.Contains(gotBody, "zh-CN-YunjianNeural") {
		t.Errorf("SSML 未含 voice: %q", gotBody)
	}
	if !strings.Contains(gotBody, "+50%") {
		t.Errorf("SSML rate 计算错误（speed 1.5 应 +50%%）: %q", gotBody)
	}
	if !strings.Contains(gotBody, "&lt;inject&gt;") || strings.Contains(gotBody, "<inject>") {
		t.Errorf("SSML 注入字符未转义: %q", gotBody)
	}
	if string(res.Audio) != "FAKE-EDGE-MP3" || res.Size != len("FAKE-EDGE-MP3") {
		t.Errorf("响应音频解析错误: Audio=%q Size=%d", res.Audio, res.Size)
	}
	if res.Format != FormatMP3 {
		t.Errorf("Format = %q, want mp3", res.Format)
	}
}

func TestEdgeTTS_Synthesize_Non200_Injected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, "down")
	}))
	defer srv.Close()

	tts := NewEdgeTTS(EdgeTTSWithEndpoint(srv.URL), EdgeTTSWithHTTPClient(newTestClient()))
	_, err := tts.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Errorf("应返回含 503 的错误, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// 回归: MiniMaxTTS 默认使用裸 &http.Client 而非 toolkit httpx.RawClient（bug #3）
//
// 修复前 NewMiniMaxTTS 默认 client 是裸 &http.Client{Timeout: 30s}，Transport 为 nil
// （会回退到 http.DefaultTransport），违反同包 OpenAI/Azure/Edge 既定的 httpx.RawClient
// 复用约定。修复后默认走 httpx.RawClient，Transport 被显式设置为非 nil。
// 本断言钉死"默认 client 必须来自 httpx.RawClient（Transport != nil）"，不得删除或弱化。
// ---------------------------------------------------------------------------

func TestMiniMaxTTS_DefaultClientUsesHTTPX(t *testing.T) {
	tts := NewMiniMaxTTS("k", "g")
	if tts.client == nil {
		t.Fatal("默认 client 不应为 nil")
	}
	if tts.client == http.DefaultClient {
		t.Error("默认 client 不应是 http.DefaultClient")
	}
	// httpx.RawClient 会显式设置 Transport；裸 &http.Client{} 的 Transport 为 nil。
	if tts.client.Transport == nil {
		t.Error("默认 client 的 Transport 为 nil，说明仍是裸 &http.Client 而非 httpx.RawClient")
	}
}

// ---------------------------------------------------------------------------
// 回归: Service.Synthesize 仅靠 text=="" 判空，纯空白文本会被透传到上游 Provider（bug #4）
//
// 修复前 Service.Synthesize 只用 text=="" 判空，纯空白（空格/换行/Tab）文本会绕过校验
// 透传到上游 Provider，浪费一次网络/计费调用。修复后改用 strings.TrimSpace 判空，
// 与各 Provider（MiniMax 已 TrimSpace）的判定标准统一。本测试钉死该不变量，断言
// 空白文本在 Service 层被拒绝且不透传到 Provider，不得删除或弱化。
// ---------------------------------------------------------------------------

func TestService_Synthesize_BlankWhitespaceRejected(t *testing.T) {
	cases := []string{"", "   ", "\n", "\t", "  \n\t ", "\r\n  \t"}
	for _, text := range cases {
		p := &passthroughTTS{}
		svc := NewService(nil, p)
		_, err := svc.Synthesize(context.Background(), text, SynthesizeOptions{})
		if err == nil {
			t.Errorf("纯空白文本 %q 应被 Service 拒绝", text)
		}
		if p.gotText != "" {
			t.Errorf("空白文本 %q 不应透传到 Provider, got %q", text, p.gotText)
		}
	}

	// 反向：含实际内容（即便首尾有空白）应正常透传，TrimSpace 仅用于判空不改写文本
	p := &passthroughTTS{}
	svc := NewService(nil, p)
	if _, err := svc.Synthesize(context.Background(), "  你好  ", SynthesizeOptions{}); err != nil {
		t.Fatalf("含内容文本不应被拒绝: %v", err)
	}
	if p.gotText != "  你好  " {
		t.Errorf("文本应原样透传（不被 trim），got %q", p.gotText)
	}
}
