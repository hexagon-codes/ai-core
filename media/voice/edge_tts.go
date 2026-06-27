package voice

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

// EdgeTTS Microsoft Edge 免费 TTS Provider
//
// 走 Edge 浏览器「大声朗读(Read Aloud)」用的免费 WebSocket 端点，免费、无需 API Key，
// 中文音质接近 Azure Neural TTS。
//
// ⚠ 历史坑（v0.4.7 修复）：旧实现 POST 到 Azure *付费* REST 端点
// (eastus.api.speech.microsoft.com/cognitiveservices/v1)——该端点需 Ocp-Apim-Subscription-Key，
// 未授权请求被拒（实测 405 Forbidden POST），即「朗读」按钮开箱即坏。真正免费的 Edge TTS 是
// 下面这套 WebSocket 协议（含 Sec-MS-GEC 防滥用 token + 二进制音频帧），与 edge-tts(Python) 一致。
type EdgeTTS struct {
	defaultVoice string
	// wssBase 合成请求的 WebSocket 基址，默认指向 Edge Read Aloud 免费端点。
	// 提供注入点便于私有部署 / 中转 / 单测 mock（httptest WS server），避免端点硬编码不可测。
	wssBase    string
	httpClient *http.Client
	// timeout 是基础超时；实际超时随文本长度伸缩（见 effectiveTimeout），避免长文本合成被固定超时切断。
	timeout time.Duration
	// gecVersion / userAgent 可经 option 或环境变量覆盖：微软会校验版本，过旧→403；
	// 暴露覆盖点使线上能热修版本号而无需重编发布整条依赖链。
	gecVersion string
	userAgent  string
	// now 注入当前时间，便于 Sec-MS-GEC token 生成的确定性单测；默认 time.Now。
	now func() time.Time
}

const (
	// defaultEdgeTTSWSS 是 Edge「大声朗读」的免费 WebSocket 端点（无需 Key）。
	defaultEdgeTTSWSS = "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1"
	// edgeTrustedClientToken 是 Edge Read Aloud 公开的固定客户端 token（与 edge-tts 社区实现一致）。
	edgeTrustedClientToken = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
	// defaultEdgeSecMSGECVersion 是 Sec-MS-GEC token 一起上送的客户端版本标识的默认值。
	// ⚠ 微软会校验该版本：过旧会被拒（403）。需随上游 edge-tts 周期性同步到较新 Edge 版本，
	// 或经环境变量 HEXCLAW_EDGE_TTS_GEC_VERSION 线上热修（无需重编）。
	defaultEdgeSecMSGECVersion = "1-143.0.3650.75"
	// defaultEdgeUserAgent 与 GEC 版本对应的 UA（可经 HEXCLAW_EDGE_TTS_UA 覆盖）。
	defaultEdgeUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
	// edgeOutputFormat 请求 24kHz 48kbps 单声道 MP3（Tauri/WKWebView 通吃）。
	edgeOutputFormat = "audio-24khz-48kbitrate-mono-mp3"
	// edgeTimestampLayout 匹配 Edge 协议要求的 X-Timestamp 文本格式。
	edgeTimestampLayout = "Mon Jan 02 2006 15:04:05 GMT+0000 (Coordinated Universal Time)"
	// edgeWSReadLimit 单条二进制音频帧上限（默认 32KB 太小，长文本会超限断开）。
	edgeWSReadLimit = 50 << 20
	// edgeMaxTotalAudio 跨帧累积音频总上限：防异常/恶意上游在超时窗口内无限推帧致内存膨胀。
	edgeMaxTotalAudio = 50 << 20

	// 覆盖用环境变量。
	edgeGECVersionEnv = "HEXCLAW_EDGE_TTS_GEC_VERSION"
	edgeUAEnv         = "HEXCLAW_EDGE_TTS_UA"

	// edgeMaxTimeout 是文本伸缩后的超时上限；edgePerRuneTimeout 是每字符追加的合成预算。
	edgeMaxTimeout     = 5 * time.Minute
	edgePerRuneTimeout = 20 * time.Millisecond
	// edgeSkewRetryThreshold：仅当客户端与服务端时钟偏移超过该阈值时，才按服务端时间重算 token 重试。
	edgeSkewRetryThreshold = time.Minute
)

// EdgeTTSOption 配置选项
type EdgeTTSOption func(*EdgeTTS)

// EdgeTTSWithVoice 设置默认音色
func EdgeTTSWithVoice(voice string) EdgeTTSOption {
	return func(t *EdgeTTS) { t.defaultVoice = voice }
}

// EdgeTTSWithEndpoint 覆盖默认 WebSocket 合成端点（用于私有部署 / 中转 / 单测 mock）。
//
// 传入的 url 应是完整 ws:// 或 wss:// 地址（如把 httptest.Server.URL 的 http→ws），
// 设置后取代硬编码端点，使合成请求 / 响应解析可在不打真实网络的前提下单测。
func EdgeTTSWithEndpoint(url string) EdgeTTSOption {
	return func(t *EdgeTTS) { t.wssBase = url }
}

// EdgeTTSWithHTTPClient 注入自定义 HTTP client（测试 mock / 自定义代理时常用）。
func EdgeTTSWithHTTPClient(c *http.Client) EdgeTTSOption {
	return func(t *EdgeTTS) { t.httpClient = c }
}

// EdgeTTSWithGECVersion 覆盖 Sec-MS-GEC-Version（微软封禁旧版时热修用）。
func EdgeTTSWithGECVersion(v string) EdgeTTSOption {
	return func(t *EdgeTTS) {
		if strings.TrimSpace(v) != "" {
			t.gecVersion = v
		}
	}
}

// EdgeTTSWithUserAgent 覆盖握手 User-Agent（与 GEC 版本配套）。
func EdgeTTSWithUserAgent(ua string) EdgeTTSOption {
	return func(t *EdgeTTS) {
		if strings.TrimSpace(ua) != "" {
			t.userAgent = ua
		}
	}
}

// NewEdgeTTS 创建 Edge TTS Provider（免费，无需 Key）
func NewEdgeTTS(opts ...EdgeTTSOption) *EdgeTTS {
	t := &EdgeTTS{
		defaultVoice: "zh-CN-XiaoxiaoNeural",
		wssBase:      defaultEdgeTTSWSS,
		httpClient:   httpx.RawClient(httpx.WithRawTimeout(60 * time.Second)),
		timeout:      60 * time.Second,
		gecVersion:   envOrDefault(edgeGECVersionEnv, defaultEdgeSecMSGECVersion),
		userAgent:    envOrDefault(edgeUAEnv, defaultEdgeUserAgent),
		now:          time.Now,
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

func envOrDefault(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// Name 返回 Provider 名称。
func (t *EdgeTTS) Name() string { return "edge-tts" }

// SupportedFormats 返回支持的音频格式（Edge Read Aloud 走 MP3）。
func (t *EdgeTTS) SupportedFormats() []AudioFormat {
	return []AudioFormat{FormatMP3}
}

// Voices 返回内置可选音色清单（含多个中文音色）。
func (t *EdgeTTS) Voices() []VoiceInfo {
	return []VoiceInfo{
		{ID: "zh-CN-XiaoxiaoNeural", Name: "晓晓", Language: "zh-CN", Gender: "female", Description: "温暖亲和"},
		{ID: "zh-CN-YunxiNeural", Name: "云希", Language: "zh-CN", Gender: "male", Description: "阳光开朗"},
		{ID: "zh-CN-YunjianNeural", Name: "云健", Language: "zh-CN", Gender: "male", Description: "沉稳大气"},
		{ID: "zh-CN-XiaoyiNeural", Name: "晓艺", Language: "zh-CN", Gender: "female", Description: "活泼可爱"},
		{ID: "zh-CN-YunyangNeural", Name: "云扬", Language: "zh-CN", Gender: "male", Description: "新闻播报"},
		{ID: "zh-TW-HsiaoChenNeural", Name: "曉臻", Language: "zh-TW", Gender: "female", Description: "台湾女声"},
		{ID: "en-US-JennyNeural", Name: "Jenny", Language: "en-US", Gender: "female", Description: "自然英文女声"},
		{ID: "en-US-GuyNeural", Name: "Guy", Language: "en-US", Gender: "male", Description: "稳重英文男声"},
		{ID: "ja-JP-NanamiNeural", Name: "七海", Language: "ja-JP", Gender: "female", Description: "温柔日文女声"},
		{ID: "ko-KR-SunHiNeural", Name: "선히", Language: "ko-KR", Gender: "female", Description: "韩文女声"},
	}
}

// effectiveTimeout 按文本长度伸缩超时：基础超时 + 每字符预算，封顶 edgeMaxTimeout。
// 修复固定 60s 长文本必超时的问题。
func (t *EdgeTTS) effectiveTimeout(text string) time.Duration {
	d := t.timeout + time.Duration(utf8.RuneCountInString(text))*edgePerRuneTimeout
	if d > edgeMaxTimeout {
		return edgeMaxTimeout
	}
	return d
}

// Synthesize 通过 Edge Read Aloud 免费 WebSocket 协议合成语音，返回 MP3。
//
// 协议（与 edge-tts 社区实现一致）：连接带 TrustedClientToken + Sec-MS-GEC token 的 wss →
// 发 speech.config + ssml 文本帧 → 循环读二进制帧 [2字节大端头长][头文本][音频字节]，累积
// Path:audio 音频；收到 Path:turn.end 文本帧即完成。
//
// 健壮性：握手 403 且检测到客户端/服务端时钟偏移（>阈值）时，按服务端时间重算 token 重试一次。
func (t *EdgeTTS) Synthesize(ctx context.Context, text string, opts SynthesizeOptions) (*SynthesizeResult, error) {
	if text == "" {
		return nil, fmt.Errorf("文本内容为空")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	voiceName := opts.Voice
	if voiceName == "" {
		voiceName = t.defaultVoice
	}
	speed := opts.Speed
	if speed <= 0 {
		speed = 1.0
	}
	ratePercent := int(math.Round((speed - 1.0) * 100)) // 四舍五入而非截断：0.8→-20% 非 -19%
	rateStr := fmt.Sprintf("%+d%%", ratePercent)
	ssml := fmt.Sprintf(
		`<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='%s'><voice name='%s'><prosody rate='%s'>%s</prosody></voice></speak>`,
		edgeVoiceLang(voiceName), escapeXML(voiceName), escapeXML(rateStr), escapeXML(text),
	)

	var skew time.Duration
	for attempt := range 2 {
		res, retrySkew, err := t.synthesizeOnce(ctx, text, ssml, skew)
		if err == nil {
			return res, nil
		}
		// 仅首次失败、且检测到显著时钟偏移时，按服务端时间重算 token 重试一次。
		if attempt == 0 && retrySkew != 0 {
			skew = retrySkew
			continue
		}
		return nil, err
	}
	return nil, fmt.Errorf("未能完成 Edge TTS 合成")
}

// synthesizeOnce 单次握手 + 合成。返回 (结果, 建议重试的时钟偏移, error)：
// 偏移非 0 表示握手 403 疑似时钟偏移所致，调用方可据此重试一次。
func (t *EdgeTTS) synthesizeOnce(ctx context.Context, text, ssml string, skew time.Duration) (*SynthesizeResult, time.Duration, error) {
	connID := newEdgeConnID()
	dialURL := fmt.Sprintf(
		"%s?TrustedClientToken=%s&Sec-MS-GEC=%s&Sec-MS-GEC-Version=%s&ConnectionId=%s",
		edgeBase(t.wssBase), edgeTrustedClientToken, edgeSecMSGEC(t.now().Add(skew)), t.gecVersion, connID,
	)

	dialCtx, cancel := context.WithTimeout(ctx, t.effectiveTimeout(text))
	defer cancel()

	// coder/websocket.Dial 内部已处理握手响应体（成功置 nil、失败自行关闭），文档明示"无需自行关闭"。
	conn, resp, err := websocket.Dial(dialCtx, dialURL, &websocket.DialOptions{ //nolint:bodyclose // 见上：Dial 自行管理 resp.Body
		HTTPClient: t.httpClient,
		HTTPHeader: t.wsHeaders(),
	})
	if err != nil {
		// 403 + 显著时钟偏移：返回偏移让上层按服务端时间重算 token 重试。
		if resp != nil && resp.StatusCode == http.StatusForbidden {
			if d := serverDateSkew(resp, t.now()); d > edgeSkewRetryThreshold || d < -edgeSkewRetryThreshold {
				return nil, d, fmt.Errorf("连接 Edge TTS 失败(403，疑时钟偏移 %s): %w", d.Round(time.Second), err)
			}
		}
		return nil, 0, fmt.Errorf("连接 Edge TTS 失败: %w", err)
	}
	defer conn.Close(websocket.StatusInternalError, "")
	conn.SetReadLimit(edgeWSReadLimit)

	timestamp := t.now().UTC().Format(edgeTimestampLayout)
	configMsg := "X-Timestamp:" + timestamp +
		"\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n" +
		`{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},"outputFormat":"` +
		edgeOutputFormat + `"}}}}`
	if werr := conn.Write(dialCtx, websocket.MessageText, []byte(configMsg)); werr != nil {
		return nil, 0, fmt.Errorf("发送 Edge TTS 配置失败: %w", werr)
	}

	ssmlMsg := "X-RequestId:" + connID +
		"\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:" + timestamp +
		"\r\nPath:ssml\r\n\r\n" + ssml
	if werr := conn.Write(dialCtx, websocket.MessageText, []byte(ssmlMsg)); werr != nil {
		return nil, 0, fmt.Errorf("发送 Edge TTS 文本失败: %w", werr)
	}

	var audio []byte
	for {
		msgType, data, readErr := conn.Read(dialCtx)
		if readErr != nil {
			return nil, 0, fmt.Errorf("读取 Edge TTS 响应失败: %w", readErr)
		}
		switch msgType {
		case websocket.MessageText:
			if strings.Contains(string(data), "Path:turn.end") {
				conn.Close(websocket.StatusNormalClosure, "")
				if len(audio) == 0 {
					return nil, 0, fmt.Errorf("未从 Edge TTS 获取到音频数据")
				}
				return &SynthesizeResult{Audio: audio, Format: FormatMP3, Size: len(audio)}, 0, nil
			}
		case websocket.MessageBinary:
			if chunk, ok := parseEdgeAudioFrame(data); ok {
				audio = append(audio, chunk...)
				if len(audio) > edgeMaxTotalAudio {
					return nil, 0, fmt.Errorf("音频累积超过上限 %d 字节", edgeMaxTotalAudio)
				}
			}
		}
	}
}

func edgeBase(wssBase string) string {
	if wssBase == "" {
		return defaultEdgeTTSWSS
	}
	return wssBase
}

// edgeVoiceLang 从音色 ID（如 en-US-JennyNeural）取 BCP-47 语言（en-US），供 SSML xml:lang，
// 避免英/日/韩音色仍标 zh-CN 的语义错配。无法解析时回退 zh-CN。
func edgeVoiceLang(voice string) string {
	parts := strings.Split(voice, "-")
	if len(parts) >= 2 && parts[0] != "" && parts[1] != "" {
		return parts[0] + "-" + parts[1]
	}
	return "zh-CN"
}

// parseEdgeAudioFrame 解析 Edge 二进制音频帧：[2字节大端头长度][头文本][音频字节]。
// 仅当帧头精确含 "Path:audio\r\n" 时返回音频负载（避免把 "Path:audio.metadata" 误当音频）。
func parseEdgeAudioFrame(data []byte) ([]byte, bool) {
	if len(data) < 2 {
		return nil, false
	}
	headerLen := int(binary.BigEndian.Uint16(data[:2]))
	if 2+headerLen > len(data) {
		return nil, false
	}
	header := data[2 : 2+headerLen]
	if !strings.Contains(string(header), "Path:audio\r\n") {
		return nil, false
	}
	return data[2+headerLen:], true
}

// edgeSecMSGEC 生成 Edge TTS 防滥用 token Sec-MS-GEC。
//
// 算法（与 edge-tts 一致）：取当前 Windows 文件时间（自 1601-01-01 起的 100ns 计数），
// 向下取整到最近 5 分钟边界，与 TrustedClientToken 拼接后 SHA256，输出大写十六进制。
func edgeSecMSGEC(now time.Time) string {
	const winEpochOffsetSec = 11644473600 // 1601→1970 秒差
	const fiveMinTicks = 3_000_000_000    // 300s × 10^7 ticks
	ticks := (now.UTC().Unix() + winEpochOffsetSec) * 10_000_000
	ticks -= ticks % fiveMinTicks
	sum := sha256.Sum256(fmt.Appendf(nil, "%d%s", ticks, edgeTrustedClientToken))
	return fmt.Sprintf("%X", sum[:])
}

// serverDateSkew 从 403 响应的 Date 头估算「服务端时间 - 客户端时间」偏移；不可解析则返回 0。
func serverDateSkew(resp *http.Response, now time.Time) time.Duration {
	if resp == nil {
		return 0
	}
	ds := resp.Header.Get("Date")
	if ds == "" {
		return 0
	}
	st, err := http.ParseTime(ds)
	if err != nil {
		return 0
	}
	return st.Sub(now)
}

// newEdgeConnID 生成 32 位大写十六进制连接 ID（X-RequestId / ConnectionId 复用）。
func newEdgeConnID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return fmt.Sprintf("%X", b[:])
}

// wsHeaders 模拟 Edge「大声朗读」扩展的握手头（Origin 是端点鉴别的关键之一）。
func (t *EdgeTTS) wsHeaders() http.Header {
	h := http.Header{}
	h.Set("Pragma", "no-cache")
	h.Set("Cache-Control", "no-cache")
	h.Set("Origin", "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold")
	h.Set("User-Agent", t.userAgent)
	return h
}
