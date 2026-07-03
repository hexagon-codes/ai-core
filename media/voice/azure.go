package voice

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

// AzureSTT Azure 语音转文本 Provider
//
// 使用 Azure Cognitive Services Speech REST API。
// 支持 100+ 语言，高精度中文识别。
type AzureSTT struct {
	subscriptionKey string
	region          string
	language        string
	// baseURL 当非空时整体覆盖由 region 拼出的端点，便于私有部署 / mock 注入。
	// 为空时回退到 https://{region}.stt.speech.microsoft.com 的官方端点。
	baseURL string
	client  *http.Client
	policy  *llm.NetworkPolicy
	tx      *transport.Transport
}

// AzureSTTOption 配置选项
type AzureSTTOption func(*AzureSTT)

// AzureSTTWithLanguage 设置默认语言
func AzureSTTWithLanguage(lang string) AzureSTTOption {
	return func(s *AzureSTT) { s.language = lang }
}

// AzureSTTWithBaseURL 覆盖默认端点（用于私有部署 / 中转 / 单测 mock）。
//
// 传入的 url 应是完整 scheme+host（可含路径前缀），如 httptest.Server.URL。
// 设置后将取代由 region 拼出的官方域名，使云请求可在不打真实网络的前提下单测。
func AzureSTTWithBaseURL(url string) AzureSTTOption {
	return func(s *AzureSTT) { s.baseURL = url }
}

// AzureSTTWithHTTPClient 注入自定义 HTTP client（测试 mock 时常用）。
func AzureSTTWithHTTPClient(c *http.Client) AzureSTTOption {
	return func(s *AzureSTT) { s.client = c }
}

// AzureSTTWithNetworkPolicy 设置上游网络出口约束。
func AzureSTTWithNetworkPolicy(policy llm.NetworkPolicy) AzureSTTOption {
	return func(s *AzureSTT) { s.policy = cloneVoiceNetworkPolicy(&policy) }
}

// NewAzureSTT 创建 Azure STT Provider
func NewAzureSTT(subscriptionKey, region string, opts ...AzureSTTOption) *AzureSTT {
	s := &AzureSTT{
		subscriptionKey: subscriptionKey,
		region:          region,
		language:        "zh-CN",
		client:          httpx.RawClient(httpx.WithRawTimeout(60 * time.Second)),
	}
	for _, opt := range opts {
		opt(s)
	}
	s.tx = transport.NewTransport(s.client, s.policy)
	return s
}

func (s *AzureSTT) Name() string { return "azure-stt" }

func (s *AzureSTT) SupportedFormats() []AudioFormat {
	return []AudioFormat{FormatWAV, FormatMP3, FormatOGG, FormatFLAC}
}

func (s *AzureSTT) SupportedLanguages() []string {
	return []string{"zh-CN", "en-US", "ja-JP", "ko-KR", "fr-FR", "de-DE", "es-ES"}
}

func (s *AzureSTT) Transcribe(ctx context.Context, audio []byte, opts TranscribeOptions) (*TranscribeResult, error) {
	if len(audio) == 0 {
		return nil, fmt.Errorf("音频数据为空")
	}

	lang := opts.Language
	if lang == "" {
		lang = s.language
	}

	// 端点优先用注入的 baseURL（私有部署 / mock），否则按 region 拼官方域名。
	base := s.baseURL
	if base == "" {
		base = fmt.Sprintf("https://%s.stt.speech.microsoft.com", s.region)
	}
	url := fmt.Sprintf("%s/speech/recognition/conversation/cognitiveservices/v1?language=%s&format=detailed",
		strings.TrimRight(base, "/"), lang)

	contentType := "audio/wav"
	switch opts.Format {
	case "mp3":
		contentType = "audio/mpeg"
	case "ogg":
		contentType = "audio/ogg"
	case "flac":
		contentType = "audio/flac"
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  s.Name(),
		Action:    "transcribe",
		Method:    http.MethodPost,
		URL:       url,
		Body:      audio,
		Client:    s.client,
		Transport: s.tx,
		Retry:     media.SubmitRetryPolicy(opts.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Ocp-Apim-Subscription-Key", s.subscriptionKey)
			r.Header.Set("Content-Type", contentType)
			if opts.IdempotencyKey != "" {
				r.Header.Set("Idempotency-Key", opts.IdempotencyKey)
			}
		},
	})
	if err != nil {
		return nil, fmt.Errorf("请求 Azure STT 失败: %w", err)
	}
	defer resp.Body.Close()

	var result azureSTTResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	confidence := 0.0
	if len(result.NBest) > 0 {
		confidence = result.NBest[0].Confidence
	}

	return &TranscribeResult{
		Text:       result.DisplayText,
		Language:   lang,
		Duration:   float64(result.Duration) / 10000000, // ticks to seconds
		Confidence: confidence,
		RequestID:  requestIDFromHeader(resp.Header),
	}, nil
}

type azureSTTResponse struct {
	RecognitionStatus string       `json:"RecognitionStatus"`
	DisplayText       string       `json:"DisplayText"`
	Duration          int64        `json:"Duration"`
	NBest             []azureNBest `json:"NBest"`
}

type azureNBest struct {
	Confidence float64 `json:"Confidence"`
	Display    string  `json:"Display"`
}

// AzureTTS Azure 文本转语音 Provider
//
// 使用 Azure Cognitive Services TTS REST API。
// 支持 400+ 神经网络语音，特别适合中文。
type AzureTTS struct {
	subscriptionKey string
	region          string
	defaultVoice    string
	// baseURL 当非空时整体覆盖由 region 拼出的端点，便于私有部署 / mock 注入。
	// 为空时回退到 https://{region}.tts.speech.microsoft.com 的官方端点。
	baseURL string
	client  *http.Client
	policy  *llm.NetworkPolicy
	tx      *transport.Transport
}

// AzureTTSOption 配置选项
type AzureTTSOption func(*AzureTTS)

// AzureTTSWithVoice 设置默认音色
func AzureTTSWithVoice(voice string) AzureTTSOption {
	return func(t *AzureTTS) { t.defaultVoice = voice }
}

// AzureTTSWithBaseURL 覆盖默认端点（用于私有部署 / 中转 / 单测 mock）。
//
// 传入的 url 应是完整 scheme+host（可含路径前缀），如 httptest.Server.URL。
// 设置后将取代由 region 拼出的官方域名，使合成请求 / 响应解析可单测。
func AzureTTSWithBaseURL(url string) AzureTTSOption {
	return func(t *AzureTTS) { t.baseURL = url }
}

// AzureTTSWithHTTPClient 注入自定义 HTTP client（测试 mock 时常用）。
func AzureTTSWithHTTPClient(c *http.Client) AzureTTSOption {
	return func(t *AzureTTS) { t.client = c }
}

// AzureTTSWithNetworkPolicy 设置上游网络出口约束。
func AzureTTSWithNetworkPolicy(policy llm.NetworkPolicy) AzureTTSOption {
	return func(t *AzureTTS) { t.policy = cloneVoiceNetworkPolicy(&policy) }
}

// NewAzureTTS 创建 Azure TTS Provider
func NewAzureTTS(subscriptionKey, region string, opts ...AzureTTSOption) *AzureTTS {
	t := &AzureTTS{
		subscriptionKey: subscriptionKey,
		region:          region,
		defaultVoice:    "zh-CN-XiaoxiaoNeural",
		client:          httpx.RawClient(httpx.WithRawTimeout(60 * time.Second)),
	}
	for _, opt := range opts {
		opt(t)
	}
	t.tx = transport.NewTransport(t.client, t.policy)
	return t
}

func (t *AzureTTS) Name() string { return "azure-tts" }

func (t *AzureTTS) SupportedFormats() []AudioFormat {
	return []AudioFormat{FormatMP3, FormatWAV, FormatOGG}
}

func (t *AzureTTS) Voices() []VoiceInfo {
	return []VoiceInfo{
		{ID: "zh-CN-XiaoxiaoNeural", Name: "晓晓", Language: "zh-CN", Gender: "female", Description: "温暖亲和的中文女声"},
		{ID: "zh-CN-YunxiNeural", Name: "云希", Language: "zh-CN", Gender: "male", Description: "阳光开朗的中文男声"},
		{ID: "zh-CN-YunjianNeural", Name: "云健", Language: "zh-CN", Gender: "male", Description: "沉稳大气的中文男声"},
		{ID: "zh-CN-XiaoyiNeural", Name: "晓艺", Language: "zh-CN", Gender: "female", Description: "活泼可爱的中文女声"},
		{ID: "en-US-JennyNeural", Name: "Jenny", Language: "en-US", Gender: "female", Description: "自然流畅的英文女声"},
		{ID: "en-US-GuyNeural", Name: "Guy", Language: "en-US", Gender: "male", Description: "专业稳重的英文男声"},
		{ID: "ja-JP-NanamiNeural", Name: "七海", Language: "ja-JP", Gender: "female", Description: "温柔的日文女声"},
	}
}

func (t *AzureTTS) Synthesize(ctx context.Context, text string, opts SynthesizeOptions) (*SynthesizeResult, error) {
	if text == "" {
		return nil, fmt.Errorf("文本内容为空")
	}

	voice := opts.Voice
	if voice == "" {
		voice = t.defaultVoice
	}
	format := opts.Format
	if format == "" {
		format = FormatMP3
	}

	// 构建 SSML（转义用户文本防止 SSML 注入）
	ssml := fmt.Sprintf(`<speak version='1.0' xml:lang='zh-CN'>
		<voice name='%s'>%s</voice>
	</speak>`, escapeXML(voice), escapeXML(text))

	// 端点优先用注入的 baseURL（私有部署 / mock），否则按 region 拼官方域名。
	base := t.baseURL
	if base == "" {
		base = fmt.Sprintf("https://%s.tts.speech.microsoft.com", t.region)
	}
	url := fmt.Sprintf("%s/cognitiveservices/v1", strings.TrimRight(base, "/"))

	outputFormat := "audio-16khz-128kbitrate-mono-mp3"
	switch format {
	case FormatWAV:
		outputFormat = "riff-16khz-16bit-mono-pcm"
	case FormatOGG:
		outputFormat = "ogg-16khz-16bit-mono-opus"
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  t.Name(),
		Action:    "synthesize",
		Method:    http.MethodPost,
		URL:       url,
		Body:      []byte(ssml),
		Client:    t.client,
		Transport: t.tx,
		Retry:     media.SubmitRetryPolicy(opts.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Ocp-Apim-Subscription-Key", t.subscriptionKey)
			r.Header.Set("Content-Type", "application/ssml+xml")
			r.Header.Set("X-Microsoft-OutputFormat", outputFormat)
			if opts.IdempotencyKey != "" {
				r.Header.Set("Idempotency-Key", opts.IdempotencyKey)
			}
		},
	})
	if err != nil {
		return nil, fmt.Errorf("请求 Azure TTS 失败: %w", err)
	}
	defer resp.Body.Close()

	audio, err := io.ReadAll(io.LimitReader(resp.Body, 50<<20)) // 50MB 限制
	if err != nil {
		return nil, fmt.Errorf("读取音频数据失败: %w", err)
	}

	return &SynthesizeResult{
		Audio:     audio,
		Format:    format,
		Size:      len(audio),
		RequestID: requestIDFromHeader(resp.Header),
	}, nil
}

// escapeXML 转义 XML 特殊字符（防止 SSML 注入）
func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "'", "&apos;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	return s
}
