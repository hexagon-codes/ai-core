package voicechat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/transport"
)

// OpenAI gpt-4o-audio-preview 协议：
//   POST {base}/chat/completions
//   {
//     "model": "gpt-4o-audio-preview",
//     "modalities": ["text", "audio"],
//     "audio": { "voice": "alloy", "format": "wav" },
//     "messages": [
//       { "role": "user", "content": [
//           { "type": "input_audio", "input_audio": { "data": "<b64>", "format": "wav" } }
//       ]}
//     ]
//   }
//
// 响应：message.audio.{ data, transcript, id }，message.audio.transcript = 模型回应文本
//
// 文档：https://platform.openai.com/docs/guides/audio

type openAIVoiceChat struct {
	apiKey  string
	baseURL string
	models  []string
	httpc   *http.Client
	timeout time.Duration
	policy  *llm.NetworkPolicy
	tx      *transport.Transport
}

type OpenAIVoiceChatOption func(*openAIVoiceChat)

// NewOpenAIVoiceChat 创建 OpenAI gpt-4o-audio Provider。
func NewOpenAIVoiceChat(apiKey, baseURL string) Provider {
	return NewOpenAIVoiceChatWithOptions(apiKey, baseURL)
}

// NewOpenAIVoiceChatWithOptions 创建可配置的 OpenAI gpt-4o-audio Provider。
func NewOpenAIVoiceChatWithOptions(apiKey, baseURL string, opts ...OpenAIVoiceChatOption) Provider {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	p := &openAIVoiceChat{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		// 复用包级权威来源 builtinModels，避免内置模型清单与文档承诺漂移。
		models:  BuiltinModels(),
		httpc:   http.DefaultClient,
		timeout: 90 * time.Second,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	p.tx = transport.NewTransport(p.httpc, p.policy)
	return p
}

// OpenAIVoiceChatWithHTTPClient 注入自定义 HTTP client。
func OpenAIVoiceChatWithHTTPClient(client *http.Client) OpenAIVoiceChatOption {
	return func(p *openAIVoiceChat) {
		if client != nil {
			p.httpc = client
		}
	}
}

// OpenAIVoiceChatWithNetworkPolicy 设置上游网络出口约束。
func OpenAIVoiceChatWithNetworkPolicy(policy llm.NetworkPolicy) OpenAIVoiceChatOption {
	return func(p *openAIVoiceChat) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *openAIVoiceChat) Name() string { return "openai-gpt4o-audio" }

// SupportedModels 返回本 Provider 支持的模型列表。
//
// 返回内部切片的防御性拷贝，避免外部修改污染 Provider 内部状态。
func (p *openAIVoiceChat) SupportedModels() []string {
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

type oaChatReq struct {
	Model      string         `json:"model"`
	Modalities []string       `json:"modalities"`
	Audio      *oaAudioConfig `json:"audio,omitempty"`
	Messages   []oaMessage    `json:"messages"`
}

type oaAudioConfig struct {
	Voice  string `json:"voice"`
	Format string `json:"format"`
}

type oaMessage struct {
	Role    string   `json:"role"`
	Content []oaPart `json:"content,omitempty"`
}

type oaPart struct {
	Type       string        `json:"type"`
	Text       string        `json:"text,omitempty"`
	InputAudio *oaInputAudio `json:"input_audio,omitempty"`
}

type oaInputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type oaChatResp struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			Audio   *struct {
				ID         string `json:"id"`
				Data       string `json:"data"`
				Transcript string `json:"transcript"`
			} `json:"audio"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (p *openAIVoiceChat) Chat(ctx context.Context, req Request) (*Result, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("%s: 未配置 API Key", p.Name())
	}
	// Provider 自校验：NewOpenAIVoiceChat 返回导出的 Provider 接口，
	// 可被绕过 Service 直接调用，因此不能假设输入已被上层校验。
	// audio 与 text 皆空时会构造 content 为空的 user 消息（上游会 400），提前拦截。
	if req.AudioB64 == "" && req.Text == "" {
		return nil, fmt.Errorf("%s: audio 与 text 至少需要一个", p.Name())
	}

	model := req.Model
	if model == "" {
		model = p.models[0]
	}
	voice := req.Voice
	if voice == "" {
		voice = "alloy"
	}
	format := req.Format
	if format == "" {
		format = "wav"
	}

	messages := []oaMessage{}
	if req.System != "" {
		messages = append(messages, oaMessage{Role: "system", Content: []oaPart{{Type: "text", Text: req.System}}})
	}
	// 历史轮次（简化：仅文本，复用音频 ID 留作 v0.5）
	for _, h := range req.History {
		if h.Text == "" {
			continue
		}
		messages = append(messages, oaMessage{Role: h.Role, Content: []oaPart{{Type: "text", Text: h.Text}}})
	}
	// 当前用户输入
	userParts := []oaPart{}
	if req.Text != "" {
		userParts = append(userParts, oaPart{Type: "text", Text: req.Text})
	}
	if req.AudioB64 != "" {
		userParts = append(userParts, oaPart{
			Type:       "input_audio",
			InputAudio: &oaInputAudio{Data: req.AudioB64, Format: format},
		})
	}
	messages = append(messages, oaMessage{Role: "user", Content: userParts})

	body := oaChatReq{
		Model:      model,
		Modalities: []string{"text", "audio"},
		Audio:      &oaAudioConfig{Voice: voice, Format: format},
		Messages:   messages,
	}
	payload, _ := json.Marshal(body)

	rctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	start := time.Now()
	resp, err := transport.Do(rctx, transport.Request{
		Provider:  p.Name(),
		Action:    "chat",
		Method:    http.MethodPost,
		URL:       p.baseURL + "/chat/completions",
		Body:      payload,
		Client:    p.httpc,
		Transport: p.tx,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Authorization", "Bearer "+p.apiKey)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("call %s: %w", p.Name(), err)
	}
	defer resp.Body.Close()

	var parsed oaChatResp
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("%s: %s", p.Name(), parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return nil, fmt.Errorf("%s: 无回应", p.Name())
	}
	msg := parsed.Choices[0].Message
	res := &Result{
		Provider:   p.Name(),
		Model:      model,
		Format:     format,
		Transcript: msg.Content,
		UsageMs:    time.Since(start).Milliseconds(),
		RequestID:  requestIDFromHeader(resp.Header),
	}
	if msg.Audio != nil {
		res.ResponseB64 = msg.Audio.Data
		if msg.Audio.Transcript != "" {
			res.Transcript = msg.Audio.Transcript
		}
	}
	return res, nil
}
