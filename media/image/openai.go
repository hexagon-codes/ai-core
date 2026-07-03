package image

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
)

// DefaultOpenAIBaseURL OpenAI 默认 endpoint（可被 Azure/智谱兼容 endpoint 覆盖）
const DefaultOpenAIBaseURL = "https://api.openai.com/v1"

// OpenAI 兼容协议 Provider（DALL-E / 智谱 CogView 等都走 OpenAI Images API 协议）。
type openAICompatProvider struct {
	name    string        // 显示名（如 "openai-dalle"、"zhipu-cogview"）
	baseURL string        // OpenAI Images endpoint base（如 https://api.openai.com/v1）
	apiKey  string        // Bearer token
	models  []string      // 支持的模型 ID 白名单
	httpc   *http.Client  // HTTP 客户端，可注入超时
	timeout time.Duration // 默认请求超时
	policy  *llm.NetworkPolicy
	tx      *transport.Transport
}

// OpenAICompatOption configures OpenAI-compatible image providers.
type OpenAICompatOption func(*openAICompatProvider)

// NewOpenAIDallE 创建 OpenAI DALL-E Provider。
//
// baseURL 为空时使用 DefaultOpenAIBaseURL。支持 dall-e-2 / dall-e-3 / gpt-image-1。
func NewOpenAIDallE(apiKey, baseURL string) Provider {
	return NewOpenAIDallEWithOptions(apiKey, baseURL)
}

// NewOpenAIDallEWithOptions 创建可配置的 OpenAI DALL-E Provider。
func NewOpenAIDallEWithOptions(apiKey, baseURL string, opts ...OpenAICompatOption) Provider {
	if baseURL == "" {
		baseURL = DefaultOpenAIBaseURL
	}
	p := &openAICompatProvider{
		name:    "openai-dalle",
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		models:  []string{"dall-e-3", "dall-e-2", "gpt-image-1"},
		httpc:   http.DefaultClient,
		timeout: 120 * time.Second, // 图像生成普遍较慢
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	p.tx = transport.NewTransport(p.httpc, p.policy)
	return p
}

// NewZhipuCogView 创建智谱 CogView Provider（OpenAI 兼容协议）。
//
// 智谱端点：https://open.bigmodel.cn/api/paas/v4/images/generations
func NewZhipuCogView(apiKey, baseURL string) Provider {
	return NewZhipuCogViewWithOptions(apiKey, baseURL)
}

// NewZhipuCogViewWithOptions 创建可配置的智谱 CogView Provider。
func NewZhipuCogViewWithOptions(apiKey, baseURL string, opts ...OpenAICompatOption) Provider {
	if baseURL == "" {
		baseURL = "https://open.bigmodel.cn/api/paas/v4"
	}
	p := &openAICompatProvider{
		name:    "zhipu-cogview",
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		models:  []string{"cogview-4", "cogview-3-plus", "cogview-3"},
		httpc:   http.DefaultClient,
		timeout: 120 * time.Second,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	p.tx = transport.NewTransport(p.httpc, p.policy)
	return p
}

// NewOpenAICompatible 通用 OpenAI Images 兼容 Provider（自托管 / 第三方）。
//
// modelIDs 用于 model→provider 反查路由。
func NewOpenAICompatible(name, apiKey, baseURL string, modelIDs []string) Provider {
	return NewOpenAICompatibleWithOptions(name, apiKey, baseURL, modelIDs)
}

// NewOpenAICompatibleWithOptions 创建可配置的通用 OpenAI Images 兼容 Provider。
func NewOpenAICompatibleWithOptions(name, apiKey, baseURL string, modelIDs []string, opts ...OpenAICompatOption) Provider {
	if baseURL == "" {
		baseURL = DefaultOpenAIBaseURL
	}
	p := &openAICompatProvider{
		name:    name,
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		models:  modelIDs,
		httpc:   http.DefaultClient,
		timeout: 120 * time.Second,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	p.tx = transport.NewTransport(p.httpc, p.policy)
	return p
}

// OpenAICompatWithHTTPClient 注入自定义 HTTP client。
func OpenAICompatWithHTTPClient(client *http.Client) OpenAICompatOption {
	return func(p *openAICompatProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

// OpenAICompatWithNetworkPolicy 设置上游网络出口约束。
func OpenAICompatWithNetworkPolicy(policy llm.NetworkPolicy) OpenAICompatOption {
	return func(p *openAICompatProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *openAICompatProvider) Name() string              { return p.name }
func (p *openAICompatProvider) SupportedModels() []string { return p.models }

// openAIImagesRequest 匹配 OpenAI POST /images/generations 协议。
type openAIImagesRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	Style          string `json:"style,omitempty"`
	Quality        string `json:"quality,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	User           string `json:"user,omitempty"`
}

type openAIImagesResponse struct {
	Created int64 `json:"created"`
	Data    []struct {
		URL           string `json:"url,omitempty"`
		B64JSON       string `json:"b64_json,omitempty"`
		RevisedPrompt string `json:"revised_prompt,omitempty"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

func (p *openAICompatProvider) Generate(ctx context.Context, req Request) (*Result, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("%s: 未配置 API Key", p.name)
	}
	model := req.Model
	if model == "" && len(p.models) > 0 {
		model = p.models[0]
	}
	body := openAIImagesRequest{
		Model:   model,
		Prompt:  req.Prompt,
		N:       req.N,
		Size:    req.Size,
		Style:   req.Style,
		Quality: req.Quality,
		User:    req.UserID,
	}
	if body.N <= 0 {
		body.N = 1
	}
	if body.Size == "" {
		body.Size = "1024x1024"
	}
	// 默认请求 base64 直接内嵌，避免依赖外部 URL 寿命；前端按需自行选择展示
	body.ResponseFormat = "b64_json"

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	start := time.Now()
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.name,
		Action:    "image.generate",
		Method:    http.MethodPost,
		URL:       p.baseURL + "/images/generations",
		Body:      payload,
		Client:    p.httpc,
		Transport: p.tx,
		Timeout:   p.timeout,
		// 本 Provider 不向上游透传幂等键，计费型生成 POST 一律不重试。
		Retry: media.SubmitRetryPolicy(""),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Authorization", "Bearer "+p.apiKey)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("call %s: %w", p.name, err)
	}
	defer resp.Body.Close()

	var parsed openAIImagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("%s: %s", p.name, parsed.Error.Message)
	}

	images := make([]Image, 0, len(parsed.Data))
	for _, d := range parsed.Data {
		images = append(images, Image{
			URL:           d.URL,
			B64JSON:       d.B64JSON,
			RevisedPrompt: d.RevisedPrompt,
		})
	}

	return &Result{
		Provider:  p.name,
		Model:     model,
		Created:   parsed.Created,
		Images:    images,
		RequestID: resp.Header.Get("x-request-id"),
		UsageMs:   time.Since(start).Milliseconds(),
	}, nil
}
