// Package ark 提供字节跳动豆包 (Ark) LLM Provider 实现
//
// 豆包是字节跳动推出的大语言模型，通过火山引擎 Ark 平台提供 API 服务。
// 本包实现了与 Hexagon 框架兼容的 Provider 接口。
//
// 支持的模型：
//   - Doubao-pro-32k: 豆包专业版，32K 上下文
//   - Doubao-pro-128k: 豆包专业版，128K 上下文
//   - Doubao-lite-32k: 豆包轻量版，32K 上下文
//   - Doubao-lite-128k: 豆包轻量版，128K 上下文
//
// 使用示例：
//
//	provider := ark.New("your-api-key",
//	    ark.WithModel("doubao-pro-32k"),
//	)
//	resp, err := provider.Complete(ctx, llm.CompletionRequest{
//	    Messages: []llm.Message{{Role: llm.RoleUser, Content: "你好"}},
//	})
package ark

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const (
	// 豆包 API 基础 URL (火山引擎)
	defaultBaseURL = "https://ark.cn-beijing.volces.com/api/v3"
	defaultModel   = "doubao-pro-32k"
)

// Provider 实现豆包 (Ark) LLM 提供者
type Provider struct {
	apiKey     string
	baseURL    string
	model      string
	endpointID string // 火山引擎的端点 ID (可选)
	httpClient *http.Client
}

// Option 是 Provider 的配置选项
type Option func(*Provider)

// WithBaseURL 设置 API 基础 URL
func WithBaseURL(url string) Option {
	return func(p *Provider) {
		p.baseURL = url
	}
}

// WithModel 设置默认模型
func WithModel(model string) Option {
	return func(p *Provider) {
		p.model = model
	}
}

// WithEndpointID 设置火山引擎端点 ID
// 在火山引擎控制台创建的推理端点 ID
func WithEndpointID(id string) Option {
	return func(p *Provider) {
		p.endpointID = id
	}
}

// WithHTTPClient 设置 HTTP 客户端
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// New 创建豆包 Provider
// apiKey 可以为空，会从环境变量 ARK_API_KEY 或 VOLC_ACCESSKEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("ARK_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("VOLC_ACCESSKEY")
		}
	}

	p := &Provider{
		apiKey:     apiKey,
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: httpx.RawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
	}

	// 从环境变量读取端点 ID
	if id := os.Getenv("ARK_ENDPOINT_ID"); id != "" {
		p.endpointID = id
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "ark"
}

// Complete 执行非流式补全请求
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ark request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("ark api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		return nil, fmt.Errorf("ark api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result arkResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return p.parseResponse(&result), nil
}

// Stream 执行流式补全请求
func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ark request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("ark api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		return nil, fmt.Errorf("ark api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	// 豆包使用 OpenAI 兼容格式
	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.OpenAIFormat), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "doubao-pro-32k",
			Name:        "Doubao Pro 32K",
			Description: "豆包专业版，32K 上下文，适合复杂任务",
			MaxTokens:   32768,
			InputCost:   0.80,
			OutputCost:  2.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "doubao-pro-128k",
			Name:        "Doubao Pro 128K",
			Description: "豆包专业版，128K 上下文，适合长文档处理",
			MaxTokens:   131072,
			InputCost:   5.00,
			OutputCost:  9.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "doubao-pro-256k",
			Name:        "Doubao Pro 256K",
			Description: "豆包专业版，256K 上下文，超长文档处理",
			MaxTokens:   262144,
			InputCost:   5.00,
			OutputCost:  9.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "doubao-lite-32k",
			Name:        "Doubao Lite 32K",
			Description: "豆包轻量版，32K 上下文，成本更低",
			MaxTokens:   32768,
			InputCost:   0.30,
			OutputCost:  0.60,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "doubao-lite-128k",
			Name:        "Doubao Lite 128K",
			Description: "豆包轻量版，128K 上下文，长文本性价比之选",
			MaxTokens:   131072,
			InputCost:   0.80,
			OutputCost:  1.00,
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "doubao-vision-pro-32k",
			Name:        "Doubao Vision Pro 32K",
			Description: "豆包视觉专业版，支持图像理解",
			MaxTokens:   32768,
			InputCost:   3.00,
			OutputCost:  9.00,
			Features:    []string{llm.FeatureVision, llm.FeatureStreaming},
		},
		{
			ID:          "doubao-character-pro-32k",
			Name:        "Doubao Character Pro 32K",
			Description: "豆包角色扮演版，适合对话场景",
			MaxTokens:   32768,
			InputCost:   0.80,
			OutputCost:  2.00,
			Features:    []string{llm.FeatureStreaming},
		},
	}
}

// CountTokens 计算消息的 Token 数量（简化实现）
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	// 简化估算：中文约 1.5 字符一个 token，英文约 4 字符一个 token
	// 这里使用保守估计
	var total int
	for _, msg := range messages {
		// 假设混合语言，使用 2.5 字符每 token
		total += len(msg.Content) * 2 / 5
	}
	return total, nil
}

// setHeaders 设置请求头
func (p *Provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}

// buildRequestBody 构建请求体
func (p *Provider) buildRequestBody(req llm.CompletionRequest, stream bool) ([]byte, error) {
	// 如果设置了端点 ID，使用端点 ID 作为 model
	model := req.Model
	if p.endpointID != "" {
		model = p.endpointID
	}

	payload := map[string]any{
		"model":    model,
		"messages": openai.ConvertMessages(req.Messages),
		"stream":   stream,
	}

	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		payload["tool_choice"] = req.ToolChoice
	}
	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		payload["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		payload["stop"] = req.Stop
	}

	// ResponseFormat 支持（兼容 OpenAI 格式）
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		payload["response_format"] = map[string]any{"type": "json_object"}
	}

	return json.Marshal(payload)
}

// 豆包 API 响应结构（兼容 OpenAI 格式）
type arkResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *arkResponse) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Created: resp.Created,
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		result.Content = choice.Message.Content
		result.FinishReason = choice.FinishReason

		for _, tc := range choice.Message.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
				ID:        tc.ID,
				Type:      tc.Type,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}

	return result
}

// 确保实现了 Provider 接口
var _ llm.Provider = (*Provider)(nil)
