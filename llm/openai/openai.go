// Package openai provides OpenAI LLM provider implementation.
package openai

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
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/toolkit/net/httpx"
	"github.com/hexagon-codes/toolkit/util/logger"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	defaultModel   = "gpt-4o"
)

// Provider 实现 OpenAI LLM 提供者
type Provider struct {
	apiKey     string
	baseURL    string
	model      string
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

// WithHTTPClient 设置 HTTP 客户端
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// New 创建 OpenAI Provider
// apiKey 可以为空，会从环境变量 OPENAI_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		model:   defaultModel,
		// 不设全局 Timeout — 流式请求的超时由调用方 context 控制
		// http.Client.Timeout 对流式响应会在整个读取期间生效，
		// thinking 模型（Qwen3/DeepSeek-R1）可能需要数分钟
		httpClient: httpx.RawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "openai"
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
		logger.WarnContext(ctx, "openai http request failed",
			logger.Component("openai"),
			logger.Action("complete"),
			logger.String("model", req.Model),
			logger.String("base_url", p.baseURL),
			logger.Err(err),
		)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			logger.ErrorContext(ctx, "openai api error and body read failed",
				logger.Component("openai"),
				logger.Action("complete"),
				logger.String("model", req.Model),
				logger.Status(resp.StatusCode),
				logger.Err(readErr),
			)
			return nil, fmt.Errorf("openai api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		logger.ErrorContext(ctx, "openai api non-2xx response",
			logger.Component("openai"),
			logger.Action("complete"),
			logger.String("model", req.Model),
			logger.Status(resp.StatusCode),
			logger.String("body", string(bodyBytes)),
		)
		return nil, fmt.Errorf("openai api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		logger.WarnContext(ctx, "openai response json decode failed",
			logger.Component("openai"),
			logger.Action("complete"),
			logger.String("model", req.Model),
			logger.Err(err),
		)
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
		logger.WarnContext(ctx, "openai stream http request failed",
			logger.Component("openai"),
			logger.Action("stream"),
			logger.String("model", req.Model),
			logger.String("base_url", p.baseURL),
			logger.Err(err),
		)
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			logger.ErrorContext(ctx, "openai stream api error and body read failed",
				logger.Component("openai"),
				logger.Action("stream"),
				logger.String("model", req.Model),
				logger.Status(resp.StatusCode),
				logger.Err(readErr),
			)
			return nil, fmt.Errorf("openai api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		logger.ErrorContext(ctx, "openai stream api non-2xx response",
			logger.Component("openai"),
			logger.Action("stream"),
			logger.String("model", req.Model),
			logger.Status(resp.StatusCode),
			logger.String("body", string(bodyBytes)),
		)
		return nil, fmt.Errorf("openai api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.OpenAIFormat), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "gpt-4o",
			Name:        "GPT-4o",
			Description: "Most capable model, great for complex tasks",
			MaxTokens:   128000,
			InputCost:   2.50,
			OutputCost:  10.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gpt-4o-mini",
			Name:        "GPT-4o Mini",
			Description: "Fast and cost-effective for simpler tasks",
			MaxTokens:   128000,
			InputCost:   0.15,
			OutputCost:  0.60,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gpt-4-turbo",
			Name:        "GPT-4 Turbo",
			Description: "Previous generation flagship model",
			MaxTokens:   128000,
			InputCost:   10.00,
			OutputCost:  30.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "o1",
			Name:        "o1",
			Description: "Reasoning model for complex tasks",
			MaxTokens:   200000,
			InputCost:   15.00,
			OutputCost:  60.00,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "o3-mini",
			Name:        "o3-mini",
			Description: "Fast reasoning model",
			MaxTokens:   200000,
			InputCost:   1.10,
			OutputCost:  4.40,
			Features:    []string{llm.FeatureStreaming},
		},
	}
}

// CountTokens 计算消息的 Token 数量（简化实现）
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	// 简化估算：约 4 个字符一个 token
	var total int
	for _, msg := range messages {
		total += len(msg.Content) / 4
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
	payload := map[string]any{
		"model":    req.Model,
		"messages": convertMessages(req.Messages),
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
	if req.User != "" {
		payload["user"] = req.User
	}

	// ResponseFormat 支持
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object":
			payload["response_format"] = map[string]any{"type": "json_object"}
		case "json_schema":
			if req.ResponseFormat.JSONSchema != nil {
				rf := map[string]any{
					"type": "json_schema",
					"json_schema": map[string]any{
						"name":   req.ResponseFormat.JSONSchema.Name,
						"schema": req.ResponseFormat.JSONSchema.Schema,
						"strict": req.ResponseFormat.JSONSchema.Strict,
					},
				}
				if req.ResponseFormat.JSONSchema.Description != "" {
					rf["json_schema"].(map[string]any)["description"] = req.ResponseFormat.JSONSchema.Description
				}
				payload["response_format"] = rf
			}
		}
	}

	return json.Marshal(payload)
}

// convertMessages 兼容内部老调用点，转发到 ConvertMessages。
func convertMessages(messages []llm.Message) []map[string]any {
	return ConvertMessages(messages)
}

// ConvertMessages 将 llm.Message 序列为 OpenAI Chat Completions 兼容的 messages 数组。
//
// 实现严格遵循 OpenAI / OpenAI 兼容网关（OpenRouter / new-api / one-api 等）的协议：
//   - 多模态 MultiContent → content 数组（text / image_url parts）
//   - assistant + ToolCalls → 输出 tool_calls 数组；Content 为空时 content 设为 JSON null
//   - role=tool + ToolCallID → 输出 tool_call_id 关联原调用
//
// 所有 OpenAI 兼容的下游 provider（qwen / ollama / ark / deepseek 等）都应调用此函数，
// 避免各自实现私有 convertMessages 时漏写关键字段（BUG-20260520）。
//
// BUG-20260523 加固：在序列化前先 SanitizeToolCallSequence 剥离孤立 tool message，
// 防止上游 Anthropic 校验 "Each tool_result block must have a corresponding tool_use
// block in the previous message" 失败。任何 source（持久化历史 / 网关翻译错位 /
// 中间层 race）导致的不一致都自愈，永远输出契约级合规的序列。
func ConvertMessages(messages []llm.Message) []map[string]any {
	messages = SanitizeToolCallSequence(messages)
	result := make([]map[string]any, len(messages))
	for i, msg := range messages {
		m := map[string]any{
			"role": string(msg.Role),
		}

		// 多模态内容优先
		if msg.HasMultiContent() {
			contentParts := make([]map[string]any, len(msg.MultiContent))
			for j, part := range msg.MultiContent {
				switch part.Type {
				case "image_url":
					p := map[string]any{
						"type": "image_url",
					}
					if part.ImageURL != nil {
						imgURL := map[string]any{"url": part.ImageURL.URL}
						if part.ImageURL.Detail != "" {
							imgURL["detail"] = part.ImageURL.Detail
						}
						p["image_url"] = imgURL
					}
					contentParts[j] = p
				default: // "text"
					contentParts[j] = map[string]any{
						"type": "text",
						"text": part.Text,
					}
				}
			}
			m["content"] = contentParts
		} else {
			m["content"] = msg.Content
		}

		// assistant + tool_calls：必须输出 tool_calls 数组。
		//
		// BUG-20260523-v4：content **强制设为 nil**（不再仅 "" 时设 nil）。
		// 部分 OpenAI 兼容网关在翻 Anthropic 时遇到
		// `{role:assistant, content:"text", tool_calls:[...]}` 共存场景，
		// 只翻 text 丢 tool_use → 下一条 tool_result 找不到对应 tool_use → 400。
		// 强制 content=nil 后网关只看 tool_calls，翻译路径单一不易踩 bug。
		// trade-off：丢失 LLM 的 reasoning text 这一段（如 "我来帮你..."），
		// 但下一轮 LLM 仍能从 tool_calls 推断意图，业务无损。
		if msg.Role == llm.RoleAssistant && len(msg.ToolCalls) > 0 {
			calls := make([]map[string]any, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				calls[j] = map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": tc.Arguments,
					},
				}
			}
			m["tool_calls"] = calls
			m["content"] = nil
		}

		// role=tool：必须带 tool_call_id 关联回原调用，否则下游网关无法构造 tool_result。
		if msg.Role == llm.RoleTool && msg.ToolCallID != "" {
			m["tool_call_id"] = msg.ToolCallID
		}

		if msg.Name != "" {
			m["name"] = msg.Name
		}
		result[i] = m
	}
	return result
}

// OpenAI API 响应结构
type openAIResponse struct {
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
		// PromptTokensDetails 输入 Token 的细分，含命中提示词缓存的 Token 数
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *openAIResponse) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Created: resp.Created,
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
			// OpenAI 仅报告缓存读取（cached_tokens），无缓存写入概念
			CacheReadTokens: resp.Usage.PromptTokensDetails.CachedTokens,
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
// EmbeddingProvider 接口验证在 embedding.go 中
var _ llm.Provider = (*Provider)(nil)

// SanitizeToolCallSequence 剥离孤立 tool/tool_result message（BUG-20260523）。
//
// 上游 Anthropic 严格校验：
//
//	Each `tool_result` block must have a corresponding `tool_use` block
//	in the previous message.
//
// 任何导致 tool message 的 ToolCallID 在前面 assistant.ToolCalls 找不到对应 ID
// 的场景都会触发 400：
//   - session 历史持久化时 ToolCalls 字段漏存，下次加载只剩 tool message
//   - 网关在 OpenAI ↔ Anthropic 翻译时 race / 异常丢失 assistant tool_use
//   - 上下文压缩误剥离 assistant tool_call message
//   - 多 provider failover 切换后老 turn 的 tool message 还在缓冲
//
// 自愈契约：
//   - 维护"当前可见 tool_call_id 集合"，每见到 assistant.ToolCalls 时**累加** ID
//   - 每条 tool message 的 ToolCallID 必须在集合内，否则**丢弃**
//   - assistant message 自身（含 ToolCalls）无条件保留——它是 producer
//   - user/system message 不重置集合（同会话内 tool_use ID 跨多轮仍可被引用，
//     但通常 tool_result 紧跟 tool_use，安全）
//
// 该函数纯函数无副作用，不改原 slice；测试 + 上线长期守护契约。
func SanitizeToolCallSequence(messages []llm.Message) []llm.Message {
	if len(messages) == 0 {
		return messages
	}
	known := make(map[string]struct{})
	out := make([]llm.Message, 0, len(messages))
	for _, m := range messages {
		if m.Role == llm.RoleAssistant && len(m.ToolCalls) > 0 {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" {
					known[tc.ID] = struct{}{}
				}
			}
			out = append(out, m)
			continue
		}
		if m.Role == llm.RoleTool {
			if m.ToolCallID == "" {
				// 没 ID 也无法对账 → 丢
				continue
			}
			if _, ok := known[m.ToolCallID]; !ok {
				// 孤立 tool → 丢，避免上游 400
				continue
			}
			out = append(out, m)
			continue
		}
		out = append(out, m)
	}
	return out
}
