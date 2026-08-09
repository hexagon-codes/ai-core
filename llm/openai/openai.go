// Package openai provides OpenAI LLM provider implementation.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
	"github.com/hexagon-codes/toolkit/util/logger"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	defaultModel   = "gpt-4o"
)

// Provider 实现 OpenAI LLM 提供者
type Provider struct {
	apiKey         string
	baseURL        string
	model          string
	httpClient     *http.Client
	requestTimeout time.Duration
	streamIdle     time.Duration
	headers        map[string]string
	networkPolicy  *llm.NetworkPolicy
	transport      *transport.Transport
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

// WithRequestTimeout 设置单次上游请求超时。
//
// 默认不设置请求级总超时，沿用调用方 context；流式请求如果设置该值，也会受该超时约束。
func WithRequestTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		p.requestTimeout = timeout
	}
}

// WithStreamIdleTimeout 设置流式响应单次读取空闲超时。
//
// 默认不设置，适合可能长时间思考才输出的模型；需要防止静默卡死时显式启用。
func WithStreamIdleTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		p.streamIdle = timeout
	}
}

// WithHeaders 设置额外请求头。认证和传输级 header 会被共享 transport 拒绝覆盖。
func WithHeaders(headers map[string]string) Option {
	return func(p *Provider) {
		p.headers = transport.CloneHeaders(headers)
	}
}

// WithNetworkPolicy 设置上游网络出口约束。默认不启用，保持本地网关/测试兼容。
func WithNetworkPolicy(policy llm.NetworkPolicy) Option {
	return func(p *Provider) {
		p.networkPolicy = transport.CloneNetworkPolicy(&policy)
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
		httpClient: httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
	}

	for _, opt := range opts {
		opt(p)
	}
	p.transport = transport.NewTransport(p.httpClient, p.networkPolicy)

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

	result, err := doJSONWithDecodeRetry[openAIResponse](ctx, p, "complete", "/chat/completions", body)
	if err != nil {
		var decodeErr *responseDecodeError
		if errors.As(err, &decodeErr) {
			logger.WarnContext(ctx, "openai response json decode failed",
				logger.Component("openai"),
				logger.Action("complete"),
				logger.String("model", req.Model),
				logger.Err(decodeErr.Err),
			)
			return nil, decodeErr.Err
		}
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

	resp, err := p.doRequest(ctx, "stream", "/chat/completions", body)
	if err != nil {
		return nil, err
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

func (p *Provider) doRequest(ctx context.Context, action, path string, body []byte) (*http.Response, error) {
	streamIdle := time.Duration(0)
	if action == "stream" {
		streamIdle = p.streamIdle
	}
	return transport.Do(ctx, transport.Request{
		Provider:      "openai",
		Action:        action,
		Method:        http.MethodPost,
		URL:           p.baseURL + path,
		Body:          body,
		Client:        p.httpClient,
		Transport:     p.transport,
		SetHeaders:    p.setHeaders,
		Headers:       p.headers,
		NetworkPolicy: p.networkPolicy,
		Timeout:       p.requestTimeout,
		StreamIdle:    streamIdle,
	})
}

type responseDecodeError struct {
	Err error
}

func (e *responseDecodeError) Error() string {
	if e == nil || e.Err == nil {
		return ""
	}
	return e.Err.Error()
}

func (e *responseDecodeError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func doJSONWithDecodeRetry[T any](ctx context.Context, p *Provider, action, path string, body []byte) (T, error) {
	var zero T
	policy := transport.DefaultRetryPolicy()
	if transport.OperationSafetyFromContext(ctx) == transport.OperationSafetyNonIdempotent {
		policy.MaxAttempts = 1
	}
	for attempt := 1; ; attempt++ {
		resp, err := p.doRequest(ctx, action, path, body)
		if err != nil {
			return zero, err
		}

		var result T
		decodeErr := json.NewDecoder(resp.Body).Decode(&result)
		if closeErr := resp.Body.Close(); decodeErr == nil && closeErr != nil {
			decodeErr = closeErr
		}
		if decodeErr == nil {
			return result, nil
		}
		if attempt >= policy.MaxAttempts || !isRetryableResponseDecodeError(ctx, decodeErr) {
			return zero, &responseDecodeError{Err: decodeErr}
		}
		if err := waitForDecodeRetry(ctx, decodeRetryDelay(policy, attempt)); err != nil {
			return zero, err
		}
	}
}

func isRetryableResponseDecodeError(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	return strings.Contains(strings.ToLower(err.Error()), "unexpected eof")
}

func decodeRetryDelay(policy transport.RetryPolicy, attempt int) time.Duration {
	delay := policy.BaseDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(float64(delay) * policy.Multiplier)
		if delay >= policy.MaxDelay {
			return policy.MaxDelay
		}
	}
	if delay > policy.MaxDelay {
		return policy.MaxDelay
	}
	return delay
}

func waitForDecodeRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		if ctx == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
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
	if effort, ok := openAIReasoningEffortFromMetadata(
		req.Metadata,
		req.Model,
		req.ReasoningPolicyScope,
	); ok {
		// OpenAI reasoning models use reasoning_effort. This is a request
		// capability, not a host-name capability: OpenAI-compatible gateways
		// are commonly deployed on loopback/private URLs and must not silently
		// lose the parameter merely because their domain is unknown.
		payload["reasoning_effort"] = effort
	} else if thinking, ok := openAIEnableThinkingFromMetadata(req.Metadata); ok && supportsEnableThinking(p.baseURL) {
		// Qwen-compatible vendors use the separate enable_thinking dialect.
		payload["enable_thinking"] = thinking
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

func openAIEnableThinkingFromMetadata(metadata map[string]any) (bool, bool) {
	if len(metadata) == 0 {
		return false, false
	}
	for _, key := range []string{"enable_thinking", "thinking", "think"} {
		value, exists := metadata[key]
		if !exists {
			continue
		}
		switch v := value.(type) {
		case bool:
			return v, true
		case string:
			switch strings.ToLower(strings.TrimSpace(v)) {
			case "on", "true", "1", "yes", "enabled":
				return true, true
			case "off", "false", "0", "no", "disabled":
				return false, true
			}
		}
	}
	return false, false
}

func openAIReasoningEffortFromMetadata(
	metadata map[string]any,
	model string,
	scope llm.ReasoningPolicyScope,
) (string, bool) {
	if len(metadata) == 0 {
		return "", false
	}

	// An explicit standard parameter always wins over the coarse on/off UI
	// switch. Passing it through also supports compatible providers whose
	// model names are not known to this package.
	for _, key := range []string{"reasoning_effort", "reasoningEffort"} {
		raw, exists := metadata[key]
		if !exists {
			continue
		}
		effort, ok := normalizeOpenAIReasoningEffort(raw)
		return effort, ok
	}

	if scope != llm.ReasoningPolicyScopeStructuredVisionRecognition {
		return "", false
	}
	if !supportsOpenAIReasoningEffort(model) {
		return "", false
	}
	enabled, exists := openAIEnableThinkingFromMetadata(metadata)
	if !exists {
		return "", false
	}
	if enabled {
		return "high", true
	}
	return minimumOpenAIReasoningEffort(model), true
}

func normalizeOpenAIReasoningEffort(raw any) (string, bool) {
	value, ok := raw.(string)
	if !ok {
		return "", false
	}
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "x-high", "x_high":
		value = "xhigh"
	}
	switch value {
	case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		return value, true
	default:
		return "", false
	}
}

func supportsOpenAIReasoningEffort(model string) bool {
	model = canonicalOpenAIModelID(model)
	if strings.HasPrefix(model, "gpt-5") {
		return true
	}
	for _, prefix := range []string{"o1", "o3", "o4"} {
		if model == prefix || strings.HasPrefix(model, prefix+"-") {
			return true
		}
	}
	return strings.HasPrefix(model, "codex-")
}

func minimumOpenAIReasoningEffort(model string) string {
	model = canonicalOpenAIModelID(model)
	if minor, ok := gpt5MinorVersion(model); ok && minor >= 1 {
		return "none"
	}
	if strings.HasPrefix(model, "gpt-5") {
		// Original GPT-5 supports minimal but not none.
		return "minimal"
	}
	// Older o-series/Codex reasoning models cannot fully disable reasoning.
	return "low"
}

func canonicalOpenAIModelID(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if slash := strings.LastIndex(model, "/"); slash >= 0 {
		model = model[slash+1:]
	}
	return model
}

func gpt5MinorVersion(model string) (int, bool) {
	const prefix = "gpt-5."
	if !strings.HasPrefix(model, prefix) {
		return 0, false
	}
	rest := model[len(prefix):]
	minor := 0
	digits := 0
	for _, r := range rest {
		if r < '0' || r > '9' {
			break
		}
		minor = minor*10 + int(r-'0')
		digits++
	}
	return minor, digits > 0
}

func supportsEnableThinking(baseURL string) bool {
	base := strings.ToLower(baseURL)
	return strings.Contains(base, "siliconflow") ||
		strings.Contains(base, "dashscope") ||
		strings.Contains(base, "aliyuncs")
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

		// 兜底：部分模型（DeepSeek 经 OpenAI 兼容网关）把工具调用以私有标记内嵌在 content
		// 而非结构化 tool_calls 字段，网关未翻译 → tool_calls=[] 且标记泄漏进正文，运行时当
		// 普通文本丢弃用户意图。仅当结构化字段为空时还原，避免误改模型既有合法 content。
		if len(result.ToolCalls) == 0 {
			if calls, cleaned, ok := streamx.RecoverInlineToolCalls(result.Content); ok {
				result.ToolCalls = calls
				result.Content = cleaned
				if result.FinishReason == "" || result.FinishReason == "stop" {
					result.FinishReason = "tool_calls"
				}
			}
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
