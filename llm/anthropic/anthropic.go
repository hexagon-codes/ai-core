// Package anthropic provides Anthropic Claude LLM provider implementation.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const (
	defaultBaseURL   = "https://api.anthropic.com/v1"
	defaultModel     = "claude-sonnet-4-20250514"
	anthropicVersion = "2023-06-01"
)

// Provider 实现 Anthropic Claude LLM 提供者
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

// New 创建 Anthropic Provider
// apiKey 可以为空，会从环境变量 ANTHROPIC_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		model:   defaultModel,
		// 不设全局 Timeout — 流式请求的超时由调用方 context 控制
		// http.Client.Timeout 对流式响应会在整个读取期间生效，
		// thinking 模型可能需要数分钟
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
	return "anthropic"
}

// Complete 执行非流式补全请求
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}
	_, reasoningReceipt, explicitReasoning, err := anthropicReasoningConfig(req.Metadata)
	if err != nil {
		if explicitReasoning {
			llm.PublishReasoningReceipt(req.Metadata, reasoningReceipt)
		}
		return nil, err
	}

	body, systemPrompt, err := p.buildRequestBody(req, false)
	if err != nil {
		return nil, err
	}

	resp, err := p.doRequest(ctx, "complete", "/messages", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	parsed := p.parseResponse(&result, systemPrompt)
	if explicitReasoning {
		receipt := reasoningReceipt.WithEvidence(true, anthropicResponseHasReasoning(&result), 0)
		parsed.ReasoningReceipt = &receipt
		llm.PublishReasoningReceipt(req.Metadata, receipt)
	}
	return parsed, nil
}

// Stream 执行流式补全请求
func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if req.Model == "" {
		req.Model = p.model
	}
	_, reasoningReceipt, explicitReasoning, err := anthropicReasoningConfig(req.Metadata)
	if err != nil {
		if explicitReasoning {
			llm.PublishReasoningReceipt(req.Metadata, reasoningReceipt)
		}
		return nil, err
	}

	body, _, err := p.buildRequestBody(req, true)
	if err != nil {
		return nil, err
	}

	resp, err := p.doRequest(ctx, "stream", "/messages", body)
	if err != nil {
		return nil, err
	}

	parser := &anthropicReasoningStreamParser{
		base:     &streamx.ClaudeParser{},
		close:    resp.Body.Close,
		metadata: req.Metadata,
	}
	if explicitReasoning {
		parser.receipt = reasoningReceipt.WithEvidence(true, false, 0)
		llm.PublishReasoningReceipt(req.Metadata, parser.receipt)
	}
	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.ClaudeFormat).SetParser(parser), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "claude-opus-4-20250514",
			Name:        "Claude Opus 4",
			Description: "Most capable Claude model for complex tasks",
			MaxTokens:   200000,
			InputCost:   15.00,
			OutputCost:  75.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "claude-sonnet-4-20250514",
			Name:        "Claude Sonnet 4",
			Description: "Best balance of intelligence and speed",
			MaxTokens:   200000,
			InputCost:   3.00,
			OutputCost:  15.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "claude-3-5-sonnet-20241022",
			Name:        "Claude 3.5 Sonnet",
			Description: "High performance with improved speed",
			MaxTokens:   200000,
			InputCost:   3.00,
			OutputCost:  15.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "claude-3-5-haiku-20241022",
			Name:        "Claude 3.5 Haiku",
			Description: "Fast and cost-effective",
			MaxTokens:   200000,
			InputCost:   0.80,
			OutputCost:  4.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "claude-3-opus-20240229",
			Name:        "Claude 3 Opus",
			Description: "Previous generation flagship model",
			MaxTokens:   200000,
			InputCost:   15.00,
			OutputCost:  75.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
	}
}

// CountTokens 计算消息的 Token 数量（简化实现）
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	return p.countTokens(context.Background(), messages)
}

// CountTokensContext 计算消息的 Token 数量，并在遍历消息时响应 context 取消。
func (p *Provider) CountTokensContext(ctx context.Context, messages []llm.Message) (int, error) {
	return p.countTokens(ctx, messages)
}

func (p *Provider) countTokens(ctx context.Context, messages []llm.Message) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	// Claude 的 tokenization 与 GPT 类似，约 4 字符一个 token
	var total int
	for _, msg := range messages {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		total += len(msg.Content) / 4
	}
	return total, nil
}

// setHeaders 设置请求头
func (p *Provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
}

func (p *Provider) doRequest(ctx context.Context, action, path string, body []byte) (*http.Response, error) {
	streamIdle := time.Duration(0)
	if action == "stream" {
		streamIdle = p.streamIdle
	}
	return transport.Do(ctx, transport.Request{
		Provider:      "anthropic",
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

// buildRequestBody 构建请求体
// Anthropic 的 API 格式与 OpenAI 不同，需要特殊处理
func (p *Provider) buildRequestBody(req llm.CompletionRequest, stream bool) ([]byte, string, error) {
	// 分离系统消息和用户消息
	var systemPrompt string
	var messages []map[string]any

	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			systemPrompt = msg.Content
			continue
		}

		// Anthropic 使用不同的消息格式
		m := map[string]any{
			"role": convertRole(msg.Role),
		}

		content, err := anthropicMessageContent(msg)
		if err != nil {
			return nil, systemPrompt, err
		}
		m["content"] = content

		messages = append(messages, m)
	}

	payload := map[string]any{
		"model":      req.Model,
		"messages":   messages,
		"max_tokens": 4096, // Anthropic 要求必须指定
		"stream":     stream,
	}

	if systemPrompt != "" {
		payload["system"] = systemPrompt
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
		payload["stop_sequences"] = req.Stop
	}
	if thinking, _, explicit, err := anthropicReasoningConfig(req.Metadata); err != nil {
		return nil, systemPrompt, err
	} else if explicit {
		payload["thinking"] = thinking
	}

	// 工具支持
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = map[string]any{
				"name":         tool.Function.Name,
				"description":  tool.Function.Description,
				"input_schema": anthropicInputSchema(tool.Function.Parameters),
			}
		}
		payload["tools"] = tools
	}

	body, err := json.Marshal(payload)
	return body, systemPrompt, err
}

func anthropicMessageContent(msg llm.Message) ([]map[string]any, error) {
	if !msg.HasMultiContent() {
		return []map[string]any{
			{"type": "text", "text": msg.Content},
		}, nil
	}

	blocks := make([]map[string]any, 0, len(msg.MultiContent))
	for _, part := range msg.MultiContent {
		switch part.Type {
		case "", "text":
			blocks = append(blocks, map[string]any{
				"type": "text",
				"text": part.Text,
			})
		case "image_url":
			if part.ImageURL == nil || strings.TrimSpace(part.ImageURL.URL) == "" {
				return nil, fmt.Errorf("anthropic image content missing image_url")
			}
			source, err := anthropicImageSource(part.ImageURL.URL)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, map[string]any{
				"type":   "image",
				"source": source,
			})
		default:
			blocks = append(blocks, map[string]any{
				"type": "text",
				"text": part.Text,
			})
		}
	}
	return blocks, nil
}

func anthropicImageSource(rawURL string) (map[string]any, error) {
	rawURL = strings.TrimSpace(rawURL)
	if !strings.HasPrefix(strings.ToLower(rawURL), "data:") {
		return map[string]any{
			"type": "url",
			"url":  rawURL,
		}, nil
	}

	payload := rawURL[len("data:"):]
	header, data, ok := strings.Cut(payload, ",")
	if !ok || strings.TrimSpace(data) == "" {
		return nil, fmt.Errorf("anthropic image data URI missing base64 data")
	}
	parts := strings.Split(header, ";")
	mediaType := strings.ToLower(strings.TrimSpace(parts[0]))
	if !anthropicSupportedImageMediaType(mediaType) {
		return nil, fmt.Errorf("anthropic image media type %q is not supported", mediaType)
	}
	hasBase64 := false
	for _, part := range parts[1:] {
		if strings.EqualFold(strings.TrimSpace(part), "base64") {
			hasBase64 = true
			break
		}
	}
	if !hasBase64 {
		return nil, fmt.Errorf("anthropic image data URI must be base64 encoded")
	}

	return map[string]any{
		"type":       "base64",
		"media_type": mediaType,
		"data":       data,
	}, nil
}

func anthropicSupportedImageMediaType(mediaType string) bool {
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

func anthropicInputSchema(schema *llm.Schema) any {
	if schema == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}
	return schema
}

// convertRole 转换角色名称
func convertRole(role llm.Role) string {
	switch role {
	case llm.RoleUser:
		return "user"
	case llm.RoleAssistant:
		return "assistant"
	default:
		return "user"
	}
}

// Anthropic API 响应结构
type anthropicResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Model        string             `json:"model"`
	Content      []anthropicContent `json:"content"`
	StopReason   string             `json:"stop_reason"`
	StopSequence string             `json:"stop_sequence"`
	Usage        struct {
		InputTokens int `json:"input_tokens"`
		// CacheCreationInputTokens 写入提示词缓存所消耗的输入 Token
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		// CacheReadInputTokens 命中提示词缓存、从缓存读取的输入 Token
		CacheReadInputTokens int `json:"cache_read_input_tokens"`
		OutputTokens         int `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
	ID       string `json:"id,omitempty"`
	Name     string `json:"name,omitempty"`
	Input    any    `json:"input,omitempty"`
}

func anthropicReasoningConfig(metadata map[string]any) (map[string]any, llm.ReasoningReceipt, bool, error) {
	if plan, explicit, err := llm.PlanReasoningFromMetadata(metadata, "", ""); explicit {
		if err != nil {
			return nil, plan.Receipt, true, err
		}
		thinking, ok := plan.Wire[string(llm.ReasoningDialectThinking)].(map[string]any)
		if !ok {
			return nil, plan.Receipt, true, fmt.Errorf("anthropic reasoning control must be an object")
		}
		return thinking, plan.Receipt, true, nil
	}
	raw, explicit := metadata["reasoning_v1"]
	if !explicit {
		return nil, llm.ReasoningReceipt{}, false, nil
	}
	values, ok := raw.(map[string]any)
	if !ok {
		return nil, llm.ReasoningReceipt{}, true, fmt.Errorf("anthropic reasoning_v1 must be an object")
	}
	mode, ok := values["mode"].(string)
	if !ok {
		return nil, llm.ReasoningReceipt{}, true, fmt.Errorf("anthropic reasoning mode is required")
	}
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode != "enabled" && mode != "adaptive" && mode != "disabled" {
		return nil, llm.ReasoningReceipt{}, true, fmt.Errorf("invalid anthropic reasoning mode %q", mode)
	}

	thinking := map[string]any{"type": mode}
	if mode == "enabled" {
		budget, ok := positiveReasoningBudget(values["budget_tokens"])
		if !ok {
			return nil, llm.ReasoningReceipt{}, true, fmt.Errorf("anthropic reasoning budget_tokens must be a positive integer")
		}
		thinking["budget_tokens"] = budget
	}
	receipt := llm.ReasoningReceipt{
		Version:     1,
		Enabled:     mode != "disabled",
		Support:     llm.ReasoningSupported,
		Dialect:     llm.ReasoningDialect("anthropic_thinking"),
		Sent:        true,
		Application: llm.ReasoningApplicationUnknown,
	}
	return thinking, receipt, true, nil
}

func positiveReasoningBudget(raw any) (int, bool) {
	switch value := raw.(type) {
	case int:
		return value, value > 0
	case int64:
		return int(value), value > 0
	case float64:
		budget := int(value)
		return budget, value > 0 && float64(budget) == value
	case json.Number:
		budget, err := value.Int64()
		return int(budget), err == nil && budget > 0
	default:
		return 0, false
	}
}

func anthropicResponseHasReasoning(resp *anthropicResponse) bool {
	if resp == nil {
		return false
	}
	for _, content := range resp.Content {
		if content.Type == "thinking" && content.Thinking != "" {
			return true
		}
	}
	return false
}

type anthropicReasoningStreamParser struct {
	base     *streamx.ClaudeParser
	close    func() error
	metadata map[string]any
	receipt  llm.ReasoningReceipt
}

func (p *anthropicReasoningStreamParser) Parse(data []byte) (*streamx.Chunk, error) {
	var envelope struct {
		Type  string `json:"type"`
		Delta *struct {
			Type     string `json:"type"`
			Thinking string `json:"thinking"`
		} `json:"delta,omitempty"`
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, err
	}
	if envelope.Type == "error" {
		if p.close != nil {
			_ = p.close()
		}
		if envelope.Error == nil {
			return nil, fmt.Errorf("anthropic stream error")
		}
		return nil, fmt.Errorf("anthropic stream error: %s: %s", envelope.Error.Type, envelope.Error.Message)
	}

	chunk, err := p.base.Parse(data)
	if err != nil || chunk == nil {
		return chunk, err
	}
	if envelope.Delta != nil && envelope.Delta.Type == "thinking_delta" {
		chunk.Content = ""
		chunk.Reasoning = envelope.Delta.Thinking
		if p.receipt.Version == 0 {
			p.receipt = llm.ReasoningReceipt{
				Version:     1,
				Enabled:     true,
				Support:     llm.ReasoningUnknown,
				Dialect:     llm.ReasoningDialect("anthropic_thinking"),
				Application: llm.ReasoningApplicationUnknown,
			}
		}
		p.receipt = p.receipt.WithEvidence(true, chunk.Reasoning != "", 0)
		llm.PublishReasoningReceipt(p.metadata, p.receipt)
	}
	if p.receipt.Version != 0 {
		chunk.ReasoningEvidence = llm.ReasoningEvidenceMap(p.receipt)
	}
	return chunk, nil
}

func (p *anthropicReasoningStreamParser) IsDone(data []byte) bool {
	return p.base.IsDone(data)
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *anthropicResponse, _ string) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		FinishReason: resp.StopReason,
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			// TotalTokens 保持"非缓存输入+输出"语义不变，缓存维度独立记录
			TotalTokens:         resp.Usage.InputTokens + resp.Usage.OutputTokens,
			CacheCreationTokens: resp.Usage.CacheCreationInputTokens,
			CacheReadTokens:     resp.Usage.CacheReadInputTokens,
		},
	}

	// 处理内容
	for _, content := range resp.Content {
		switch content.Type {
		case "text":
			result.Content += content.Text
		case "tool_use":
			args, _ := json.Marshal(content.Input)
			result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
				ID:        content.ID,
				Type:      "function",
				Name:      content.Name,
				Arguments: string(args),
			})
		}
	}

	return result
}

// 确保实现了 Provider 接口
var _ llm.Provider = (*Provider)(nil)
