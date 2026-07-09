// Package ollama provides Ollama local LLM provider implementation.
package ollama

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const (
	defaultBaseURL = "http://localhost:11434"
	defaultModel   = "llama3.2"

	defaultOllamaNumCtx             = 32768
	maxAutomaticNumCtx              = 32768
	defaultModelMaxTokens           = 32768
	ollamaDefaultResponseHeaderWait = 120 * time.Second
	ollamaStreamResponseHeaderWait  = 10 * time.Minute

	// defaultKeepAlive 每次请求随发的模型驻留时长（BUG-20260710）。
	// Ollama 服务端默认仅 5 分钟：空闲即卸载模型、KV 前缀缓存全丢——纯 CPU 机器上
	// 大 system prompt（真机取证 ~7.9k token @ 23 tok/s ≈ 344s）每次都要冷 prefill。
	// 请求级 keep_alive 每次刷新驻留窗口，让模型与 KV 缓存跨对话间隙存活。
	defaultKeepAlive = "30m"
)

// Provider 实现 Ollama LLM 提供者
type Provider struct {
	baseURL          string
	model            string
	keepAlive        string // 请求级模型驻留时长（空串=不下发，回落 Ollama 服务端默认）
	httpClient       *http.Client
	streamHTTPClient *http.Client
	transport        *transport.Transport
	streamTransport  *transport.Transport
	policy           *llm.NetworkPolicy
	models           []llm.ModelInfo // 缓存的模型列表
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

// WithKeepAlive 设置请求级模型驻留时长（如 "30m"/"2h"；BUG-20260710，见 defaultKeepAlive）。
func WithKeepAlive(d string) Option {
	return func(p *Provider) {
		p.keepAlive = d
	}
}

// WithHTTPClient 设置 HTTP 客户端
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
		p.streamHTTPClient = client
	}
}

// WithNetworkPolicy 设置上游网络出口约束。
//
// Ollama 默认允许本地 HTTP/私网访问；传入该选项可收紧到调用方指定策略。
func WithNetworkPolicy(policy llm.NetworkPolicy) Option {
	return func(p *Provider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

// New 创建 Ollama Provider
func New(opts ...Option) *Provider {
	defaultPolicy := llm.NetworkPolicy{AllowHTTP: true, AllowPrivate: true}
	p := &Provider{
		baseURL:   defaultBaseURL,
		model:     defaultModel,
		keepAlive: defaultKeepAlive,
		// 不设全局 Timeout — 流式请求的超时由调用方 context 控制
		// http.Client.Timeout 对流式响应会在整个读取期间生效，
		// 本地模型推理可能需要数分钟
		httpClient:       httpx.RawClient(httpx.WithResponseHeaderTimeout(ollamaDefaultResponseHeaderWait)),
		streamHTTPClient: httpx.RawClient(httpx.WithResponseHeaderTimeout(ollamaStreamResponseHeaderWait)),
		policy:           &defaultPolicy,
	}

	// 从环境变量读取
	if url := os.Getenv("OLLAMA_HOST"); url != "" {
		p.baseURL = url
	}
	if model := os.Getenv("OLLAMA_MODEL"); model != "" {
		p.model = model
	}

	for _, opt := range opts {
		opt(p)
	}
	p.transport = transport.NewTransport(p.httpClient, p.policy)
	p.streamTransport = transport.NewTransport(p.streamHTTPClient, p.policy)

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "ollama"
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

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/chat", body, false)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	var result ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return p.parseResponse(&result, req.Model), nil
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

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/chat", body, true)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}

	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.CustomFormat).SetParser(&ollamaStreamParser{}), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	// 如果已缓存，直接返回
	if len(p.models) > 0 {
		return p.models
	}

	// 尝试从 Ollama 获取本地模型列表
	models, err := p.fetchLocalModels()
	if err == nil && len(models) > 0 {
		p.models = models
		return p.models
	}

	// 返回常见模型的默认列表
	return []llm.ModelInfo{
		{
			ID:          "llama3.2",
			Name:        "Llama 3.2",
			Description: "Meta's Llama 3.2 model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "llama3.2:1b",
			Name:        "Llama 3.2 1B",
			Description: "Meta's Llama 3.2 1B parameter model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "llama3.1",
			Name:        "Llama 3.1",
			Description: "Meta's Llama 3.1 model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "qwen2.5",
			Name:        "Qwen 2.5",
			Description: "Alibaba's Qwen 2.5 model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "qwen2.5:7b",
			Name:        "Qwen 2.5 7B",
			Description: "Alibaba's Qwen 2.5 7B model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "mistral",
			Name:        "Mistral",
			Description: "Mistral AI's model",
			MaxTokens:   32768,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "codellama",
			Name:        "Code Llama",
			Description: "Meta's Code Llama model for coding tasks",
			MaxTokens:   16384,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "deepseek-coder-v2",
			Name:        "DeepSeek Coder V2",
			Description: "DeepSeek's coding model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "llava",
			Name:        "LLaVA",
			Description: "Vision-language model",
			MaxTokens:   4096,
			Features:    []string{llm.FeatureVision, llm.FeatureStreaming},
		},
	}
}

// fetchLocalModels 从 Ollama 获取本地模型列表
func (p *Provider) fetchLocalModels() ([]llm.ModelInfo, error) {
	resp, err := p.doRequest(context.Background(), http.MethodGet, p.baseURL+"/api/tags", nil, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			Model        string   `json:"model"`
			Name         string   `json:"name"`
			Size         int64    `json:"size"`
			ModifiedAt   string   `json:"modified_at"`
			Capabilities []string `json:"capabilities"`
			Details      struct {
				Format            string `json:"format"`
				Family            string `json:"family"`
				ParameterSize     string `json:"parameter_size"`
				QuantizationLevel string `json:"quantization_level"`
				ContextLength     int    `json:"context_length"`
			} `json:"details"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	models := make([]llm.ModelInfo, 0, len(result.Models))
	for i := range result.Models {
		m := &result.Models[i]
		modelID := strings.TrimSpace(m.Name)
		if modelID == "" {
			modelID = strings.TrimSpace(m.Model)
		}
		if modelID == "" {
			continue
		}
		capabilities := m.Capabilities
		if len(capabilities) == 0 {
			if showCaps, err := p.fetchModelCapabilities(context.Background(), modelID); err == nil {
				capabilities = showCaps
			}
		}
		maxTokens := m.Details.ContextLength
		if maxTokens <= 0 {
			maxTokens = defaultModelMaxTokens
		}
		models = append(models, llm.ModelInfo{
			ID:          modelID,
			Name:        modelID,
			Description: fmt.Sprintf("%s model (%s)", m.Details.Family, m.Details.ParameterSize),
			MaxTokens:   maxTokens,
			Features:    ollamaCapabilitiesToFeatures(capabilities),
		})
	}

	return models, nil
}

func (p *Provider) fetchModelCapabilities(ctx context.Context, model string) ([]string, error) {
	body, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return nil, err
	}
	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/show", body, false)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			return
		}
	}()

	var result struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Capabilities, nil
}

func ollamaCapabilitiesToFeatures(capabilities []string) []string {
	features := []string{llm.FeatureStreaming}
	seen := map[string]bool{llm.FeatureStreaming: true}
	add := func(feature string) {
		if feature == "" || seen[feature] {
			return
		}
		features = append(features, feature)
		seen[feature] = true
	}

	for _, capability := range capabilities {
		switch strings.ToLower(strings.TrimSpace(capability)) {
		case "vision":
			add(llm.FeatureVision)
		case "tools", "tool", "function", "functions":
			add(llm.FeatureFunctions)
		case "embedding", "embeddings", "embed":
			add(llm.FeatureEmbedding)
		}
	}
	return features
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

// buildRequestBody 构建请求体
func (p *Provider) buildRequestBody(req llm.CompletionRequest, stream bool) ([]byte, error) {
	messages, err := convertMessagesForOllama(req.Messages)
	if err != nil {
		return nil, err
	}

	payload := map[string]any{
		"model":    req.Model,
		"messages": messages,
		"stream":   stream,
	}
	if think, ok := ollamaThinkFromMetadata(req.Metadata); ok {
		payload["think"] = think
	}
	// 模型驻留（BUG-20260710）：metadata 按请求覆盖 > 选项/默认；空串=不下发
	keepAlive := p.keepAlive
	if v, ok := req.Metadata["keep_alive"].(string); ok && strings.TrimSpace(v) != "" {
		keepAlive = strings.TrimSpace(v)
	}
	if keepAlive != "" {
		payload["keep_alive"] = keepAlive
	}

	// Ollama 使用 options 嵌套参数
	options := make(map[string]any)
	if req.Temperature != nil {
		options["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		options["top_p"] = *req.TopP
	}
	if req.MaxTokens > 0 {
		options["num_predict"] = req.MaxTokens
	}
	if len(req.Stop) > 0 {
		options["stop"] = req.Stop
	}
	if numCtx := p.numCtxForRequest(req); numCtx > 0 {
		options["num_ctx"] = numCtx
	}

	if len(options) > 0 {
		payload["options"] = options
	}

	// 工具支持（部分模型支持）
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}

	// ResponseFormat 支持
	// Ollama 使用 format 参数指定输出格式
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object", "json_schema":
			payload["format"] = "json"
		}
	}

	return json.Marshal(payload)
}

func (p *Provider) numCtxForRequest(req llm.CompletionRequest) int {
	if numCtx, ok := ollamaNumCtxFromMetadata(req.Metadata); ok {
		return numCtx
	}
	for _, model := range p.models {
		if model.ID == req.Model || model.Name == req.Model {
			return clampAutomaticNumCtx(model.MaxTokens)
		}
	}
	return defaultOllamaNumCtx
}

func clampAutomaticNumCtx(contextLength int) int {
	if contextLength <= 0 {
		return defaultOllamaNumCtx
	}
	if contextLength > maxAutomaticNumCtx {
		return maxAutomaticNumCtx
	}
	return contextLength
}

func ollamaNumCtxFromMetadata(metadata map[string]any) (int, bool) {
	if len(metadata) == 0 {
		return 0, false
	}
	for _, key := range []string{"num_ctx", "ollama_num_ctx", "context_length", "max_context_tokens"} {
		if value, exists := metadata[key]; exists {
			if numCtx, ok := positiveInt(value); ok {
				return numCtx, true
			}
		}
	}
	return 0, false
}

func positiveInt(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, v > 0
	case int64:
		return int(v), v > 0
	case int32:
		return int(v), v > 0
	case float64:
		return int(v), v > 0
	case float32:
		return int(v), v > 0
	case json.Number:
		n, err := strconv.Atoi(v.String())
		return n, err == nil && n > 0
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		return n, err == nil && n > 0
	default:
		return 0, false
	}
}

func convertMessagesForOllama(messages []llm.Message) ([]map[string]any, error) {
	converted := openai.ConvertMessages(messages)
	for _, msg := range converted {
		// tool_calls[].function.arguments 归一化：OpenAI 协议把参数序列化为 JSON
		// 字符串，但 Ollama 原生 /api/chat 要求它是 JSON 对象，收到字符串会返回
		// 400 "Value looks like object, but can't find closing '}' symbol"。
		// 必须在 content 处理前跑——assistant+tool_calls 消息的 content 为 nil，
		// 会走下面的 continue 分支跳过。
		normalizeOllamaToolCallArguments(msg)

		contentParts, ok := contentPartsFromOpenAIMessage(msg["content"])
		if !ok {
			continue
		}

		texts := make([]string, 0, len(contentParts))
		images := make([]string, 0)
		for _, part := range contentParts {
			partType, ok := part["type"].(string)
			if !ok {
				continue
			}
			switch partType {
			case "image_url":
				imageURL, ok := part["image_url"].(map[string]any)
				if !ok {
					return nil, fmt.Errorf("ollama image_url part missing image_url")
				}
				rawURL, ok := imageURL["url"].(string)
				if !ok {
					return nil, fmt.Errorf("ollama image_url part missing url")
				}
				payload, err := ollamaImagePayload(rawURL)
				if err != nil {
					return nil, err
				}
				images = append(images, payload)
			default:
				text, ok := part["text"].(string)
				if ok && text != "" {
					texts = append(texts, text)
				}
			}
		}

		msg["content"] = strings.Join(texts, "\n")
		if len(images) > 0 {
			msg["images"] = images
		}
	}
	return converted, nil
}

// normalizeOllamaToolCallArguments 把消息里 tool_calls[].function.arguments 从
// OpenAI 风格的 JSON 字符串转成 Ollama 原生要求的 JSON 对象。
//
// openai.ConvertMessages 遵循 OpenAI /v1/chat/completions 协议，arguments 恒为字符串；
// Ollama /api/chat 却要求对象，收到字符串会返回 400。这里就地改写 map 值：
//   - 合法 JSON 对象字符串 → 解析为 map[string]any
//   - 空串（无参工具）或无法解析为对象 → 归一化为空对象 {}（避免透传字符串触发 400）
func normalizeOllamaToolCallArguments(msg map[string]any) {
	calls, ok := msg["tool_calls"].([]map[string]any)
	if !ok {
		return
	}
	for _, call := range calls {
		fn, ok := call["function"].(map[string]any)
		if !ok {
			continue
		}
		argStr, ok := fn["arguments"].(string)
		if !ok {
			// 已是对象或缺省，无需处理
			continue
		}
		obj := map[string]any{}
		if trimmed := strings.TrimSpace(argStr); trimmed != "" {
			if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
				obj = map[string]any{}
			}
		}
		fn["arguments"] = obj
	}
}

func contentPartsFromOpenAIMessage(content any) ([]map[string]any, bool) {
	switch parts := content.(type) {
	case []map[string]any:
		return parts, true
	case []any:
		result := make([]map[string]any, 0, len(parts))
		for _, part := range parts {
			m, ok := part.(map[string]any)
			if !ok {
				return nil, false
			}
			result = append(result, m)
		}
		return result, true
	default:
		return nil, false
	}
}

func ollamaImagePayload(raw string) (string, error) {
	payload := strings.TrimSpace(raw)
	if payload == "" {
		return "", fmt.Errorf("ollama image content is empty")
	}
	if strings.HasPrefix(strings.ToLower(payload), "data:") {
		comma := strings.Index(payload, ",")
		if comma < 0 {
			return "", fmt.Errorf("ollama image data URI missing base64 payload")
		}
		payload = strings.TrimSpace(payload[comma+1:])
	}
	if _, err := base64.StdEncoding.DecodeString(payload); err != nil {
		return "", fmt.Errorf("ollama image content must be base64 or data URI: %w", err)
	}
	return payload, nil
}

func ollamaThinkFromMetadata(metadata map[string]any) (any, bool) {
	if len(metadata) == 0 {
		return false, false
	}
	for _, key := range []string{"thinking", "think"} {
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
			case "low", "medium", "high":
				return strings.ToLower(strings.TrimSpace(v)), true
			}
		}
	}
	return false, false
}

// Ollama API 响应结构
type ollamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		Thinking  string `json:"thinking,omitempty"`
		ToolCalls []struct {
			Function struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls,omitempty"`
	} `json:"message"`
	Error              string `json:"error,omitempty"`
	Done               bool   `json:"done"`
	DoneReason         string `json:"done_reason"`
	TotalDuration      int    `json:"total_duration"`
	LoadDuration       int    `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int    `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int    `json:"eval_duration"`
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *ollamaResponse, model string) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:      resp.CreatedAt,
		Model:   model,
		Content: resp.Message.Content,
		Usage: llm.Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
	}

	if resp.Done {
		result.FinishReason = "stop"
	}

	// 解析工具调用
	for i, tc := range resp.Message.ToolCalls {
		args, _ := json.Marshal(tc.Function.Arguments)
		result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
			ID:        fmt.Sprintf("call_%d", i),
			Type:      "function",
			Name:      tc.Function.Name,
			Arguments: string(args),
		})
	}

	return result
}

type ollamaStreamParser struct{}

func (p *ollamaStreamParser) Parse(data []byte) (*streamx.Chunk, error) {
	var resp ollamaResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("ollama stream error: %s", resp.Error)
	}

	chunk := &streamx.Chunk{
		ID:        resp.CreatedAt,
		Model:     resp.Model,
		Role:      resp.Message.Role,
		Content:   resp.Message.Content,
		Reasoning: resp.Message.Thinking,
		Raw:       data,
	}
	if resp.Done {
		chunk.FinishReason = resp.DoneReason
		if chunk.FinishReason == "" {
			chunk.FinishReason = "stop"
		}
	}
	for i, tc := range resp.Message.ToolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			return nil, err
		}
		chunk.ToolCalls = append(chunk.ToolCalls, streamx.ToolCall{
			Index:     i,
			ID:        fmt.Sprintf("call_%d", i),
			Type:      "function",
			Name:      tc.Function.Name,
			Arguments: string(args),
		})
	}
	return chunk, nil
}

func (p *ollamaStreamParser) IsDone(data []byte) bool {
	var resp struct {
		Done bool `json:"done"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return false
	}
	return resp.Done
}

// Ping 检查 Ollama 服务是否可用
func (p *Provider) Ping(ctx context.Context) error {
	resp, err := p.doRequest(ctx, http.MethodGet, p.baseURL+"/api/tags", nil, false)
	if err != nil {
		return fmt.Errorf("ollama service unavailable: %w", err)
	}
	defer resp.Body.Close()

	return nil
}

// PullModel 拉取模型
func (p *Provider) PullModel(ctx context.Context, model string) error {
	payload := map[string]any{
		"name":   model,
		"stream": false,
	}
	body, _ := json.Marshal(payload)

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/pull", body, false)
	if err != nil {
		return fmt.Errorf("pull model failed: %w", err)
	}
	defer resp.Body.Close()

	return nil
}

func (p *Provider) doRequest(ctx context.Context, method, url string, body []byte, stream bool) (*http.Response, error) {
	client := p.httpClient
	requestTransport := p.transport
	if stream {
		client = p.streamHTTPClient
		requestTransport = p.streamTransport
	}
	cfg := transport.Request{
		Provider:  p.Name(),
		Action:    strings.TrimPrefix(strings.TrimPrefix(url, strings.TrimRight(p.baseURL, "/")), "/"),
		Method:    method,
		URL:       url,
		Body:      body,
		Client:    client,
		Transport: requestTransport,
	}
	if body != nil {
		cfg.SetHeaders = func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
		}
	}
	if stream {
		cfg.StreamIdle = 10 * time.Minute
	}
	return transport.Do(ctx, cfg)
}

// 确保实现了 Provider 接口
// EmbeddingProvider 接口验证在 embedding.go 中
var _ llm.Provider = (*Provider)(nil)
