// Package ollama provides Ollama local LLM provider implementation.
package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
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
)

// Provider 实现 Ollama LLM 提供者
type Provider struct {
	baseURL    string
	model      string
	httpClient *http.Client
	transport  *transport.Transport
	policy     *llm.NetworkPolicy
	models     []llm.ModelInfo // 缓存的模型列表
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
		baseURL: defaultBaseURL,
		model:   defaultModel,
		// 不设全局 Timeout — 流式请求的超时由调用方 context 控制
		// http.Client.Timeout 对流式响应会在整个读取期间生效，
		// 本地模型推理可能需要数分钟
		httpClient: httpx.RawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
		policy:     &defaultPolicy,
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

	// Ollama 使用类似 OpenAI 的格式，但需要自定义解析
	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.OpenAIFormat), nil
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
			Name       string `json:"name"`
			Size       int64  `json:"size"`
			ModifiedAt string `json:"modified_at"`
			Details    struct {
				Format            string `json:"format"`
				Family            string `json:"family"`
				ParameterSize     string `json:"parameter_size"`
				QuantizationLevel string `json:"quantization_level"`
			} `json:"details"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	models := make([]llm.ModelInfo, len(result.Models))
	for i, m := range result.Models {
		models[i] = llm.ModelInfo{
			ID:          m.Name,
			Name:        m.Name,
			Description: fmt.Sprintf("%s model (%s)", m.Details.Family, m.Details.ParameterSize),
			MaxTokens:   32768, // 默认值，Ollama 不提供此信息
			Features:    []string{llm.FeatureStreaming},
		}
	}

	return models, nil
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
	payload := map[string]any{
		"model":    req.Model,
		"messages": openai.ConvertMessages(req.Messages),
		"stream":   stream,
	}
	if think, ok := ollamaThinkFromMetadata(req.Metadata); ok {
		payload["think"] = think
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

func ollamaThinkFromMetadata(metadata map[string]any) (bool, bool) {
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
		ToolCalls []struct {
			Function struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done               bool `json:"done"`
	TotalDuration      int  `json:"total_duration"`
	LoadDuration       int  `json:"load_duration"`
	PromptEvalCount    int  `json:"prompt_eval_count"`
	PromptEvalDuration int  `json:"prompt_eval_duration"`
	EvalCount          int  `json:"eval_count"`
	EvalDuration       int  `json:"eval_duration"`
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
	cfg := transport.Request{
		Provider:  p.Name(),
		Action:    strings.TrimPrefix(strings.TrimPrefix(url, strings.TrimRight(p.baseURL, "/")), "/"),
		Method:    method,
		URL:       url,
		Body:      body,
		Client:    p.httpClient,
		Transport: p.transport,
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
