// Package gemini provides Google Gemini LLM provider implementation.
package gemini

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
	"github.com/hexagon-codes/ai-core/tokenizer"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
	"github.com/hexagon-codes/toolkit/util/idgen"
)

const (
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	defaultModel   = "gemini-2.0-flash"
)

// Provider 实现 Google Gemini LLM 提供者
type Provider struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
	transport  *transport.Transport
	policy     *llm.NetworkPolicy
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
func WithNetworkPolicy(policy llm.NetworkPolicy) Option {
	return func(p *Provider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

// New 创建 Gemini Provider
// apiKey 可以为空，会从环境变量 GOOGLE_API_KEY 或 GEMINI_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
	}

	p := &Provider{
		apiKey:     apiKey,
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: httpx.RawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
	}

	for _, opt := range opts {
		opt(p)
	}
	p.transport = transport.NewTransport(p.httpClient, p.policy)

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "gemini"
}

// Complete 执行非流式补全请求
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}

	body, err := p.buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	resp, err := p.doRequest(ctx, "complete", fmt.Sprintf("%s/models/%s:generateContent", p.baseURL, req.Model), body, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result geminiResponse
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

	body, err := p.buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	resp, err := p.doRequest(ctx, "stream", fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", p.baseURL, req.Model), body, true)
	if err != nil {
		return nil, err
	}

	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.GeminiFormat), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "gemini-2.0-flash",
			Name:        "Gemini 2.0 Flash",
			Description: "Next-gen fast and versatile model",
			MaxTokens:   1048576,
			InputCost:   0.10,
			OutputCost:  0.40,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-2.0-flash-thinking",
			Name:        "Gemini 2.0 Flash Thinking",
			Description: "Fast model with enhanced reasoning",
			MaxTokens:   1048576,
			InputCost:   0.10,
			OutputCost:  0.40,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-1.5-pro",
			Name:        "Gemini 1.5 Pro",
			Description: "Most capable model for complex tasks",
			MaxTokens:   2097152,
			InputCost:   1.25,
			OutputCost:  5.00,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-1.5-flash",
			Name:        "Gemini 1.5 Flash",
			Description: "Fast and efficient for most tasks",
			MaxTokens:   1048576,
			InputCost:   0.075,
			OutputCost:  0.30,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "gemini-1.5-flash-8b",
			Name:        "Gemini 1.5 Flash 8B",
			Description: "Lightweight and cost-effective",
			MaxTokens:   1048576,
			InputCost:   0.0375,
			OutputCost:  0.15,
			Features:    []string{llm.FeatureVision, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
	}
}

// CountTokens 计算消息的 Token 数量
//
// 使用 tokenizer 的 Gemini 计数器进行混合策略估算，相比旧的 len/4 整数除法
// 不会把短内容截断为 0，并且会计入多模态文本与工具调用参数（W3-54）。
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	counter := tokenizer.New(tokenizer.Gemini)

	var total int
	for _, msg := range messages {
		// 纯文本内容
		total += counter.Count(msg.Content)

		// 多模态内容中的文本部分（图片不计入文本 token）
		for _, part := range msg.MultiContent {
			if part.Type == "text" || (part.Type != "image_url" && part.Text != "") {
				total += counter.Count(part.Text)
			}
		}

		// 工具调用请求的参数（assistant 发起的 functionCall）
		for _, tc := range msg.ToolCalls {
			total += counter.Count(tc.Name)
			total += counter.Count(tc.Arguments)
		}
	}
	return total, nil
}

// buildRequestBody 构建请求体
// Gemini 使用独特的 API 格式
func (p *Provider) buildRequestBody(req llm.CompletionRequest) ([]byte, error) {
	// 转换消息格式
	var contents []geminiContent
	var systemInstruction *geminiContent

	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			systemInstruction = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
			continue
		}

		// 工具结果消息：Gemini 没有独立 "tool" 角色，需转换为携带
		// functionResponse part 的 user content，否则工具语义丢失（W3-53）。
		if msg.Role == llm.RoleTool {
			contents = append(contents, geminiContent{
				Role:  convertRole(msg.Role),
				Parts: []geminiPart{{FunctionResponse: buildFunctionResponse(msg)}},
			})
			continue
		}

		contents = append(contents, geminiContent{
			Role:  convertRole(msg.Role),
			Parts: buildParts(msg),
		})
	}

	payload := map[string]any{
		"contents": contents,
	}

	if systemInstruction != nil {
		payload["systemInstruction"] = systemInstruction
	}

	// 生成配置
	generationConfig := make(map[string]any)
	if req.MaxTokens > 0 {
		generationConfig["maxOutputTokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		generationConfig["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		generationConfig["topP"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		generationConfig["stopSequences"] = req.Stop
	}

	// ResponseFormat 支持
	// Gemini 通过 responseMimeType 和 responseSchema 控制输出格式
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object":
			generationConfig["responseMimeType"] = "application/json"
		case "json_schema":
			generationConfig["responseMimeType"] = "application/json"
			if req.ResponseFormat.JSONSchema != nil && req.ResponseFormat.JSONSchema.Schema != nil {
				generationConfig["responseSchema"] = req.ResponseFormat.JSONSchema.Schema
			}
		}
	}

	if len(generationConfig) > 0 {
		payload["generationConfig"] = generationConfig
	}

	// 工具支持
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0)
		functionDeclarations := make([]map[string]any, len(req.Tools))
		for i, tool := range req.Tools {
			functionDeclarations[i] = map[string]any{
				"name":        tool.Function.Name,
				"description": tool.Function.Description,
				"parameters":  tool.Function.Parameters,
			}
		}
		tools = append(tools, map[string]any{
			"functionDeclarations": functionDeclarations,
		})
		payload["tools"] = tools
	}

	return json.Marshal(payload)
}

// convertRole 转换角色名称
//
// Gemini 仅支持 "user" 与 "model" 两种 content 角色：
//   - RoleAssistant → "model"
//   - RoleUser / RoleTool / 其他 → "user"
//
// 注意：RoleTool 虽映射为 "user"，但其 part 必须是 functionResponse
// 而非普通文本（见 buildRequestBody），以保留工具结果语义。
func convertRole(role llm.Role) string {
	switch role {
	case llm.RoleUser:
		return "user"
	case llm.RoleAssistant:
		return "model"
	default:
		return "user"
	}
}

// buildParts 将一条消息转换为 Gemini parts 列表。
//
// 优先处理多模态内容（msg.MultiContent）：
//   - 文本 part → {text}
//   - base64 data URI 图片 → {inlineData:{mimeType,data}}
//   - http(s) URL 图片 → {fileData:{fileUri}}
//
// 当无多模态内容时，回退到纯文本 msg.Content（W3-52）。
func buildParts(msg llm.Message) []geminiPart {
	if !msg.HasMultiContent() {
		return []geminiPart{{Text: msg.Content}}
	}

	parts := make([]geminiPart, 0, len(msg.MultiContent))
	for _, part := range msg.MultiContent {
		switch part.Type {
		case "image_url":
			if part.ImageURL == nil {
				continue
			}
			parts = append(parts, buildImagePart(part.ImageURL.URL))
		default:
			// 默认按文本处理（含显式 "text" 类型）
			parts = append(parts, geminiPart{Text: part.Text})
		}
	}
	return parts
}

// buildImagePart 根据图片 URL 形态选择 Gemini 的承载方式：
//   - base64 data URI（"data:image/png;base64,..."）→ inlineData
//   - 普通 http(s) URL → fileData
func buildImagePart(url string) geminiPart {
	if mimeType, data, ok := parseDataURI(url); ok {
		return geminiPart{InlineData: &geminiInlineData{MimeType: mimeType, Data: data}}
	}
	return geminiPart{FileData: &geminiFileData{FileURI: url}}
}

// parseDataURI 解析 base64 data URI，返回 mimeType 与裸 base64 负载。
// 形如 "data:image/png;base64,iVBORw0KGgo="；非该形态时返回 ok=false。
func parseDataURI(url string) (mimeType, data string, ok bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", "", false
	}
	// 去除 "data:" 前缀后按 "," 分割元信息与负载
	meta, payload, found := strings.Cut(url[len("data:"):], ",")
	if !found {
		return "", "", false
	}
	// meta 形如 "image/png;base64"，取 ";base64" 之前部分作为 mimeType
	mimeType, _, _ = strings.Cut(meta, ";")
	return mimeType, payload, true
}

// buildFunctionResponse 将工具结果消息转换为 Gemini 的 functionResponse。
//
// Gemini 要求 functionResponse 含函数名与结构化 response 对象：
//   - name 取自 msg.ToolCallID（关联到此前的工具调用）
//   - response 将工具返回内容包装为 {"content": <内容>} 对象
func buildFunctionResponse(msg llm.Message) *geminiFunctionResponse {
	return &geminiFunctionResponse{
		Name:     msg.ToolCallID,
		Response: map[string]any{"content": msg.Content},
	}
}

// Gemini 数据结构
type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

// geminiInlineData 承载内联（base64）二进制数据，用于 Vision 等多模态请求。
// 对应 OpenAI 的 base64 data URI 图片。
type geminiInlineData struct {
	// MimeType 媒体类型，如 "image/png"、"image/jpeg"
	MimeType string `json:"mimeType"`
	// Data base64 编码的负载（不含 data URI 前缀）
	Data string `json:"data"`
}

// geminiFileData 承载通过 URI 引用的远程文件，用于 http(s) 图片地址。
type geminiFileData struct {
	// MimeType 媒体类型，可为空（由 Gemini 服务端推断）
	MimeType string `json:"mimeType,omitempty"`
	// FileURI 文件的 http(s) 地址
	FileURI string `json:"fileUri"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// geminiFunctionResponse 承载工具调用结果，回传给模型。
// Gemini 没有独立的 "tool" 角色，工具结果以 functionResponse part
// 放入 role="user" 的 content 中回传。
type geminiFunctionResponse struct {
	// Name 关联的函数名
	Name string `json:"name"`
	// Response 工具返回内容，包装为对象（Gemini 要求 response 为结构化对象）
	Response map[string]any `json:"response"`
}

// Gemini API 响应结构
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
			Role  string       `json:"role"`
		} `json:"content"`
		FinishReason  string `json:"finishReason"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// parseResponse 解析响应
//
// Gemini 的 generateContent API 不返回 id 与 created 字段，因此本地生成
// 响应 ID 并以当前时间戳填充 Created，避免关键字段恒为零值（W3-55）。
func (p *Provider) parseResponse(resp *geminiResponse, model string) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:      "gemini-" + idgen.NanoID(),
		Model:   model,
		Created: time.Now().Unix(),
		Usage: llm.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		},
	}

	if len(resp.Candidates) > 0 {
		candidate := resp.Candidates[0]
		result.FinishReason = candidate.FinishReason

		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				result.Content += part.Text
			}
			if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
					ID:        fmt.Sprintf("call_%s", part.FunctionCall.Name),
					Type:      "function",
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				})
			}
		}
	}

	return result
}

// Embed 生成文本的向量嵌入
func (p *Provider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return p.EmbedWithModel(ctx, "text-embedding-004", texts)
}

// EmbedWithModel 使用指定模型生成嵌入
func (p *Provider) EmbedWithModel(ctx context.Context, model string, texts []string) ([][]float32, error) {
	// 批量嵌入
	requests := make([]map[string]any, len(texts))
	for i, text := range texts {
		requests[i] = map[string]any{
			"model": fmt.Sprintf("models/%s", model),
			"content": map[string]any{
				"parts": []map[string]string{{"text": text}},
			},
		}
	}

	payload := map[string]any{
		"requests": requests,
	}
	body, _ := json.Marshal(payload)

	resp, err := p.doRequest(ctx, "embed", fmt.Sprintf("%s/models/%s:batchEmbedContents", p.baseURL, model), body, false)
	if err != nil {
		return nil, fmt.Errorf("gemini embed error: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Embeddings []struct {
			Values []float32 `json:"values"`
		} `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	embeddings := make([][]float32, len(result.Embeddings))
	for i, e := range result.Embeddings {
		embeddings[i] = e.Values
	}

	return embeddings, nil
}

func (p *Provider) doRequest(ctx context.Context, action, url string, body []byte, stream bool) (*http.Response, error) {
	cfg := transport.Request{
		Provider:  p.Name(),
		Action:    action,
		Method:    http.MethodPost,
		URL:       url,
		Body:      body,
		Client:    p.httpClient,
		Transport: p.transport,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("x-goog-api-key", p.apiKey)
		},
	}
	if stream {
		cfg.StreamIdle = 5 * time.Minute
	}
	return transport.Do(ctx, cfg)
}

// 确保实现了 Provider 和 EmbeddingProvider 接口
var _ llm.Provider = (*Provider)(nil)
var _ llm.EmbeddingProvider = (*Provider)(nil)
