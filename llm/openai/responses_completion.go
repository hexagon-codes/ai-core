package openai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/transport"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

// ResponsesContinuationProtocol 标识本适配器产生的 Responses 续传内容。
const ResponsesContinuationProtocol = "openai_responses"

type responsesProtocolKey struct{}

type responsesProtocolBinding struct {
	enabled bool
	scope   string
}

// WithResponsesProtocol 显式选择普通补全的 Responses 协议。
// scope 由调用方绑定实际配置和任务身份，不会发送给上游；空 scope 保留已有非空绑定。
// 本函数不证明模型或兼容端点具备任何能力。
func WithResponsesProtocol(ctx context.Context, scope string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if scope == "" {
		if previous, ok := ctx.Value(responsesProtocolKey{}).(responsesProtocolBinding); ok {
			scope = previous.scope
		}
	}
	return context.WithValue(ctx, responsesProtocolKey{}, responsesProtocolBinding{enabled: true, scope: scope})
}

// WithChatCompletionsProtocol 显式选择原有聊天协议，覆盖继承的 Responses 选择和模型选项。
func WithChatCompletionsProtocol(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, responsesProtocolKey{}, responsesProtocolBinding{})
}

// WithResponsesModels 为明确支持 Responses 的实际通道启用指定的精确模型 ID。
// 未命中的模型沿用原协议；调用方仍负责能力发现，本选项不认证官方服务或同名兼容模型。
func WithResponsesModels(models ...string) Option {
	return func(p *Provider) {
		p.responsesModels = make(map[string]struct{}, len(models))
		for _, model := range models {
			if model != "" {
				p.responsesModels[model] = struct{}{}
			}
		}
	}
}

func (p *Provider) useResponsesProtocol(ctx context.Context, model string) bool {
	if ctx != nil {
		if binding, ok := ctx.Value(responsesProtocolKey{}).(responsesProtocolBinding); ok {
			return binding.enabled
		}
	}
	_, enabled := p.responsesModels[model]
	return enabled
}

// ResponsesContinuationSource 返回实际端点、凭据身份、精确模型和调用作用域的不可逆指纹。
// 调用方可以比较或显式持久化该不透明值，不应复制其计算规则或记录原始绑定输入。
func (p *Provider) ResponsesContinuationSource(ctx context.Context, model string) string {
	if p == nil {
		return ""
	}
	scope := ""
	if ctx != nil {
		if binding, ok := ctx.Value(responsesProtocolKey{}).(responsesProtocolBinding); ok {
			scope = binding.scope
		}
	}
	endpoint := strings.TrimRight(p.baseURL, "/") + "/responses"
	if parsed, err := url.Parse(p.baseURL); err == nil {
		parsed.Scheme = strings.ToLower(parsed.Scheme)
		parsed.Host = strings.ToLower(parsed.Host)
		parsed.Path = strings.TrimRight(parsed.Path, "/") + "/responses"
		parsed.RawPath = ""
		parsed.Fragment = ""
		parsed.RawQuery = parsed.Query().Encode()
		endpoint = parsed.String()
	}
	credential := sha256.Sum256([]byte(p.apiKey))
	identity, _ := json.Marshal([]string{ResponsesContinuationProtocol, endpoint, hex.EncodeToString(credential[:]), model, scope})
	digest := sha256.Sum256(identity)
	return hex.EncodeToString(digest[:])
}

type responseFunctionCall struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status"`
}

func (p *Provider) buildCompletionResponsesBody(ctx context.Context, req llm.CompletionRequest, stream bool) ([]byte, llm.ReasoningPlan, bool, error) {
	plan, explicit, err := llm.PlanReasoningFromMetadata(req.Metadata, req.Model, p.baseURL)
	if err != nil {
		return nil, plan, explicit, err
	}
	if len(req.Stop) > 0 {
		return nil, plan, explicit, errors.New("responses does not support stop sequences")
	}
	input, err := completionResponsesInput(req.Messages, p.ResponsesContinuationSource(ctx, req.Model), req.Model)
	if err != nil {
		return nil, plan, explicit, err
	}
	payload := map[string]any{
		"model": req.Model, "input": input, "store": false, "stream": stream,
		"include": []string{"reasoning.encrypted_content"},
	}
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, definition := range req.Tools {
			if definition.Type != "function" {
				return nil, plan, explicit, errors.New("ordinary Responses completion requires function tools")
			}
			// 现有工具 Schema 沿用聊天协议的非 strict 语义，不隐式收紧可选参数。
			tool := map[string]any{"type": "function", "name": definition.Function.Name,
				"description": definition.Function.Description, "strict": false}
			if definition.Function.Parameters != nil {
				tool["parameters"] = definition.Function.Parameters
			}
			tools = append(tools, tool)
		}
		payload["tools"] = tools
	}
	if req.ToolChoice != nil {
		choice, err := completionResponsesToolChoice(req.ToolChoice)
		if err != nil {
			return nil, plan, explicit, err
		}
		payload["tool_choice"] = choice
	}
	if req.MaxTokens > 0 {
		payload["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		payload["top_p"] = *req.TopP
	}
	if req.User != "" {
		payload["user"] = req.User
	}
	if explicit {
		if effort, exists := plan.Wire[string(llm.ReasoningDialectEffort)]; exists {
			payload["reasoning"] = map[string]any{"effort": effort}
		} else if len(plan.Wire) > 0 {
			return nil, plan, explicit, errors.New("responses requires reasoning effort control")
		}
	} else if effort, ok := trustedOpenAIReasoningEffort(req); ok {
		payload["reasoning"] = map[string]any{"effort": effort}
	}
	if format := req.ResponseFormat; format != nil {
		wire := map[string]any{"type": format.Type}
		switch format.Type {
		case "text", "json_object":
		case "json_schema":
			if format.JSONSchema == nil {
				return nil, plan, explicit, errors.New("responses JSON schema format requires a schema")
			}
			wire["name"], wire["schema"], wire["strict"] = format.JSONSchema.Name, format.JSONSchema.Schema, format.JSONSchema.Strict
			if format.JSONSchema.Description != "" {
				wire["description"] = format.JSONSchema.Description
			}
		default:
			return nil, plan, explicit, errors.New("unsupported Responses text format")
		}
		payload["text"] = map[string]any{"format": wire}
	}
	body, err := json.Marshal(payload)
	return body, plan, explicit, err
}

func completionResponsesToolChoice(choice any) (any, error) {
	if value, ok := choice.(string); ok {
		switch value {
		case "auto", "none", "required":
			return value, nil
		default:
			if value != "" {
				return map[string]any{"type": "function", "name": value}, nil
			}
		}
	}
	raw, err := json.Marshal(choice)
	if err != nil {
		return nil, err
	}
	var value struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, err
	}
	if value.Name == "" {
		value.Name = value.Function.Name
	}
	if value.Type != "function" || value.Name == "" {
		return nil, errors.New("unsupported Responses tool choice")
	}
	return map[string]any{"type": "function", "name": value.Name}, nil
}

func completionResponsesInput(messages []llm.Message, source, model string) ([]json.RawMessage, error) {
	input := make([]json.RawMessage, 0, len(messages))
	pending := make(map[string]bool)
	seenCalls := make(map[string]bool)
	appendItem := func(value any) error {
		raw, err := json.Marshal(value)
		if err == nil {
			input = append(input, raw)
		}
		return err
	}
	for _, message := range messages {
		if message.Role == llm.RoleAssistant && message.ProviderContinuation != nil {
			state := message.ProviderContinuation
			if state.Protocol == ResponsesContinuationProtocol && state.Source == source && state.Model == model && len(state.Output) > 0 {
				for _, raw := range state.Output {
					if !json.Valid(raw) {
						return nil, errors.New("invalid Responses continuation item")
					}
					var call responseFunctionCall
					if err := json.Unmarshal(raw, &call); err != nil {
						return nil, err
					}
					if call.Type == "function_call" {
						if call.CallID == "" || seenCalls[call.CallID] {
							return nil, errors.New("responses continuation has a missing or duplicate function call_id")
						}
						seenCalls[call.CallID] = true
						pending[call.CallID] = true
					}
					input = append(input, append(json.RawMessage(nil), raw...))
				}
				continue
			}
		}
		if message.Role == llm.RoleTool && pending[message.ToolCallID] {
			var output any = message.Content
			if len(message.MultiContent) > 0 {
				output = completionResponsesContent(message, "input_text")
			}
			if err := appendItem(map[string]any{"type": "function_call_output", "call_id": message.ToolCallID, "output": output}); err != nil {
				return nil, err
			}
			delete(pending, message.ToolCallID)
			continue
		}
		role := string(message.Role)
		content := completionResponsesContent(message, "input_text")
		if message.Role == llm.RoleTool {
			// 不匹配的来源只作为已执行事实输入，不伪造旧推理链的 function_call_output。
			role = "user"
			label := fmt.Sprintf("Previously executed tool result (call_id=%s, tool=%s); use this observation without replaying the tool.", message.ToolCallID, message.Name)
			content = append([]map[string]any{{"type": "input_text", "text": label}}, content...)
		} else if message.Role == llm.RoleAssistant {
			if len(message.ToolCalls) > 0 {
				// 缺少合法续传时保留调用身份与正文，不把历史动作重新声明成待执行调用。
				refs, err := json.Marshal(message.ToolCalls)
				if err != nil {
					return nil, err
				}
				role = "user"
				label := "Previously recorded tool calls; their results follow as observations. Do not replay these calls: " + string(refs)
				content = append([]map[string]any{{"type": "input_text", "text": label}}, content...)
			} else {
				content = completionResponsesContent(message, "output_text")
			}
		}
		if len(content) == 0 {
			continue
		}
		if err := appendItem(map[string]any{"type": "message", "role": role, "content": content}); err != nil {
			return nil, err
		}
	}
	if len(pending) > 0 {
		return nil, errors.New("responses continuation is missing a function result")
	}
	return input, nil
}

func completionResponsesContent(message llm.Message, textType string) []map[string]any {
	if len(message.MultiContent) == 0 {
		if message.Content == "" {
			return nil
		}
		return []map[string]any{{"type": textType, "text": message.Content}}
	}
	content := make([]map[string]any, 0, len(message.MultiContent))
	for _, part := range message.MultiContent {
		if part.Type == "image_url" && part.ImageURL != nil {
			image := map[string]any{"type": "input_image", "image_url": part.ImageURL.URL}
			if part.ImageURL.Detail != "" {
				image["detail"] = part.ImageURL.Detail
			}
			content = append(content, image)
		} else if part.Text != "" {
			content = append(content, map[string]any{"type": textType, "text": part.Text})
		}
	}
	return content
}

type completionResponsesClient struct {
	provider   *Provider
	action     string
	dispatched atomic.Bool
	statusCode int
	requestID  string
}

type responsesDispatchTransport struct {
	base http.RoundTripper
	call *completionResponsesClient
}

func (t responsesDispatchTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.call.dispatched.Store(true)
	return t.base.RoundTrip(req)
}

// Do 让官方 SDK 复用普通补全的实际出口及派发钩子，两个重试层均只允许一次请求。
func (c *completionResponsesClient) Do(req *http.Request) (*http.Response, error) {
	p := c.provider
	if err := transport.ValidateURL(req.Context(), req.URL.String(), p.networkPolicy); err != nil {
		return nil, err
	}
	body, readErr := io.ReadAll(req.Body)
	closeErr := req.Body.Close()
	if readErr != nil {
		return nil, readErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	client, err := p.transport.Client()
	if err != nil {
		return nil, err
	}
	onceClient := *client
	onceClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	base := onceClient.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	onceClient.Transport = responsesDispatchTransport{base: base, call: c}
	resp, err := transport.Do(req.Context(), transport.Request{
		Provider: p.Name(), Action: c.action, Method: req.Method, URL: req.URL.String(), Body: body,
		Client: &onceClient, Timeout: p.requestTimeout, StreamIdle: p.streamIdle,
		Headers: p.headers, Retry: transport.RetryPolicy{MaxAttempts: 1},
		SetHeaders: func(out *http.Request) { out.Header = req.Header.Clone() },
	})
	if resp != nil {
		c.statusCode, c.requestID = responseReceipt(resp)
	}
	var providerErr *transport.ProviderError
	if errors.As(err, &providerErr) {
		c.statusCode, c.requestID = providerErr.StatusCode, providerErr.RequestID
	}
	return resp, err
}

func (p *Provider) completionResponsesService(action string) (responses.ResponseService, *completionResponsesClient) {
	call := &completionResponsesClient{provider: p, action: action}
	return responses.NewResponseService(option.WithAPIKey(p.apiKey), option.WithBaseURL(p.baseURL),
		option.WithHTTPClient(call), option.WithMaxRetries(0)), call
}

func (c *completionResponsesClient) failure(code string, cause error) *ResponsesError {
	var apiError *openaisdk.Error
	if errors.As(cause, &apiError) && apiError.Code != "" {
		code = apiError.Code
	}
	dispatched := c.dispatched.Load()
	return &ResponsesError{Code: code, RequestID: c.requestID, StatusCode: c.statusCode, Dispatched: dispatched,
		OutcomeUnknown: dispatched && (c.statusCode == 0 || c.statusCode < 400 || c.statusCode == http.StatusRequestTimeout || c.statusCode >= 500), cause: cause}
}

func (p *Provider) completeResponses(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, &ResponsesError{Code: "cancelled_before_dispatch", cause: err}
	}
	body, plan, explicit, err := p.buildCompletionResponsesBody(ctx, req, false)
	if err != nil {
		if explicit {
			llm.PublishReasoningReceipt(req.Metadata, plan.Receipt)
		}
		return nil, &ResponsesError{Code: "invalid_input", cause: err}
	}
	service, call := p.completionResponsesService("complete")
	result, err := service.New(ctx, responses.ResponseNewParams{}, option.WithRequestBody("application/json", body))
	if err != nil {
		return nil, call.failure("request_failed", err)
	}
	if result == nil {
		return nil, call.failure("missing_response", errors.New("responses returned no response"))
	}
	parsed, err := parseResponsesResponse([]byte(result.RawJSON()))
	if err != nil {
		var partial *llm.CompletionResponse
		if parsed != nil {
			partial = &llm.CompletionResponse{ID: parsed.ID, RequestID: call.requestID, Model: parsed.Model}
			if parsed.Usage != nil {
				usage := *parsed.Usage
				partial.Usage, partial.UsageReceipt = usage, &usage
			}
		}
		return partial, call.failure("invalid_response", err)
	}
	parsed.RequestID = call.requestID
	completion, err := p.completionFromResponse(ctx, req, parsed)
	if explicit && completion != nil {
		receipt := plan.Receipt.WithEvidence(true, false, responsesReasoningTokens(parsed.RawUsage))
		completion.ReasoningReceipt = &receipt
		llm.PublishReasoningReceipt(req.Metadata, receipt)
	}
	return completion, err
}

func (p *Provider) completionFromResponse(ctx context.Context, req llm.CompletionRequest, response *ResponsesResponse) (*llm.CompletionResponse, error) {
	completion := &llm.CompletionResponse{ID: response.ID, RequestID: response.RequestID, Model: response.Model, FinishReason: responsesCompletionFinishReason(response)}
	var envelope struct {
		CreatedAt float64 `json:"created_at"`
	}
	if json.Unmarshal(response.RawJSON, &envelope) == nil {
		completion.Created = int64(envelope.CreatedAt)
	}
	if response.Usage != nil {
		usage := *response.Usage
		completion.Usage, completion.UsageReceipt = usage, &usage
	}
	var contentText strings.Builder
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Type == "output_text" {
				contentText.WriteString(content.Text)
			} else if content.Type == "refusal" {
				contentText.WriteString(content.Refusal)
			}
		}
	}
	completion.Content = contentText.String()
	if response.Status != "completed" {
		code := "response_" + response.Status
		if response.Failure != nil && response.Failure.Code != "" {
			code = response.Failure.Code
		}
		return completion, &ResponsesError{Code: code, RequestID: response.RequestID, Dispatched: true,
			OutcomeUnknown: response.Status != "failed" && response.Status != "incomplete" && response.Status != "cancelled"}
	}
	state := &llm.ProviderContinuation{Protocol: ResponsesContinuationProtocol,
		Source: p.ResponsesContinuationSource(ctx, req.Model), Model: req.Model, Output: make([]json.RawMessage, 0, len(response.Output))}
	callIDs := make(map[string]bool)
	for _, item := range response.Output {
		if item.Type == "function_call" {
			var call responseFunctionCall
			if err := json.Unmarshal(item.RawJSON, &call); err != nil || call.CallID == "" || call.Name == "" || !json.Valid([]byte(call.Arguments)) || call.Status != "" && call.Status != "completed" {
				completion.ToolCalls = nil
				return completion, &ResponsesError{Code: "invalid_function_call", RequestID: response.RequestID, Dispatched: true, cause: err}
			}
			if callIDs[call.CallID] {
				completion.ToolCalls = nil
				return completion, &ResponsesError{Code: "duplicate_function_call", RequestID: response.RequestID, Dispatched: true}
			}
			callIDs[call.CallID] = true
			completion.ToolCalls = append(completion.ToolCalls, llm.ToolCall{Index: len(completion.ToolCalls), ID: call.CallID, Type: "function", Name: call.Name, Arguments: call.Arguments})
		}
		state.Output = append(state.Output, append(json.RawMessage(nil), item.RawJSON...))
	}
	if len(completion.ToolCalls) > 0 && completion.FinishReason == "stop" {
		completion.FinishReason = "tool_calls"
	}
	completion.ProviderContinuation = state
	return completion, nil
}

// responsesCompletionFinishReason 保留模型终态事实，不由 HTTP 成功或正文非空推断完成。
func responsesCompletionFinishReason(response *ResponsesResponse) string {
	var details struct {
		Reason string `json:"reason"`
	}
	if json.Unmarshal(response.IncompleteDetails, &details) == nil && details.Reason != "" {
		switch details.Reason {
		case "max_output_tokens":
			return "length"
		case "content_filter", "refusal":
			return details.Reason
		default:
			return details.Reason
		}
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Type == "refusal" {
				return "refusal"
			}
		}
	}
	if response.Failure != nil && (response.Failure.Code == "content_filter" || response.Failure.Code == "refusal") {
		return response.Failure.Code
	}
	switch response.Status {
	case "completed":
		return "stop"
	case "":
		return "unknown"
	default:
		return response.Status
	}
}

func responsesReasoningTokens(raw json.RawMessage) int {
	var usage struct {
		OutputTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	}
	if json.Unmarshal(raw, &usage) != nil {
		return 0
	}
	return usage.OutputTokensDetails.ReasoningTokens
}
