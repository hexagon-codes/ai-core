package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/transport"
	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

// ResponsesProvider 是可选的 Responses 能力，不改变 llm.Provider 合同。
// 实现此接口只表示能够传输协议；具体模型是否支持 computer 仍由实际配置与响应决定。
type ResponsesProvider interface {
	CreateResponse(context.Context, ResponsesRequest) (*ResponsesResponse, error)
}

// ResponsesRequest 描述一次模型交互。调用方负责轮次身份、预算和重试。
// Model 必须明确提供；本接口不把聊天默认模型推断为具有 Computer Use 能力。
type ResponsesRequest struct {
	Model              string              `json:"model"`
	Input              []ResponseInputItem `json:"input"`
	Tools              []ComputerTool      `json:"tools,omitempty"`
	PreviousResponseID string              `json:"previous_response_id,omitempty"`
	Instructions       string              `json:"instructions,omitempty"`
	MaxOutputTokens    int                 `json:"max_output_tokens,omitempty"`
	Store              *bool               `json:"store,omitempty"`
	Include            []string            `json:"include,omitempty"`
	// BeforeSend 在本地构造与网络策略检查后、唯一物理请求派发前调用。
	// 返回错误阻止派发，适用于调用方持久保存执行身份和派发回执。
	BeforeSend func(context.Context) error `json:"-"`
}

// ComputerTool 声明实际使用的 Computer 协议。当前协议使用 Type="computer"；
// 旧协议的显示尺寸和环境只在显式选择 computer_use_preview 时提供。
type ComputerTool struct {
	Type          string `json:"type"`
	DisplayWidth  int    `json:"display_width,omitempty"`
	DisplayHeight int    `json:"display_height,omitempty"`
	Environment   string `json:"environment,omitempty"`
}

// ResponseInputItem 支持消息、Computer 操作结果及按原协议回传的上下文条目。
// RawJSON 仅用于无损接续已获得的协议条目；非空时优先于其余字段。
type ResponseInputItem struct {
	Type                     string                 `json:"type,omitempty"`
	Role                     string                 `json:"role,omitempty"`
	Content                  []ResponseInputContent `json:"content,omitempty"`
	CallID                   string                 `json:"call_id,omitempty"`
	Output                   *ComputerScreenshot    `json:"output,omitempty"`
	AcknowledgedSafetyChecks []ComputerSafetyCheck  `json:"acknowledged_safety_checks,omitempty"`
	RawJSON                  json.RawMessage        `json:"-"`
}

// MarshalJSON 保留接续条目的原始协议字段，避免丢失 reasoning 等上下文。
func (i ResponseInputItem) MarshalJSON() ([]byte, error) {
	if len(i.RawJSON) > 0 {
		if !json.Valid(i.RawJSON) {
			return nil, errors.New("invalid Responses input JSON")
		}
		return i.RawJSON, nil
	}
	type inputItem ResponseInputItem
	return json.Marshal(inputItem(i))
}

// ResponseInputContent 是文本或图像消息内容；截图使用原始分辨率的数据 URL。
type ResponseInputContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// ComputerScreenshot 是对已执行 computer_call 的真实图像观察。
type ComputerScreenshot struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url"`
	Detail   string `json:"detail,omitempty"`
}

// ComputerSafetyCheck 保留上游检查身份；本传输层不替调用方确认或执行检查。
type ComputerSafetyCheck struct {
	ID      string `json:"id"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// ComputerCoordinate 是截图坐标系中的一点。
type ComputerCoordinate struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// UnmarshalJSON 要求拖动路径的每个点均有真实坐标。
func (p *ComputerCoordinate) UnmarshalJSON(raw []byte) error {
	var point struct {
		X *int `json:"x"`
		Y *int `json:"y"`
	}
	if err := json.Unmarshal(raw, &point); err != nil {
		return err
	}
	if point.X == nil || point.Y == nil {
		return errors.New("computer coordinate is missing x or y")
	}
	p.X, p.Y = *point.X, *point.Y
	return nil
}

// ComputerAction 是模型生成的有序操作意图，不代表页面动作已经执行。
// RawJSON 保留未来协议字段；不认识的 Type 交由实际执行器报告能力缺口。
type ComputerAction struct {
	Type    string               `json:"type"`
	X       int                  `json:"x,omitempty"`
	Y       int                  `json:"y,omitempty"`
	Button  string               `json:"button,omitempty"`
	Text    string               `json:"text,omitempty"`
	Keys    []string             `json:"keys,omitempty"`
	Path    []ComputerCoordinate `json:"path,omitempty"`
	ScrollX int                  `json:"scroll_x,omitempty"`
	ScrollY int                  `json:"scroll_y,omitempty"`
	RawJSON json.RawMessage      `json:"-"`
}

// UnmarshalJSON 保留动作原文，并区分缺失坐标与真实的零坐标。
func (a *ComputerAction) UnmarshalJSON(raw []byte) error {
	type action ComputerAction
	var parsed action
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	required := []string{"type"}
	switch parsed.Type {
	case "click", "double_click", "move":
		required = append(required, "x", "y")
	case "scroll":
		required = append(required, "x", "y", "scroll_x", "scroll_y")
	case "type":
		required = append(required, "text")
	case "keypress":
		required = append(required, "keys")
	case "drag":
		required = append(required, "path")
	}
	for _, name := range required {
		if value, ok := fields[name]; !ok || string(value) == "null" {
			return fmt.Errorf("computer action is missing %s", name)
		}
	}
	if parsed.Type == "" {
		return errors.New("computer action is missing its type")
	}
	*a = ComputerAction(parsed)
	a.RawJSON = append(json.RawMessage(nil), raw...)
	return nil
}

// ResponseOutputContent 保留文字、拒绝结果等消息内容及其协议类型。
type ResponseOutputContent struct {
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

// ResponseOutputItem 保留响应条目身份、computer 批次与原始协议内容。
// Action 只承载旧协议单动作；当前协议的 Actions 顺序不能被合并或重排。
type ResponseOutputItem struct {
	ID                  string                  `json:"id,omitempty"`
	Type                string                  `json:"type"`
	Status              string                  `json:"status,omitempty"`
	Role                string                  `json:"role,omitempty"`
	CallID              string                  `json:"call_id,omitempty"`
	Content             []ResponseOutputContent `json:"content,omitempty"`
	Actions             []ComputerAction        `json:"actions,omitempty"`
	Action              *ComputerAction         `json:"action,omitempty"`
	PendingSafetyChecks []ComputerSafetyCheck   `json:"pending_safety_checks,omitempty"`
	RawJSON             json.RawMessage         `json:"-"`
}

// ResponseFailure 保留上游已确认的模型失败原因，不把它解释为页面执行结果。
// Message 属于未脱敏上游内容，不应直接写入普通日志。
type ResponseFailure struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ResponsesResponse 包含一次模型交互的实际回执。Usage 为 nil 表示上游未提供，
// 不能解释为零消耗；RawJSON 和 RawUsage 用于调用方保存协议证据，不应写入普通日志。
type ResponsesResponse struct {
	ID                string               `json:"id"`
	Model             string               `json:"model"`
	Status            string               `json:"status"`
	Output            []ResponseOutputItem `json:"output"`
	RequestID         string               `json:"request_id,omitempty"`
	Usage             *llm.Usage           `json:"usage,omitempty"`
	RawUsage          json.RawMessage      `json:"-"`
	RawJSON           json.RawMessage      `json:"-"`
	IncompleteDetails json.RawMessage      `json:"incomplete_details,omitempty"`
	Failure           *ResponseFailure     `json:"error,omitempty"`
}

// OutputText 只汇总真实 output_text，不把工具动作或拒绝内容伪装为最终答案。
func (r *ResponsesResponse) OutputText() string {
	if r == nil {
		return ""
	}
	var text strings.Builder
	for _, item := range r.Output {
		for _, content := range item.Content {
			if content.Type == "output_text" {
				text.WriteString(content.Text)
			}
		}
	}
	return text.String()
}

// ResponsesError 区分本地未派发、明确拒绝和结果未知。
// Error 不包含原始请求、截图、认证头或上游错误正文；响应身份可用于后续查证。
type ResponsesError struct {
	StatusCode     int
	RequestID      string
	Code           string
	Dispatched     bool
	OutcomeUnknown bool
	cause          error
}

// UnsafeToReplay 将派发结果未知的事实交给通用重试边界，避免自动重做付费或外部动作。
func (e *ResponsesError) UnsafeToReplay() bool { return e != nil && e.OutcomeUnknown }

// HTTPStatusCode 让公共分类器读取状态事实，不暴露上游正文或改变派发结果语义。
func (e *ResponsesError) HTTPStatusCode() int {
	if e == nil {
		return 0
	}
	return e.StatusCode
}

func (e *ResponsesError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if !e.Dispatched {
		return "openai Responses request was not dispatched"
	}
	if e.OutcomeUnknown {
		return "openai Responses outcome is unknown"
	}
	if e.StatusCode >= 400 {
		return fmt.Sprintf("openai Responses request failed (HTTP %d)", e.StatusCode)
	}
	return "openai Responses model response failed"
}

// Unwrap 保留取消和底层错误的程序化识别能力；调用方不应记录未脱敏的 SDK 错误。
func (e *ResponsesError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// CreateResponse 通过官方 SDK 执行唯一一次 Responses 请求。
// SDK 重试与 HTTP 重定向均关闭；后续轮次和未知结果恢复由调用方决定。
// BeforeSend 成功后发生的连接、读取或协议错误保留为结果未知，不能盲重发。
// 返回的响应仅描述模型输出，不表示其中的页面动作已执行。
func (p *Provider) CreateResponse(ctx context.Context, req ResponsesRequest) (*ResponsesResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if p == nil || strings.TrimSpace(req.Model) == "" {
		return nil, &ResponsesError{Code: "missing_model", cause: errors.New("Responses requires an explicit model")}
	}
	if err := ctx.Err(); err != nil {
		return nil, &ResponsesError{Code: "cancelled_before_dispatch", cause: err}
	}
	if p.requestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.requestTimeout)
		defer cancel()
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, &ResponsesError{Code: "invalid_input", cause: err}
	}
	client, err := p.transport.Client()
	if err != nil {
		return nil, &ResponsesError{Code: "invalid_transport", cause: err}
	}
	onceClient := *client
	onceClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	var dispatched bool
	var rawResponse *http.Response
	service := responses.NewResponseService(
		option.WithAPIKey(p.apiKey),
		option.WithBaseURL(p.baseURL),
		option.WithHTTPClient(&onceClient),
		option.WithMaxRetries(0),
		option.WithMiddleware(func(httpReq *http.Request, next option.MiddlewareNext) (*http.Response, error) {
			if err := transport.ValidateURL(httpReq.Context(), httpReq.URL.String(), p.networkPolicy); err != nil {
				return nil, err
			}
			transport.ApplySafeHeaders(httpReq, p.headers)
			if err := httpReq.Context().Err(); err != nil {
				return nil, err
			}
			if req.BeforeSend != nil {
				if err := req.BeforeSend(httpReq.Context()); err != nil {
					return nil, err
				}
			}
			dispatched = true
			return next(httpReq)
		}),
	)
	result, err := service.New(ctx, responses.ResponseNewParams{},
		option.WithRequestBody("application/json", body), option.WithResponseInto(&rawResponse))
	statusCode, requestID := responseReceipt(rawResponse)
	if err != nil {
		code := "request_failed"
		var apiErr *openaisdk.Error
		if errors.As(err, &apiErr) {
			code = apiErr.Code
		}
		return nil, &ResponsesError{
			StatusCode: statusCode, RequestID: requestID, Code: code, Dispatched: dispatched,
			OutcomeUnknown: dispatched && (statusCode == 0 || statusCode < 400 || statusCode == http.StatusRequestTimeout || statusCode >= 500),
			cause:          err,
		}
	}
	if result == nil {
		return nil, &ResponsesError{StatusCode: statusCode, RequestID: requestID, Code: "missing_response", Dispatched: dispatched, OutcomeUnknown: dispatched}
	}
	parsed, err := parseResponsesResponse([]byte(result.RawJSON()))
	if err != nil {
		if parsed != nil {
			parsed.RequestID = requestID
		}
		return parsed, &ResponsesError{StatusCode: statusCode, RequestID: requestID, Code: "invalid_response", Dispatched: dispatched, OutcomeUnknown: dispatched, cause: err}
	}
	parsed.RequestID = requestID
	if parsed.Status != "completed" {
		code := "response_" + parsed.Status
		if parsed.Failure != nil && parsed.Failure.Code != "" {
			code = parsed.Failure.Code
		}
		return parsed, &ResponsesError{
			StatusCode: statusCode, RequestID: requestID, Code: code, Dispatched: dispatched,
			OutcomeUnknown: parsed.Status != "failed" && parsed.Status != "incomplete" && parsed.Status != "cancelled",
		}
	}
	return parsed, nil
}

func responseReceipt(resp *http.Response) (int, string) {
	if resp == nil {
		return 0, ""
	}
	return resp.StatusCode, resp.Header.Get("x-request-id")
}

func parseResponsesResponse(raw []byte) (*ResponsesResponse, error) {
	var envelope struct {
		ID                string            `json:"id"`
		Model             string            `json:"model"`
		Status            string            `json:"status"`
		Output            []json.RawMessage `json:"output"`
		Usage             json.RawMessage   `json:"usage"`
		IncompleteDetails json.RawMessage   `json:"incomplete_details"`
		Failure           *ResponseFailure  `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, err
	}
	parsed := &ResponsesResponse{
		ID: envelope.ID, Model: envelope.Model, Status: envelope.Status,
		RawJSON: append(json.RawMessage(nil), raw...), RawUsage: envelope.Usage,
		IncompleteDetails: envelope.IncompleteDetails,
		Failure:           envelope.Failure,
		Output:            make([]ResponseOutputItem, 0, len(envelope.Output)),
	}
	if envelope.ID == "" || envelope.Model == "" || envelope.Status == "" || envelope.Output == nil {
		return parsed, errors.New("Responses response is missing required fields")
	}
	var err error
	parsed.Usage, err = parseResponseUsage(envelope.Usage)
	if err != nil {
		return parsed, err
	}
	for _, rawItem := range envelope.Output {
		var item ResponseOutputItem
		if err := json.Unmarshal(rawItem, &item); err != nil {
			return parsed, err
		}
		item.RawJSON = rawItem
		if item.Type == "" {
			return parsed, errors.New("Responses output item is missing its type")
		}
		if item.Type == "computer_call" && (item.CallID == "" || len(item.Actions) == 0 && item.Action == nil) {
			return parsed, errors.New("computer call is missing its identity or actions")
		}
		if item.Type == "computer_call" && item.Status != "completed" {
			return parsed, errors.New("computer call is not complete")
		}
		parsed.Output = append(parsed.Output, item)
	}
	return parsed, nil
}

func parseResponseUsage(raw json.RawMessage) (*llm.Usage, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var usage struct {
		InputTokens        *int `json:"input_tokens"`
		OutputTokens       *int `json:"output_tokens"`
		TotalTokens        *int `json:"total_tokens"`
		InputTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
	}
	if err := json.Unmarshal(raw, &usage); err != nil {
		return nil, err
	}
	if usage.InputTokens == nil || usage.OutputTokens == nil || usage.TotalTokens == nil {
		return nil, nil
	}
	return &llm.Usage{PromptTokens: *usage.InputTokens, CompletionTokens: *usage.OutputTokens,
		TotalTokens: *usage.TotalTokens, CacheReadTokens: usage.InputTokensDetails.CachedTokens}, nil
}

var _ ResponsesProvider = (*Provider)(nil)
