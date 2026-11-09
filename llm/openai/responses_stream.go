package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

type completionResponsesEventStream interface {
	Next() bool
	Current() responses.ResponseStreamEventUnion
	Err() error
	Close() error
}

// completionResponsesReader 只把 SDK 已解码的事件重新分帧给现有 streamx，
// 不构造伪造的 Chat Completions 响应，也不额外创建后台生产协程。
type completionResponsesReader struct {
	stream     completionResponsesEventStream
	call       *completionResponsesClient
	buffer     []byte
	pendingErr error
	terminal   bool
	closeOnce  sync.Once
	closeErr   error
}

func (r *completionResponsesReader) Close() error {
	r.closeOnce.Do(func() { r.closeErr = r.stream.Close() })
	return r.closeErr
}

func (r *completionResponsesReader) Read(dst []byte) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}
	if len(r.buffer) > 0 {
		n := copy(dst, r.buffer)
		r.buffer = r.buffer[n:]
		return n, nil
	}
	if r.pendingErr != nil {
		err := r.pendingErr
		r.pendingErr = nil
		return 0, err
	}
	if r.terminal {
		return 0, io.EOF
	}
	if !r.stream.Next() {
		err := r.stream.Err()
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		r.terminal = true
		_ = r.Close()
		return 0, r.call.failure("stream_interrupted", err)
	}
	raw := r.stream.Current().RawJSON()
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil || envelope.Type == "" {
		r.terminal = true
		_ = r.Close()
		return 0, r.call.failure("invalid_stream_event", err)
	}
	switch envelope.Type {
	case "response.completed":
		r.terminal = true
		_ = r.Close()
	case "response.failed", "response.incomplete", "response.cancelled", "error":
		r.terminal = true
		r.pendingErr = &ResponsesError{Code: envelope.Type, RequestID: r.call.requestID,
			StatusCode: r.call.statusCode, Dispatched: r.call.dispatched.Load(), OutcomeUnknown: envelope.Type == "error"}
		_ = r.Close()
	}
	r.buffer = append([]byte(raw), '\n')
	return r.Read(dst)
}

type responseFunctionFragments struct {
	itemID    string
	callID    string
	name      string
	arguments string
	done      bool
}

type completionResponsesParser struct {
	provider  *Provider
	ctx       context.Context
	request   llm.CompletionRequest
	call      *completionResponsesClient
	plan      llm.ReasoningPlan
	explicit  bool
	emitted   strings.Builder
	lastUsage *llm.Usage
	functions map[int]*responseFunctionFragments
	itemIndex map[string]int
	completed atomic.Bool
	failure   atomic.Pointer[ResponsesError]
}

func (p *completionResponsesParser) AllowInlineToolCallRecovery() bool { return false }

func (p *completionResponsesParser) NormalizeTerminalError(err error) error {
	if failure := p.failure.Load(); failure != nil {
		return failure
	}
	if p.completed.Load() {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil
		}
		return err
	}
	var responseErr *ResponsesError
	if errors.As(err, &responseErr) {
		return err
	}
	if err == nil {
		err = io.ErrUnexpectedEOF
	}
	return p.call.failure("stream_interrupted", err)
}

func (p *completionResponsesParser) Parse(raw []byte) (*streamx.Chunk, error) {
	var event struct {
		Type        string          `json:"type"`
		Delta       string          `json:"delta"`
		ItemID      string          `json:"item_id"`
		OutputIndex *int            `json:"output_index"`
		Arguments   string          `json:"arguments"`
		Item        json.RawMessage `json:"item"`
		Response    json.RawMessage `json:"response"`
	}
	if err := json.Unmarshal(raw, &event); err != nil {
		return nil, err
	}
	chunk := &streamx.Chunk{RequestID: p.call.requestID, Model: p.request.Model, Role: "assistant"}
	if len(event.Response) > 0 {
		var receipt struct {
			ID    string          `json:"id"`
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(event.Response, &receipt); err != nil {
			return nil, err
		}
		chunk.ID = receipt.ID
		usage, err := parseResponseUsage(receipt.Usage)
		if err != nil {
			return nil, err
		}
		if usage != nil {
			p.lastUsage, chunk.Usage = usage, usage
		}
	}
	switch event.Type {
	case "response.output_text.delta", "response.refusal.delta":
		chunk.Content = event.Delta
		p.emitted.WriteString(event.Delta)
	case "response.output_item.added", "response.output_item.done":
		var call responseFunctionCall
		if err := json.Unmarshal(event.Item, &call); err != nil {
			return nil, err
		}
		if call.Type == "function_call" {
			fragment := p.functionFragment(event.OutputIndex, call.ID)
			if fragment == nil {
				return nil, p.fail("invalid_function_call", errors.New("responses function event is missing its item identity"))
			}
			if fragment.callID != "" && call.CallID != "" && fragment.callID != call.CallID {
				return nil, p.fail("invalid_function_call", errors.New("responses changed a function call identity"))
			}
			fragment.itemID = call.ID
			if call.CallID != "" {
				fragment.callID = call.CallID
			}
			if call.Name != "" {
				fragment.name = call.Name
			}
			if event.Type == "response.output_item.done" {
				if fragment.done && fragment.arguments != call.Arguments {
					return nil, p.fail("invalid_function_call", errors.New("responses changed completed function arguments"))
				}
				fragment.arguments, fragment.done = call.Arguments, true
			} else if call.Arguments != "" {
				fragment.arguments = call.Arguments
			}
		}
	case "response.function_call_arguments.delta":
		fragment := p.functionFragment(event.OutputIndex, event.ItemID)
		if fragment == nil {
			return nil, p.fail("invalid_function_call", errors.New("responses function delta is missing its item identity"))
		}
		if fragment.done {
			return nil, p.fail("invalid_function_call", errors.New("responses emitted function arguments after completion"))
		}
		fragment.arguments += event.Delta
	case "response.function_call_arguments.done":
		fragment := p.functionFragment(event.OutputIndex, event.ItemID)
		if fragment == nil || fragment.done && fragment.arguments != event.Arguments {
			return nil, p.fail("invalid_function_call", errors.New("responses changed or omitted completed function arguments"))
		}
		fragment.arguments, fragment.done = event.Arguments, true
	case "response.completed":
		if failure := p.failure.Load(); failure != nil {
			return nil, failure
		}
		response, err := parseResponsesResponse(event.Response)
		if err != nil {
			return nil, p.fail("invalid_response", err)
		}
		response.RequestID = p.call.requestID
		if err := p.validateFunctions(response); err != nil {
			return nil, p.fail("invalid_function_call", err)
		}
		completion, err := p.provider.completionFromResponse(p.ctx, p.request, response)
		if err != nil {
			var failure *ResponsesError
			if errors.As(err, &failure) {
				p.failure.Store(failure)
			}
			return nil, err
		}
		chunk.ID, chunk.Model, chunk.FinishReason = completion.ID, completion.Model, completion.FinishReason
		chunk.ToolCalls = completion.ToolCalls
		chunk.ProviderContinuation = completion.ProviderContinuation
		if !strings.HasPrefix(completion.Content, p.emitted.String()) {
			return nil, p.fail("invalid_response", errors.New("responses final text differs from the streamed content"))
		}
		chunk.Content = completion.Content[p.emitted.Len():]
		if completion.UsageReceipt != nil {
			chunk.Usage = completion.UsageReceipt
		} else {
			chunk.Usage = p.lastUsage
		}
		if p.explicit {
			receipt := p.plan.Receipt.WithEvidence(true, false, responsesReasoningTokens(response.RawUsage))
			llm.PublishReasoningReceipt(p.request.Metadata, receipt)
			chunk.ReasoningEvidence = llm.ReasoningEvidenceMap(receipt)
		}
		p.completed.Store(true)
	case "response.failed", "response.incomplete", "response.cancelled", "error":
		chunk.FinishReason = event.Type
	}
	return chunk, nil
}

func (p *completionResponsesParser) fail(code string, cause error) *ResponsesError {
	failure := p.call.failure(code, cause)
	p.failure.Store(failure)
	return failure
}

func (p *completionResponsesParser) functionFragment(index *int, itemID string) *responseFunctionFragments {
	i := 0
	if index != nil {
		i = *index
	} else if existing, ok := p.itemIndex[itemID]; ok {
		i = existing
	} else {
		return nil
	}
	if itemID != "" {
		p.itemIndex[itemID] = i
	}
	fragment := p.functions[i]
	if fragment == nil {
		fragment = &responseFunctionFragments{itemID: itemID}
		p.functions[i] = fragment
	}
	return fragment
}

func (p *completionResponsesParser) validateFunctions(response *ResponsesResponse) error {
	for index, item := range response.Output {
		if item.Type != "function_call" {
			continue
		}
		var call responseFunctionCall
		if err := json.Unmarshal(item.RawJSON, &call); err != nil {
			return err
		}
		fragment := p.functions[index]
		if mapped, ok := p.itemIndex[call.ID]; ok {
			fragment = p.functions[mapped]
		}
		if fragment == nil {
			continue
		}
		if fragment.callID != "" && fragment.callID != call.CallID || fragment.name != "" && fragment.name != call.Name ||
			fragment.done && fragment.arguments != call.Arguments || !fragment.done && !strings.HasPrefix(call.Arguments, fragment.arguments) {
			return errors.New("responses final function call differs from its streamed identity or arguments")
		}
	}
	return nil
}

func (p *completionResponsesParser) IsDone(raw []byte) bool {
	var event struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(raw, &event) == nil && event.Type == "response.completed"
}

func (p *Provider) streamResponses(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, &ResponsesError{Code: "cancelled_before_dispatch", cause: err}
	}
	body, plan, explicit, err := p.buildCompletionResponsesBody(ctx, req, true)
	if err != nil {
		if explicit {
			llm.PublishReasoningReceipt(req.Metadata, plan.Receipt)
		}
		return nil, &ResponsesError{Code: "invalid_input", cause: err}
	}
	service, call := p.completionResponsesService("stream")
	sdkStream := service.NewStreaming(ctx, responses.ResponseNewParams{}, option.WithRequestBody("application/json", body))
	if err := sdkStream.Err(); err != nil {
		_ = sdkStream.Close()
		return nil, call.failure("request_failed", err)
	}
	reader := &completionResponsesReader{stream: sdkStream, call: call}
	parser := &completionResponsesParser{provider: p, ctx: ctx, request: req, call: call, plan: plan, explicit: explicit,
		functions: make(map[int]*responseFunctionFragments), itemIndex: make(map[string]int)}
	stream := streamx.NewStreamWithContext(ctx, reader, streamx.CustomFormat).SetParser(parser)
	stream.SetRequestID(call.requestID)
	if explicit {
		receipt := plan.Receipt.WithEvidence(true, false, 0)
		llm.PublishReasoningReceipt(req.Metadata, receipt)
	}
	return stream, nil
}
