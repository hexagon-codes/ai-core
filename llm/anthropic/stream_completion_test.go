package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

// 原生终帧缺失时保留部分结果及用量事实，但不能将工具块当作消息完成。
func TestNativeStreamRequiresMessageStop(t *testing.T) {
	frames := []string{
		`{"type":"message_start","message":{"id":"partial","role":"assistant","model":"fixture","content":[],"usage":{"input_tokens":2,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call","name":"write","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		`{"type":"content_block_stop","index":1}`,
	}
	for _, tc := range []struct {
		name         string
		tail         []string
		incomplete   bool
		outputTokens int
	}{
		{name: "missing terminal", incomplete: true, outputTokens: 1},
		{name: "reason without message stop", tail: []string{`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`}, incomplete: true, outputTokens: 3},
		{name: "complete", tail: []string{`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`, `{"type":"message_stop"}`}, outputTokens: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				for _, frame := range append(append([]string{}, frames...), tc.tail...) {
					fmt.Fprintf(w, "data: %s\n\n", frame)
				}
			}))
			defer server.Close()
			ledger := llm.NewAttemptLedger()
			var receiptErr error
			ctx := llm.WithAttemptObserver(t.Context(), llm.AttemptObserverFunc(func(_ context.Context, receipt llm.AttemptReceipt) {
				if err := ledger.Merge(receipt); err != nil {
					receiptErr = err
				}
			}))
			stream, err := New("local", WithBaseURL(server.URL), WithModel("fixture")).Stream(ctx, llm.CompletionRequest{})
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			for range stream.Chunks() {
			}
			if errors.Is(stream.Err(), io.ErrUnexpectedEOF) != tc.incomplete {
				t.Fatalf("terminal error=%v incomplete=%v", stream.Err(), tc.incomplete)
			}
			result := stream.Result()
			if result.Content != "partial" || len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "write" || result.ToolCalls[0].Arguments != "{}" {
				t.Fatalf("partial facts lost: %+v", result)
			}
			if !tc.incomplete && result.FinishReason != "tool_use" {
				t.Fatalf("normal finish reason changed: %q", result.FinishReason)
			}
			usage := ledger.Summary().Usage
			if receiptErr != nil || usage.PromptTokens != 2 || usage.CompletionTokens != tc.outputTokens {
				t.Fatalf("receipt facts lost: usage=%+v error=%v", usage, receiptErr)
			}
		})
	}
	// 通用流的 EOF 合同由其调用方决定，原生 Provider 的校验不能污染它。
	generic := streamx.NewStreamWithContext(t.Context(), strings.NewReader("data: "+frames[2]+"\n\n"), streamx.ClaudeFormat)
	defer generic.Close()
	if _, err := generic.Collect(); err != nil {
		t.Fatalf("generic EOF contract changed: %v", err)
	}
}
