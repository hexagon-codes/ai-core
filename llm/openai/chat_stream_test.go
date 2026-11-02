package openai

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

func TestChatStreamRequiresCompletionEvidence(t *testing.T) {
	partial := "data: {\"choices\":[{\"delta\":{\"content\":\"partial\",\"tool_calls\":[{\"index\":0,\"id\":\"call\",\"type\":\"function\",\"function\":{\"name\":\"write\",\"arguments\":\"{}\"}}]}}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n"
	for _, tc := range []struct {
		name, suffix string
		incomplete   bool
	}{
		{name: "missing terminal", incomplete: true},
		{name: "finish reason", suffix: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"},
		{name: "done marker", suffix: "data: [DONE]\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, partial+tc.suffix)
			}))
			defer server.Close()
			provider := New("local", WithBaseURL(server.URL), WithModel("local"))
			stream, err := provider.Stream(WithChatCompletionsProtocol(t.Context()), llm.CompletionRequest{})
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			for range stream.Chunks() {
			}
			result := stream.Result()
			if errors.Is(stream.Err(), io.ErrUnexpectedEOF) != tc.incomplete {
				t.Fatalf("terminal error = %v, incomplete = %v", stream.Err(), tc.incomplete)
			}
			if result.Content != "partial" || len(result.ToolCalls) != 1 || result.Usage.TotalTokens != 3 {
				t.Fatalf("partial response facts were lost: %+v", result)
			}
		})
	}
	// 通用解析器仍允许调用方以 EOF 表达结束，完整性要求只归 Chat 适配器。
	stream := streamx.NewStreamWithContext(context.Background(), strings.NewReader(partial), streamx.OpenAIFormat)
	defer stream.Close()
	if _, err := stream.Collect(); err != nil {
		t.Fatalf("generic stream EOF contract changed: %v", err)
	}
}
