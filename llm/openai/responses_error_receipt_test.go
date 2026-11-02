package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestNativeResponsesErrorsRetainHTTPClassification(t *testing.T) {
	for _, test := range []struct {
		status int
		want   llm.FailoverReason
	}{{401, llm.FailInvalidKey}, {429, llm.FailRateLimit}} {
		t.Run(fmt.Sprint(test.status), func(t *testing.T) {
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(`{"error":{"message":"request rejected","type":"test_error","code":"fixture"}}`))
			}))
			defer server.Close()
			provider := New("fixture", WithBaseURL(server.URL))
			_, err := provider.CreateResponse(context.Background(), ResponsesRequest{
				Model: "test-model",
				Input: []ResponseInputItem{{Type: "message", Role: "user", Content: []ResponseInputContent{{Type: "input_text", Text: "hello"}}}},
			})
			if err == nil {
				t.Fatal("expected upstream failure")
			}
			if got := llm.ClassifyError(fmt.Errorf("wrapped: %w", err), 0, ""); got != test.want {
				t.Fatalf("classification=%s, want %s", got, test.want)
			}
			if calls.Load() != 1 {
				t.Fatalf("physical requests=%d, want exactly one", calls.Load())
			}
		})
	}
}

func TestCompletionResponsesKeepsUsageOnInvalidOutput(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Request-Id", "request-known")
		_, _ = w.Write([]byte(`{"id":"response-known","model":"test-model","status":"completed","usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10},"output":[{"id":"invalid-item","content":[{"type":"output_text","text":"unverified"}]}]}`))
	}))
	defer server.Close()
	provider := New("fixture", WithBaseURL(server.URL))
	response, err := provider.Complete(WithResponsesProtocol(context.Background(), "test-scope"), llm.CompletionRequest{
		Model: "test-model", Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	})
	if err == nil {
		t.Fatal("invalid output was accepted")
	}
	if response == nil || response.UsageReceipt == nil {
		t.Fatalf("reported usage was lost: response=%+v error=%v", response, err)
	}
	if response.ID != "response-known" || response.RequestID != "request-known" || response.UsageReceipt.PromptTokens != 7 || response.UsageReceipt.CompletionTokens != 3 || response.UsageReceipt.TotalTokens != 10 {
		t.Fatalf("trusted receipt=%+v, want known identity and 7/3/10 usage", response)
	}
	if response.Content != "" || len(response.ToolCalls) != 0 || response.ProviderContinuation != nil || response.FinishReason != "" {
		t.Fatalf("unvalidated output escaped through partial response: %+v", response)
	}
	if calls.Load() != 1 {
		t.Fatalf("physical requests=%d, want one", calls.Load())
	}
}
