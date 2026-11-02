package openai

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestCompletionResponsesActualHTTPFinishReasonAndPartialFacts(t *testing.T) {
	for _, test := range []struct {
		name, status, reason, contentType, failureCode, wantReason string
		wantError, wantUnknown                                     bool
	}{
		{name: "length", status: "incomplete", reason: "max_output_tokens", wantReason: "length", wantError: true},
		{name: "filter", status: "incomplete", reason: "content_filter", wantReason: "content_filter", wantError: true},
		{name: "refusal", status: "failed", contentType: "refusal", failureCode: "refusal", wantReason: "refusal", wantError: true},
		{name: "completed refusal", status: "completed", contentType: "refusal", wantReason: "refusal"},
		{name: "cancelled", status: "cancelled", wantReason: "cancelled", wantError: true},
		{name: "failed", status: "failed", failureCode: "server_error", wantReason: "failed", wantError: true},
		{name: "queued", status: "queued", wantReason: "queued", wantError: true, wantUnknown: true},
		{name: "in progress", status: "in_progress", wantReason: "in_progress", wantError: true, wantUnknown: true},
		{name: "unknown status", status: "provider_future_status", wantReason: "provider_future_status", wantError: true, wantUnknown: true},
		{name: "unknown incomplete reason", status: "incomplete", reason: "provider_future_reason", wantReason: "provider_future_reason", wantError: true},
		{name: "completed", status: "completed", wantReason: "stop"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("x-request-id", "request-known")
				kind := test.contentType
				if kind == "" {
					kind = "output_text"
				}
				content := map[string]any{"type": kind}
				if kind == "refusal" {
					content["refusal"] = "known partial text"
				} else {
					content["text"] = "known partial text"
				}
				body := map[string]any{"id": "response-known", "model": "actual-model", "status": test.status, "output": []any{map[string]any{"id": "message-known", "type": "message", "role": "assistant", "content": []any{content}}}, "usage": map[string]int{"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}
				if test.reason != "" {
					body["incomplete_details"] = map[string]string{"reason": test.reason}
				}
				if test.failureCode != "" {
					body["error"] = map[string]string{"code": test.failureCode, "message": "synthetic model failure"}
				}
				if err := json.NewEncoder(w).Encode(body); err != nil {
					t.Error(err)
				}
			}))
			defer server.Close()
			response, err := New("synthetic", WithBaseURL(server.URL)).Complete(WithResponsesProtocol(t.Context(), "synthetic-scope"), llm.CompletionRequest{Model: "requested-model", Messages: []llm.Message{{Role: llm.RoleUser, Content: "synthetic request"}}})
			if (err != nil) != test.wantError {
				t.Fatalf("response error=%v wantError=%v", err, test.wantError)
			}
			if response == nil || response.FinishReason != test.wantReason || response.Content != "known partial text" || response.UsageReceipt == nil || response.UsageReceipt.TotalTokens != 3 || response.RequestID != "request-known" || response.Model != "actual-model" {
				t.Fatalf("terminal/partial facts lost: %+v error=%v", response, err)
			}
			if test.wantError {
				var typed *ResponsesError
				if !errors.As(err, &typed) || typed.OutcomeUnknown != test.wantUnknown || !typed.Dispatched {
					t.Fatalf("original error facts changed: %+v error=%v", typed, err)
				}
				if response.ProviderContinuation != nil {
					t.Fatal("non-completed response produced continuation")
				}
			}
			if test.status == "completed" && response.ProviderContinuation == nil {
				t.Fatal("valid completed continuation was lost")
			}
			if calls.Load() != 1 {
				t.Fatalf("terminal response was blindly resent: %d", calls.Load())
			}
		})
	}
}
