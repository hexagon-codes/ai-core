package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestStreamCarriesHTTPRequestID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("x-request-id", "req-stream-1")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"resp\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithModel("m"))
	stream, err := p.Stream(context.Background(), llm.CompletionRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	for range stream.Chunks() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream Err() = %v", err)
	}
	if got := stream.Result().RequestID; got != "req-stream-1" {
		t.Fatalf("Result().RequestID = %q, want req-stream-1", got)
	}
}
