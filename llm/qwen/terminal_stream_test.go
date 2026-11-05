package qwen

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestStreamRejectsTruncatedResponse(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"1\",\"model\":\"qwen-max\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
		// 不发送 [DONE] 或 finish_reason，模拟上游/代理截断。
	}))
	defer server.Close()

	provider := New("test-key", WithBaseURL(server.URL))
	stream, err := provider.Stream(context.Background(), llm.CompletionRequest{
		Model:    "qwen-max",
		Messages: []llm.Message{llm.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()

	result, err := stream.Collect()
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("Collect() error = %v, want io.ErrUnexpectedEOF (result=%+v)", err, result)
	}
}

func TestStreamRejectsMalformedFirstFrame(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: not-json\n\n")
	}))
	defer server.Close()

	provider := New("test-key", WithBaseURL(server.URL))
	stream, err := provider.Stream(context.Background(), llm.CompletionRequest{
		Model:    "qwen-max",
		Messages: []llm.Message{llm.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()

	if _, err := stream.Collect(); err == nil {
		t.Fatal("Collect() error = nil, want malformed frame error")
	}
}

func TestStreamAcceptsCompletedResponse(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"1\",\"model\":\"qwen-max\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := New("test-key", WithBaseURL(server.URL))
	stream, err := provider.Stream(context.Background(), llm.CompletionRequest{
		Model:    "qwen-max",
		Messages: []llm.Message{llm.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()

	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if result.Content != "hi" || result.FinishReason != "stop" {
		t.Fatalf("unexpected result: %+v", result)
	}
}
