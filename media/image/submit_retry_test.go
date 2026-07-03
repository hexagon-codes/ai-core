package image

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// newCountingFailServer 返回一个统计请求次数并恒定 5xx 的服务器。
func newCountingFailServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

// 无幂等键的任务创建 POST 属计费操作，5xx 语义二义（任务可能已建成计费），
// 绝不能自动重试；带幂等键时上游可去重，维持 transport 默认重试。
func TestFluxSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewFlux("key", FluxWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "cat"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestFluxSubmitWithIdempotencyKeyKeepsDefaultRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewFlux("key", FluxWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "cat", IdempotencyKey: "task-42"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 3 {
		t.Fatalf("submit attempts = %d, want 3 (idempotency key set keeps default retry)", calls.Load())
	}
}

func TestAsyncCompatibleSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewAsyncCompatible("key", CompatibleConfig{
		Name:             "gw",
		BaseURL:          srv.URL,
		SubmitPath:       "/v1/images/generations",
		PollPathTemplate: "/v1/images/generations/{task_id}",
	})
	if _, err := p.Submit(context.Background(), Request{Prompt: "cat"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

// OpenAI 兼容图片生成不向上游透传幂等键，因此即便调用方设置了
// IdempotencyKey，生成 POST 也不能重试（上游无法去重）。
func TestOpenAICompatGenerateNeverRetriesEvenWithIdempotencyKey(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewOpenAIDallE("key", srv.URL)
	if _, err := p.Generate(context.Background(), Request{Prompt: "cat", IdempotencyKey: "task-42"}); err == nil {
		t.Fatal("expected generate error")
	}
	if calls.Load() != 1 {
		t.Fatalf("generate attempts = %d, want 1 (idempotency key is not transmitted upstream)", calls.Load())
	}
}
