package video

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
func TestViduSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewVidu("key", ViduWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "city"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestViduSubmitWithIdempotencyKeyKeepsDefaultRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewVidu("key", ViduWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "city", IdempotencyKey: "task-42"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 3 {
		t.Fatalf("submit attempts = %d, want 3 (idempotency key set keeps default retry)", calls.Load())
	}
}

func TestVeoSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewVeo("key", VeoWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "ocean"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestKlingSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewKling("bearer-token", "", KlingWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "city"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestSeedanceSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewSeedanceCN("key", SeedanceWithBaseURL(srv.URL))
	if _, err := p.Submit(context.Background(), Request{Prompt: "city"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestCompatibleSubmitWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewCompatible("key", CompatibleConfig{
		Name:             "gw",
		BaseURL:          srv.URL,
		SubmitPath:       "/v1/video/generations",
		PollPathTemplate: "/v1/video/generations/{task_id}",
	})
	if _, err := p.Submit(context.Background(), Request{Prompt: "city"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

// zhipu 不向上游透传幂等键，因此即便调用方设置了 IdempotencyKey，
// 任务创建 POST 也不能重试（上游无法去重）。
func TestZhipuSubmitNeverRetriesEvenWithIdempotencyKey(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewZhipuCogVideoX("key", srv.URL)
	if _, err := p.Submit(context.Background(), Request{Prompt: "city", IdempotencyKey: "task-42"}); err == nil {
		t.Fatal("expected submit error")
	}
	if calls.Load() != 1 {
		t.Fatalf("submit attempts = %d, want 1 (idempotency key is not transmitted upstream)", calls.Load())
	}
}

// 轮询 GET 无计费副作用，保持 transport 默认重试。
func TestViduPollKeepsDefaultRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	p := NewVidu("key", ViduWithBaseURL(srv.URL))
	if _, err := p.Poll(context.Background(), "vidu_123"); err == nil {
		t.Fatal("expected poll error")
	}
	if calls.Load() != 3 {
		t.Fatalf("poll attempts = %d, want 3 (poll keeps default retry)", calls.Load())
	}
}
