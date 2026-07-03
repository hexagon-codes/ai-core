package voice

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

// 无幂等键的合成/转录 POST 属计费操作，5xx 语义二义（上游可能已合成计费），
// 绝不能自动重试；带幂等键时上游可去重，维持 transport 默认重试。
func TestMiniMaxSynthesizeWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	tts := NewMiniMaxTTS("key", "group", WithMiniMaxBaseURL(srv.URL))
	if _, err := tts.Synthesize(context.Background(), "你好", SynthesizeOptions{}); err == nil {
		t.Fatal("expected synthesize error")
	}
	if calls.Load() != 1 {
		t.Fatalf("synthesize attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestMiniMaxSynthesizeWithIdempotencyKeyKeepsDefaultRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	tts := NewMiniMaxTTS("key", "group", WithMiniMaxBaseURL(srv.URL))
	if _, err := tts.Synthesize(context.Background(), "你好", SynthesizeOptions{IdempotencyKey: "task-42"}); err == nil {
		t.Fatal("expected synthesize error")
	}
	if calls.Load() != 3 {
		t.Fatalf("synthesize attempts = %d, want 3 (idempotency key set keeps default retry)", calls.Load())
	}
}

func TestElevenLabsSynthesizeWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	tts := NewElevenLabsTTS("key", ElevenLabsWithBaseURL(srv.URL))
	if _, err := tts.Synthesize(context.Background(), "hello", SynthesizeOptions{}); err == nil {
		t.Fatal("expected synthesize error")
	}
	if calls.Load() != 1 {
		t.Fatalf("synthesize attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

func TestAzureSynthesizeWithoutIdempotencyKeyDoesNotRetry(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	tts := NewAzureTTS("key", "eastus", AzureTTSWithBaseURL(srv.URL))
	if _, err := tts.Synthesize(context.Background(), "hello", SynthesizeOptions{}); err == nil {
		t.Fatal("expected synthesize error")
	}
	if calls.Load() != 1 {
		t.Fatalf("synthesize attempts = %d, want 1 (no idempotency key, 5xx must not be retried)", calls.Load())
	}
}

// OpenAI 兼容 TTS 不向上游透传幂等键，因此即便调用方设置了 IdempotencyKey，
// 合成 POST 也不能重试（上游无法去重）。
func TestOpenAITTSSynthesizeNeverRetriesEvenWithIdempotencyKey(t *testing.T) {
	srv, calls := newCountingFailServer(t)
	tts := NewOpenAITTS("key", "tts-1", TTSWithBaseURL(srv.URL))
	if _, err := tts.Synthesize(context.Background(), "hello", SynthesizeOptions{IdempotencyKey: "task-42"}); err == nil {
		t.Fatal("expected synthesize error")
	}
	if calls.Load() != 1 {
		t.Fatalf("synthesize attempts = %d, want 1 (idempotency key is not transmitted upstream)", calls.Load())
	}
}
