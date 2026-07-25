package router

import (
	"context"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestRouterDoesNotFallbackNonIdempotentOperation(t *testing.T) {
	primary := &countingProvider{name: "primary"}
	fallback := &countingProvider{name: "fallback"}
	r := New(WithFallback("fallback"))
	r.Register("primary", primary)
	r.Register("fallback", fallback)
	ctx := llm.WithOperationSafety(context.Background(), llm.OperationSafetyNonIdempotent)

	if _, err := r.Complete(ctx, llm.CompletionRequest{Model: "primary-model"}); err == nil {
		t.Fatal("expected primary provider error")
	}
	if primary.calls.Load() != 1 || fallback.calls.Load() != 0 {
		t.Fatalf("provider calls primary=%d fallback=%d, want 1/0",
			primary.calls.Load(), fallback.calls.Load())
	}
}

func TestExecuteWithRetryDoesNotReplayNonIdempotentOperation(t *testing.T) {
	provider := &countingProvider{name: "primary"}
	r := New()
	r.Register("primary", provider)
	ctx := llm.WithOperationSafety(context.Background(), llm.OperationSafetyNonIdempotent)

	if _, err := ExecuteWithRetry(ctx, r, llm.CompletionRequest{}, 5); err == nil {
		t.Fatal("expected provider error")
	}
	if provider.calls.Load() != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.calls.Load())
	}
}
