package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRetryMiddlewareDoesNotReplayNonIdempotentOperation(t *testing.T) {
	provider := &mockProvider{
		name:        "test",
		completeErr: errors.New("temporary upstream reset"),
	}
	wrapped := Chain(provider, WithRetry(3, time.Millisecond))
	ctx := WithOperationSafety(context.Background(), OperationSafetyNonIdempotent)

	if _, err := wrapped.Complete(ctx, CompletionRequest{}); err == nil {
		t.Fatal("expected provider error")
	}
	if provider.callCount.Load() != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.callCount.Load())
	}
}
