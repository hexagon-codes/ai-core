package router

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

// countingProvider counts how many times Complete is invoked and always fails.
type countingProvider struct {
	name  string
	calls atomic.Int64
}

func (p *countingProvider) Name() string            { return p.name }
func (p *countingProvider) Models() []llm.ModelInfo { return []llm.ModelInfo{{ID: p.name + "-model"}} }
func (p *countingProvider) CountTokens(messages []llm.Message) (int, error) {
	return 0, nil
}
func (p *countingProvider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	p.calls.Add(1)
	return nil, errors.New("boom")
}
func (p *countingProvider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	return nil, nil
}

var _ llm.Provider = (*countingProvider)(nil)

// TestExecuteWithRetry_Bug_PreCancelledCtx verifies ExecuteWithRetry stops
// immediately when the context is already cancelled, instead of calling the
// provider maxRetries times.
func TestExecuteWithRetry_Bug_PreCancelledCtx(t *testing.T) {
	prov := &countingProvider{name: "p"}
	r := New()
	r.Register("p", prov)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before any attempt

	_, err := ExecuteWithRetry(ctx, r, llm.CompletionRequest{}, 5)
	if err == nil {
		t.Fatal("expected error from pre-cancelled context")
	}
	if got := prov.calls.Load(); got != 0 {
		t.Fatalf("provider called %d times with pre-cancelled ctx, want 0", got)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected ctx cancellation error, got %v", err)
	}
}

// TestExecuteWithRetry_Bug_CancelDuringRetry verifies that cancelling the
// context partway stops further attempts promptly (backoff is interruptible).
func TestExecuteWithRetry_Bug_CancelDuringRetry(t *testing.T) {
	prov := &countingProvider{name: "p"}
	r := New()
	r.Register("p", prov)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := ExecuteWithRetry(ctx, r, llm.CompletionRequest{}, 100)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error")
	}
	// With ctx-awareness, must stop well before exhausting 100 attempts.
	if got := prov.calls.Load(); got >= 100 {
		t.Fatalf("provider called %d times despite cancellation, want far fewer", got)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("ExecuteWithRetry took %v, expected to stop promptly after cancel", elapsed)
	}
}
