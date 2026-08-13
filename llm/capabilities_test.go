package llm

import (
	"context"
	"errors"
	"testing"
)

// completerOnly 只实现 Complete，用于验证消费方可只依赖窄接口 Completer。
type completerOnly struct{ reply string }

type capabilitiesContextKey struct{}

func (c *completerOnly) Complete(_ context.Context, _ CompletionRequest) (*CompletionResponse, error) {
	return &CompletionResponse{Content: c.reply}, nil
}

// useCompleter 是只依赖 Completer 窄接口的消费方。
func useCompleter(ctx context.Context, c Completer, q string) (string, error) {
	resp, err := c.Complete(ctx, CompletionRequest{Messages: []Message{UserMessage(q)}})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// TestCompleter_NarrowDependency 验证只实现 Complete 的类型即可被只依赖 Completer 的消费方使用。
func TestCompleter_NarrowDependency(t *testing.T) {
	got, err := useCompleter(context.Background(), &completerOnly{reply: "ok"}, "hi")
	if err != nil {
		t.Fatalf("useCompleter: %v", err)
	}
	if got != "ok" {
		t.Errorf("got %q, want ok", got)
	}
}

type legacyTokenCounter struct {
	count int
	err   error
	calls int
}

func (c *legacyTokenCounter) CountTokens(_ []Message) (int, error) {
	c.calls++
	return c.count, c.err
}

type contextAwareTokenCounter struct {
	*legacyTokenCounter
	contextCount int
	contextErr   error
	contextCalls int
	receivedCtx  context.Context
}

func (c *contextAwareTokenCounter) CountTokensContext(
	ctx context.Context,
	_ []Message,
) (int, error) {
	c.contextCalls++
	c.receivedCtx = ctx
	return c.contextCount, c.contextErr
}

// TestCountTokensContext_PrefersContextAwareCounter 验证新能力优先于旧方法。
func TestCountTokensContext_PrefersContextAwareCounter(t *testing.T) {
	wantErr := errors.New("context token count failed")
	counter := &contextAwareTokenCounter{
		legacyTokenCounter: &legacyTokenCounter{count: 11},
		contextCount:       23,
		contextErr:         wantErr,
	}
	ctx := context.WithValue(context.Background(), capabilitiesContextKey{}, "trace")

	got, err := CountTokensContext(ctx, counter, []Message{UserMessage("hi")})
	if got != 23 || !errors.Is(err, wantErr) {
		t.Fatalf("CountTokensContext() = (%d, %v), want (23, %v)", got, err, wantErr)
	}
	if counter.contextCalls != 1 || counter.receivedCtx != ctx {
		t.Fatalf("context calls = %d, received context = %v, want one call with original context", counter.contextCalls, counter.receivedCtx)
	}
	if counter.calls != 0 {
		t.Fatalf("legacy CountTokens calls = %d, want 0", counter.calls)
	}
}

// TestCountTokensContext_FallsBackToLegacyCounter 验证旧实现无需改动即可继续工作。
func TestCountTokensContext_FallsBackToLegacyCounter(t *testing.T) {
	wantErr := errors.New("legacy token count failed")
	counter := &legacyTokenCounter{count: 17, err: wantErr}

	got, err := CountTokensContext(context.Background(), counter, nil)
	if got != 17 || !errors.Is(err, wantErr) {
		t.Fatalf("CountTokensContext() = (%d, %v), want (17, %v)", got, err, wantErr)
	}
	if counter.calls != 1 {
		t.Fatalf("legacy CountTokens calls = %d, want 1", counter.calls)
	}
}

// TestCountTokensContext_CanceledBeforeLegacyCall 验证取消发生在旧调用前时不会启动旧实现。
func TestCountTokensContext_CanceledBeforeLegacyCall(t *testing.T) {
	counter := &legacyTokenCounter{count: 17}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	got, err := CountTokensContext(ctx, counter, nil)
	if got != 0 || !errors.Is(err, context.Canceled) {
		t.Fatalf("CountTokensContext() = (%d, %v), want (0, context.Canceled)", got, err)
	}
	if counter.calls != 0 {
		t.Fatalf("legacy CountTokens calls = %d, want 0", counter.calls)
	}
}

var (
	_ TokenCounter        = (*legacyTokenCounter)(nil)
	_ ContextTokenCounter = (*contextAwareTokenCounter)(nil)
)
