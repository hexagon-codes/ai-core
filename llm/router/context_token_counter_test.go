package router

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

type routerContextTokenProvider struct {
	*mockRouterProvider
	contextCalls int
	legacyCalls  int
	receivedCtx  context.Context
}

type routerContextKey struct{}

func (p *routerContextTokenProvider) CountTokens(_ []llm.Message) (int, error) {
	p.legacyCalls++
	return 11, nil
}

func (p *routerContextTokenProvider) CountTokensContext(
	ctx context.Context,
	_ []llm.Message,
) (int, error) {
	p.contextCalls++
	p.receivedCtx = ctx
	return 23, nil
}

type routerLegacyTokenProvider struct {
	*mockRouterProvider
	legacyCalls int
}

func (p *routerLegacyTokenProvider) CountTokens(_ []llm.Message) (int, error) {
	p.legacyCalls++
	return 11, nil
}

// TestRouter_CountTokensContextPropagatesOptionalCapability 验证嵌套路由不遮蔽新能力。
func TestRouter_CountTokensContextPropagatesOptionalCapability(t *testing.T) {
	provider := &routerContextTokenProvider{
		mockRouterProvider: &mockRouterProvider{name: "context"},
	}
	r := New()
	r.Register("context", provider)
	ctx := context.WithValue(context.Background(), routerContextKey{}, "trace")

	got, err := r.CountTokensContext(ctx, nil)
	if err != nil || got != 23 {
		t.Fatalf("CountTokensContext() = (%d, %v), want (23, nil)", got, err)
	}
	if provider.contextCalls != 1 || provider.receivedCtx != ctx {
		t.Fatalf("context calls = %d, received context = %v, want one call with original context", provider.contextCalls, provider.receivedCtx)
	}
	if provider.legacyCalls != 0 {
		t.Fatalf("legacy CountTokens calls = %d, want 0", provider.legacyCalls)
	}
}

// TestRouter_CountTokensContextRejectsLegacyProvider 验证路由器不会阻塞式降级并伪装可取消。
func TestRouter_CountTokensContextRejectsLegacyProvider(t *testing.T) {
	provider := &routerLegacyTokenProvider{
		mockRouterProvider: &mockRouterProvider{name: "legacy"},
	}
	r := New()
	r.Register("legacy", provider)

	got, err := r.CountTokensContext(context.Background(), nil)
	if got != 0 || !errors.Is(err, llm.ErrContextTokenCountingUnsupported) {
		t.Fatalf("CountTokensContext() = (%d, %v), want (0, ErrContextTokenCountingUnsupported)", got, err)
	}
	if provider.legacyCalls != 0 {
		t.Fatalf("legacy CountTokens calls = %d, want 0", provider.legacyCalls)
	}
}

// TestHealthChecker_SkipsLegacyTokenCounter 验证旧 Provider 保持既有健康态且不会被阻塞探测。
func TestHealthChecker_SkipsLegacyTokenCounter(t *testing.T) {
	provider := &routerLegacyTokenProvider{
		mockRouterProvider: &mockRouterProvider{name: "legacy"},
	}
	r := New()
	r.Register("legacy", provider)
	r.SetHealthy("legacy", false)

	NewHealthChecker(r, time.Hour).checkAll(context.Background())

	if r.GetStats().Providers["legacy"].Healthy {
		t.Fatal("legacy provider health changed, want previous unhealthy state preserved")
	}
	if provider.legacyCalls != 0 {
		t.Fatalf("legacy CountTokens calls = %d, want 0", provider.legacyCalls)
	}
}

var (
	_ llm.Provider            = (*routerContextTokenProvider)(nil)
	_ llm.ContextTokenCounter = (*routerContextTokenProvider)(nil)
	_ llm.Provider            = (*routerLegacyTokenProvider)(nil)
)
