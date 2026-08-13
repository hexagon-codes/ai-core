package router

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

// ============================================================================
// BUG-1: Router.Register 重复注册同一 Provider 导致 providerList 膨胀
// ============================================================================

func TestRouter_RegisterDuplicate_ProviderListGrows(t *testing.T) {
	r := New()
	mock := &mockRouterProvider{name: "openai"}

	r.Register("openai", mock)
	r.Register("openai", mock)
	r.Register("openai", mock)

	r.mu.RLock()
	listLen := len(r.providerList)
	r.mu.RUnlock()

	if listLen != 1 {
		t.Errorf("BUG: 重复注册导致 providerList 有 %d 个 'openai' 条目（期望 1）。"+
			"这会导致 RoundRobin 策略多次选中同一 Provider，Weighted 策略权重翻倍", listLen)
	}
}

// ============================================================================
// BUG-2: Router.selectProvider 使用写锁但大多数策略只需要读锁
// ============================================================================

func TestRouter_SelectProviderWriteLockContention(t *testing.T) {
	r := New(WithStrategy(StrategyRandom))
	r.Register("p1", &mockRouterProvider{name: "p1"})
	r.Register("p2", &mockRouterProvider{name: "p2"})

	// 由于 selectProvider 使用 mu.Lock()（写锁），
	// 并发 Complete 调用会串行化，即使策略本身不需要修改状态
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.Complete(context.Background(), llm.CompletionRequest{})
		}()
	}
	wg.Wait()

	t.Log("PERF: selectProvider 使用 mu.Lock()（写锁），即使 Random/LeastLatency 等策略只需读操作。" +
		"只有 RoundRobin 需要写锁（递增 rrIndex）。应分离 rrIndex 为 atomic 操作")
}

// ============================================================================
// BUG-3: Router.Complete 和 recordLatency 之间的竞态窗口
// ============================================================================

func TestRouter_CompleteThenRecordLatency_LockGap(t *testing.T) {
	// selectProvider() 获取 mu.Lock → 释放
	// provider.Complete() 执行（无锁）
	// getProviderName() 获取 mu.RLock → 释放
	// recordLatency() 获取 mu.Lock → 释放
	//
	// 在这 3 次 lock/unlock 之间，其他 goroutine 可以修改 providers map
	// 比如 Unregister 掉刚选中的 provider

	r := New()
	mock := &mockRouterProvider{name: "p1"}
	r.Register("p1", mock)

	var wg sync.WaitGroup

	// 并发 Complete
	wg.Add(1)
	go func() {
		defer wg.Done()
		r.Complete(context.Background(), llm.CompletionRequest{})
	}()

	// 并发 Unregister
	wg.Add(1)
	go func() {
		defer wg.Done()
		r.Unregister("p1")
	}()

	wg.Wait()
	// 不应该 panic，但 getProviderName 可能返回空字符串
	// recordLatency 会给空字符串 key 记录延迟
}

// ============================================================================
// BUG-4: Router.Complete fallback 没有记录延迟
// ============================================================================

func TestRouter_FallbackDoesNotRecordLatency(t *testing.T) {
	primary := &mockRouterProvider{
		name: "primary",
		completeErr: func() error {
			return context.DeadlineExceeded
		},
	}
	fallback := &mockRouterProvider{name: "fallback"}

	r := New(WithStrategy(StrategyRoundRobin), WithFallback("fallback"))
	r.Register("primary", primary)
	r.Register("fallback", fallback)

	r.Complete(context.Background(), llm.CompletionRequest{})

	stats := r.GetStats()

	// Primary 的延迟被记录了
	// 但 Fallback 的延迟没有被记录
	if stats.Providers["fallback"].Latency > 0 {
		t.Log("Fallback latency recorded (good)")
	} else {
		t.Log("BUG: Fallback 执行后未记录延迟，LeastLatency 策略无法评估 fallback 的性能")
	}
}

// ============================================================================
// BUG-5: leastCostSelect 的成本比较逻辑有误
// ============================================================================

func TestRouter_LeastCostSelect_ZeroCostProvider(t *testing.T) {
	r := New(WithStrategy(StrategyLeastCost))

	// Provider1: 有模型信息和成本
	p1 := &mockRouterProvider{
		name: "p1",
		models: []llm.ModelInfo{
			{ID: "gpt-4", InputCost: 30, OutputCost: 60},
		},
	}
	// Provider2: 有模型但成本为 0（免费或未配置）
	p2 := &mockRouterProvider{
		name: "p2",
		models: []llm.ModelInfo{
			{ID: "free-model", InputCost: 0, OutputCost: 0},
		},
	}

	r.Register("p1", p1)
	r.Register("p2", p2)

	// leastCostSelect 用 i==0 来判断第一个 provider
	// 但当 model 不匹配时，可能跳过所有 models 导致 selected 为空
	provider, err := r.selectProvider(llm.CompletionRequest{Model: "unknown-model"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Selected provider: %s", provider.Name())

	// 当指定模型匹配不到任何 provider 的模型时，
	// leastCostSelect 可能返回 available[0] 而不是真正最便宜的
}

// ============================================================================
// BUG-6: CreateDefaultRouter fallback 选择不确定
// ============================================================================

func TestCreateDefaultRouter_FallbackIsNondeterministic(t *testing.T) {
	providers := map[string]llm.Provider{
		"openai":    &mockRouterProvider{name: "openai"},
		"anthropic": &mockRouterProvider{name: "anthropic"},
		"deepseek":  &mockRouterProvider{name: "deepseek"},
	}

	// 多次创建 router，检查 fallback 是否一致
	fallbacks := make(map[string]bool)
	for i := 0; i < 10; i++ {
		r := CreateDefaultRouter(providers)
		fallbacks[r.fallback] = true
	}

	if len(fallbacks) > 1 {
		t.Logf("BUG: CreateDefaultRouter 的 fallback 选择不确定（map 遍历顺序随机）: %v。"+
			"多次创建 router 可能选择不同的 fallback", fallbacks)
	}
}

// ============================================================================
// BUG-7: HealthChecker.checkAll 使用了已取消的 ctx
// ============================================================================

func TestHealthChecker_UsesTickerCtxAfterStop(t *testing.T) {
	r := New()
	r.Register("p1", &mockRouterProvider{name: "p1"})

	checker := NewHealthChecker(r, 50*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())

	checker.Start(ctx)
	time.Sleep(100 * time.Millisecond) // 让 checker 运行一两次

	// 取消 ctx
	cancel()
	time.Sleep(50 * time.Millisecond)

	// checkAll 使用 Start 时传入的 ctx
	// ctx 取消后，checkProvider 的 CountTokens 调用不受影响（因为 CountTokens 不接受 ctx）
	// 但 checkProvider 自己的 WithTimeout context 基于已取消的 ctx，会立即超时
	t.Log("DESIGN: HealthChecker 使用 Start 时的 ctx 作为 checkProvider 的 parent context。" +
		"当 ctx 取消时，所有健康检查会立即超时，Provider 被标记为不健康")
}

// ============================================================================
// BUG-8: HealthChecker.checkProvider 必须把 ctx 传给可取消的 Token 计数实现
// ============================================================================

func TestHealthChecker_CheckProviderPropagatesTokenCountContext(t *testing.T) {
	provider := newContextTokenHealthProvider()
	checker := NewHealthChecker(New(), time.Hour)
	ctx, cancel := context.WithCancel(context.Background())

	result := make(chan bool, 1)
	go func() {
		result <- checker.checkProvider(ctx, provider)
	}()

	select {
	case <-provider.contextEntered:
		cancel()
	case healthy := <-result:
		cancel()
		t.Fatalf("checkProvider() returned %v before entering context-aware token counting", healthy)
	case <-time.After(time.Second):
		cancel()
		t.Fatal("checkProvider() did not enter context-aware token counting")
	}

	select {
	case healthy := <-result:
		if healthy {
			t.Fatal("checkProvider() = true after context cancellation, want false")
		}
	case <-time.After(time.Second):
		t.Fatal("checkProvider() did not return after context cancellation")
	}

	select {
	case err := <-provider.contextResult:
		if err != context.Canceled {
			t.Fatalf("context-aware token counting error = %v, want context.Canceled", err)
		}
	default:
		t.Fatal("context-aware token counting did not report its context result")
	}

	select {
	case <-provider.legacyCalled:
		t.Fatal("checkProvider() called legacy CountTokens")
	default:
	}
}

// ============================================================================
// BUG-9: Router.Unregister 后 RoundRobin index 可能越界
// ============================================================================

func TestRouter_UnregisterMidRoundRobin(t *testing.T) {
	r := New(WithStrategy(StrategyRoundRobin))
	r.Register("p1", &mockRouterProvider{name: "p1"})
	r.Register("p2", &mockRouterProvider{name: "p2"})
	r.Register("p3", &mockRouterProvider{name: "p3"})

	// 执行几次让 rrIndex 增长
	for i := 0; i < 5; i++ {
		r.Complete(context.Background(), llm.CompletionRequest{})
	}

	// 注销一个 provider
	r.Unregister("p2")

	// rrIndex 可能大于新的 providerList 长度
	// roundRobinSelect 用 % 取模，所以不会越界 panic
	// 但连续调用可能不均匀
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("BUG: Unregister 后 RoundRobin panic: %v", r)
		}
	}()

	for i := 0; i < 10; i++ {
		r.Complete(context.Background(), llm.CompletionRequest{})
	}
}

// ============================================================================
// BENCHMARK: Router.selectProvider 锁竞争
// ============================================================================

func BenchmarkRouter_SelectProvider_Parallel(b *testing.B) {
	r := New(WithStrategy(StrategyRoundRobin))
	for i := 0; i < 10; i++ {
		r.Register(
			fmt.Sprintf("p%d", i),
			&mockRouterProvider{name: fmt.Sprintf("p%d", i)},
		)
	}

	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		req := llm.CompletionRequest{}
		for pb.Next() {
			r.Complete(ctx, req)
		}
	})
}

// ============================================================================
// 辅助类型
// ============================================================================

type mockRouterProvider struct {
	name        string
	models      []llm.ModelInfo
	completeErr func() error
}

type contextTokenHealthProvider struct {
	*mockRouterProvider
	contextEntered chan struct{}
	contextResult  chan error
	legacyCalled   chan struct{}
}

func newContextTokenHealthProvider() *contextTokenHealthProvider {
	return &contextTokenHealthProvider{
		mockRouterProvider: &mockRouterProvider{name: "context-token-health"},
		contextEntered:     make(chan struct{}),
		contextResult:      make(chan error, 1),
		legacyCalled:       make(chan struct{}, 1),
	}
}

func (p *contextTokenHealthProvider) CountTokens(_ []llm.Message) (int, error) {
	p.legacyCalled <- struct{}{}
	return 1, nil
}

func (p *contextTokenHealthProvider) CountTokensContext(
	ctx context.Context,
	_ []llm.Message,
) (int, error) {
	close(p.contextEntered)
	<-ctx.Done()
	p.contextResult <- ctx.Err()
	return 0, ctx.Err()
}

func (m *mockRouterProvider) Name() string { return m.name }
func (m *mockRouterProvider) Models() []llm.ModelInfo {
	if m.models != nil {
		return m.models
	}
	return []llm.ModelInfo{{ID: m.name + "-model"}}
}
func (m *mockRouterProvider) CountTokens(messages []llm.Message) (int, error) { return 0, nil }
func (m *mockRouterProvider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if m.completeErr != nil {
		return nil, m.completeErr()
	}
	return &llm.CompletionResponse{Content: "ok from " + m.name}, nil
}
func (m *mockRouterProvider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	return nil, nil
}

var _ llm.Provider = (*mockRouterProvider)(nil)
var _ llm.Provider = (*contextTokenHealthProvider)(nil)
