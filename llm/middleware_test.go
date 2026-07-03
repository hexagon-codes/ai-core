package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/streamx"
)

// mockProvider 模拟 Provider 用于测试
type mockProvider struct {
	name         string
	completeResp *CompletionResponse
	completeErr  error
	callCount    atomic.Int32
}

func (m *mockProvider) Name() string { return m.name }
func (m *mockProvider) Models() []ModelInfo {
	return []ModelInfo{{ID: "test-model"}}
}
func (m *mockProvider) CountTokens(messages []Message) (int, error) { return 0, nil }
func (m *mockProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	m.callCount.Add(1)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return m.completeResp, m.completeErr
}
func (m *mockProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	m.callCount.Add(1)
	return nil, m.completeErr
}

// TestChain 测试中间件链
func TestChain(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "hello"},
	}

	var order []string

	// 创建三个记录调用顺序的中间件
	makeMiddleware := func(name string) Middleware {
		return func(next Provider) Provider {
			return &orderTrackingProvider{next: next, name: name, order: &order}
		}
	}

	enhanced := Chain(mock, makeMiddleware("A"), makeMiddleware("B"), makeMiddleware("C"))

	resp, err := enhanced.Complete(context.Background(), CompletionRequest{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "hello" {
		t.Fatalf("expected 'hello', got '%s'", resp.Content)
	}

	// A → B → C → Provider
	expected := []string{"A-before", "B-before", "C-before", "C-after", "B-after", "A-after"}
	if len(order) != len(expected) {
		t.Fatalf("expected %d calls, got %d: %v", len(expected), len(order), order)
	}
	for i, v := range expected {
		if order[i] != v {
			t.Fatalf("expected order[%d]='%s', got '%s'", i, v, order[i])
		}
	}
}

type orderTrackingProvider struct {
	next  Provider
	name  string
	order *[]string
}

func (p *orderTrackingProvider) Name() string        { return p.next.Name() }
func (p *orderTrackingProvider) Models() []ModelInfo { return p.next.Models() }
func (p *orderTrackingProvider) CountTokens(messages []Message) (int, error) {
	return p.next.CountTokens(messages)
}
func (p *orderTrackingProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	return p.next.Stream(ctx, req)
}
func (p *orderTrackingProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	*p.order = append(*p.order, p.name+"-before")
	resp, err := p.next.Complete(ctx, req)
	*p.order = append(*p.order, p.name+"-after")
	return resp, err
}

// TestRetryMiddleware 测试重试中间件
func TestRetryMiddleware(t *testing.T) {
	t.Run("成功不重试", func(t *testing.T) {
		mock := &mockProvider{
			name:         "test",
			completeResp: &CompletionResponse{Content: "ok"},
		}
		p := Chain(mock, WithRetry(3, time.Millisecond))

		resp, err := p.Complete(context.Background(), CompletionRequest{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "ok" {
			t.Fatalf("expected 'ok', got '%s'", resp.Content)
		}
		if mock.callCount.Load() != 1 {
			t.Fatalf("expected 1 call, got %d", mock.callCount.Load())
		}
	})

	t.Run("失败后重试", func(t *testing.T) {
		// 模拟前两次失败，第三次成功
		callCount := atomic.Int32{}
		failProvider := &failThenSucceedProvider{
			failCount: 2,
			callCount: &callCount,
			resp:      &CompletionResponse{Content: "finally"},
		}

		p := Chain(failProvider, WithRetry(3, time.Millisecond))

		resp, err := p.Complete(context.Background(), CompletionRequest{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "finally" {
			t.Fatalf("expected 'finally', got '%s'", resp.Content)
		}
		if callCount.Load() != 3 {
			t.Fatalf("expected 3 calls, got %d", callCount.Load())
		}
	})

	t.Run("全部失败", func(t *testing.T) {
		mock := &mockProvider{
			name:        "test",
			completeErr: errors.New("always fail"),
		}
		p := Chain(mock, WithRetry(2, time.Millisecond))

		_, err := p.Complete(context.Background(), CompletionRequest{})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if mock.callCount.Load() != 3 { // 1 + 2 retries
			t.Fatalf("expected 3 calls, got %d", mock.callCount.Load())
		}
	})

	t.Run("ProviderError 状态码控制重试", func(t *testing.T) {
		badRequest := &mockProvider{
			name: "test",
			completeErr: &ProviderError{
				Provider:   "openai",
				StatusCode: 400,
				Status:     "400 Bad Request",
				Body:       `{"error":"bad request"}`,
			},
		}
		p := Chain(badRequest, WithRetry(3, time.Millisecond))
		_, err := p.Complete(context.Background(), CompletionRequest{})
		if err == nil {
			t.Fatal("expected error")
		}
		if badRequest.callCount.Load() != 1 {
			t.Fatalf("400 ProviderError 不应重试, got %d calls", badRequest.callCount.Load())
		}

		serverError := &mockProvider{
			name: "test",
			completeErr: &ProviderError{
				Provider:   "openai",
				StatusCode: 503,
				Status:     "503 Service Unavailable",
				Body:       `{"error":"down"}`,
			},
		}
		p = Chain(serverError, WithRetry(1, time.Millisecond))
		_, err = p.Complete(context.Background(), CompletionRequest{})
		if err == nil {
			t.Fatal("expected error")
		}
		if serverError.callCount.Load() != 2 {
			t.Fatalf("503 ProviderError 应重试 1 次, got %d calls", serverError.callCount.Load())
		}
	})

	t.Run("NetworkPolicy 错误不应重试", func(t *testing.T) {
		policyBlocked := &mockProvider{
			name: "test",
			completeErr: &ProviderError{
				Provider: "openai",
				Action:   "complete",
				Cause:    fmt.Errorf("%w: scheme \"http\" is not allowed", ErrNetworkPolicy),
			},
		}
		p := Chain(policyBlocked, WithRetry(3, time.Millisecond))
		_, err := p.Complete(context.Background(), CompletionRequest{})
		if err == nil {
			t.Fatal("expected error")
		}
		if policyBlocked.callCount.Load() != 1 {
			t.Fatalf("network policy error should not be retried, got %d calls", policyBlocked.callCount.Load())
		}
	})
}

type failThenSucceedProvider struct {
	failCount int
	callCount *atomic.Int32
	resp      *CompletionResponse
}

func (p *failThenSucceedProvider) Name() string                                { return "test" }
func (p *failThenSucceedProvider) Models() []ModelInfo                         { return nil }
func (p *failThenSucceedProvider) CountTokens(messages []Message) (int, error) { return 0, nil }
func (p *failThenSucceedProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	return nil, nil
}
func (p *failThenSucceedProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	n := p.callCount.Add(1)
	if int(n) <= p.failCount {
		return nil, errors.New("temporary error")
	}
	return p.resp, nil
}

// TestTimeoutMiddleware 测试超时中间件
func TestTimeoutMiddleware(t *testing.T) {
	slow := &slowProvider{delay: 200 * time.Millisecond}
	p := Chain(slow, WithTimeout(50*time.Millisecond))

	_, err := p.Complete(context.Background(), CompletionRequest{})
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

type slowProvider struct {
	delay time.Duration
}

func (p *slowProvider) Name() string                                { return "slow" }
func (p *slowProvider) Models() []ModelInfo                         { return nil }
func (p *slowProvider) CountTokens(messages []Message) (int, error) { return 0, nil }
func (p *slowProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	return nil, nil
}
func (p *slowProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(p.delay):
		return &CompletionResponse{Content: "done"}, nil
	}
}

// TestCallbackMiddleware 测试回调中间件
func TestCallbackMiddleware(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "hello"},
	}

	var startCalled, endCalled bool
	var endEvent *CallbackEndEvent

	cb := &CallbackFunc{
		StartFn: func(ctx context.Context, event *CallbackStartEvent) {
			startCalled = true
			if event.Provider != "test" {
				t.Errorf("expected provider 'test', got '%s'", event.Provider)
			}
		},
		EndFn: func(ctx context.Context, event *CallbackEndEvent) {
			endCalled = true
			endEvent = event
		},
	}

	p := Chain(mock, WithCallback(cb))

	resp, err := p.Complete(context.Background(), CompletionRequest{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "hello" {
		t.Fatalf("expected 'hello', got '%s'", resp.Content)
	}
	if !startCalled {
		t.Fatal("start callback not called")
	}
	if !endCalled {
		t.Fatal("end callback not called")
	}
	if endEvent.Error != nil {
		t.Fatalf("expected no error, got %v", endEvent.Error)
	}
	if endEvent.DurationMs < 0 {
		t.Fatalf("expected non-negative duration, got %d", endEvent.DurationMs)
	}
}

// TestCacheMiddleware 测试缓存中间件
func TestCacheMiddleware(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "cached"},
	}

	cache := &inMemoryTestCache{data: make(map[string]*CompletionResponse)}
	p := Chain(mock, WithCache(cache, nil))

	req := CompletionRequest{Model: "gpt-4o", Messages: []Message{{Role: RoleUser, Content: "hi"}}}

	// 第一次调用 → miss
	resp1, err := p.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp1.Content != "cached" {
		t.Fatalf("expected 'cached', got '%s'", resp1.Content)
	}
	if mock.callCount.Load() != 1 {
		t.Fatalf("expected 1 call, got %d", mock.callCount.Load())
	}

	// 第二次调用 → hit（不调用 Provider）
	resp2, err := p.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp2.Content != "cached" {
		t.Fatalf("expected 'cached', got '%s'", resp2.Content)
	}
	if mock.callCount.Load() != 1 { // 仍然是 1
		t.Fatalf("expected 1 call (cached), got %d", mock.callCount.Load())
	}
}

func TestDefaultCacheKey_IncludesMultiContent(t *testing.T) {
	key1 := defaultCacheKey(&CompletionRequest{
		Model: "gpt-4o",
		Messages: []Message{{
			Role:         RoleUser,
			MultiContent: []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/a.png", Detail: "low"}}},
		}},
	})
	key2 := defaultCacheKey(&CompletionRequest{
		Model: "gpt-4o",
		Messages: []Message{{
			Role:         RoleUser,
			MultiContent: []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/b.png", Detail: "low"}}},
		}},
	})

	if key1 == key2 {
		t.Fatal("different multimodal message content must not share a cache key")
	}
}

func TestDefaultCacheKey_IncludesMetadata(t *testing.T) {
	key1 := defaultCacheKey(&CompletionRequest{
		Model:    "qwen",
		Messages: []Message{{Role: RoleUser, Content: "think?"}},
		Metadata: map[string]any{"enable_thinking": true},
	})
	key2 := defaultCacheKey(&CompletionRequest{
		Model:    "qwen",
		Messages: []Message{{Role: RoleUser, Content: "think?"}},
		Metadata: map[string]any{"enable_thinking": false},
	})

	if key1 == key2 {
		t.Fatal("metadata that changes upstream request semantics must affect cache key")
	}
}

// 序列化失败的请求不能落回 %#v 伪键（会打印指针地址导致同请求键漂移），
// defaultCacheKey 应返回空串标记为不可缓存。
func TestDefaultCacheKey_UnmarshalableRequestYieldsNoKey(t *testing.T) {
	key := defaultCacheKey(&CompletionRequest{
		Model:    "gpt-4o",
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Metadata: map[string]any{"bad": make(chan int)},
	})
	if key != "" {
		t.Fatalf("unmarshalable request must be uncacheable, got key %q", key)
	}
}

// 不可缓存的请求应整体跳过缓存读写：不写入伪键、每次直达底层 Provider。
func TestCacheMiddleware_SkipsUnmarshalableRequest(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "fresh"},
	}
	cache := &inMemoryTestCache{data: make(map[string]*CompletionResponse)}
	p := Chain(mock, WithCache(cache, nil))

	req := CompletionRequest{
		Model:    "gpt-4o",
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Metadata: map[string]any{"bad": make(chan int)},
	}

	for i := 0; i < 2; i++ {
		resp, err := p.Complete(context.Background(), req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "fresh" {
			t.Fatalf("expected 'fresh', got %q", resp.Content)
		}
	}
	if mock.callCount.Load() != 2 {
		t.Fatalf("uncacheable request must reach provider every time, got %d calls", mock.callCount.Load())
	}
	cache.mu.RLock()
	defer cache.mu.RUnlock()
	if len(cache.data) != 0 {
		t.Fatalf("uncacheable request must not be written to cache, got %d entries", len(cache.data))
	}
}

// inMemoryTestCache 测试用缓存（并发安全）
type inMemoryTestCache struct {
	mu   sync.RWMutex
	data map[string]*CompletionResponse
}

func (c *inMemoryTestCache) Get(_ context.Context, key string) (*CompletionResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if resp, ok := c.data[key]; ok {
		return resp, nil
	}
	return nil, nil
}

func (c *inMemoryTestCache) Set(_ context.Context, key string, resp *CompletionResponse) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = resp
	return nil
}

// TestRateLimitMiddleware 测试限流中间件
func TestRateLimitMiddleware(t *testing.T) {
	mock := &mockProvider{
		name:         "test",
		completeResp: &CompletionResponse{Content: "ok"},
	}

	// 限制每秒 100 次
	p := Chain(mock, WithRateLimit(100))

	// 快速连续调用 5 次应该全部成功
	for i := 0; i < 5; i++ {
		_, err := p.Complete(context.Background(), CompletionRequest{})
		if err != nil {
			t.Fatalf("call %d unexpected error: %v", i, err)
		}
	}

	if mock.callCount.Load() != 5 {
		t.Fatalf("expected 5 calls, got %d", mock.callCount.Load())
	}
}

// TestResponseFormat 测试 ResponseFormat 结构
func TestResponseFormat(t *testing.T) {
	rf := &ResponseFormat{
		Type: "json_schema",
		JSONSchema: &ResponseFormatJSONSchema{
			Name:        "user",
			Description: "用户信息",
			Schema:      nil, // 简化测试
			Strict:      true,
		},
	}

	if rf.Type != "json_schema" {
		t.Fatalf("expected type 'json_schema', got '%s'", rf.Type)
	}
	if rf.JSONSchema.Name != "user" {
		t.Fatalf("expected name 'user', got '%s'", rf.JSONSchema.Name)
	}
	if !rf.JSONSchema.Strict {
		t.Fatal("expected strict=true")
	}
}
