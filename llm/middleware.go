package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/toolkit/lang/syncx"
	"github.com/hexagon-codes/toolkit/util/retry"
)

// isRetryableError 判断错误是否值得重试
// 认证错误、参数错误等不应重试
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrNetworkPolicy) {
		return false
	}
	var providerErr *ProviderError
	if errors.As(err, &providerErr) && providerErr.StatusCode > 0 {
		switch providerErr.StatusCode {
		case 408, 409, 429:
			return true
		case 400, 401, 402, 403, 404, 422:
			return false
		default:
			return providerErr.StatusCode >= 500
		}
	}
	msg := err.Error()
	// 常见不可重试错误特征
	for _, keyword := range []string{
		"401", "403", "Unauthorized", "Forbidden",
		"invalid api key", "invalid_api_key",
		"authentication", "permission",
	} {
		if strings.Contains(msg, keyword) {
			return false
		}
	}
	return true
}

// Middleware 定义 Provider 装饰器函数
//
// Middleware 接收一个 Provider 并返回一个增强后的 Provider，
// 可用于注入缓存、限流、重试、可观测性等横切关注点。
//
// 使用示例:
//
//	provider := openai.New("key")
//	enhanced := llm.Chain(provider,
//	    llm.WithRetry(3, time.Second),
//	    llm.WithRateLimit(10),
//	    llm.WithTimeout(30 * time.Second),
//	    llm.WithCallback(myCallback),
//	)
type Middleware func(Provider) Provider

// Chain 将多个中间件应用到 Provider 上
//
// 中间件按顺序应用，第一个中间件在最外层。
// 例如 Chain(p, A, B, C) 的调用顺序为: A → B → C → Provider
//
// 参数:
//   - provider: 原始 Provider
//   - middlewares: 中间件列表
//
// 返回:
//   - Provider: 增强后的 Provider
func Chain(provider Provider, middlewares ...Middleware) Provider {
	for i := len(middlewares) - 1; i >= 0; i-- {
		provider = middlewares[i](provider)
	}
	return provider
}

// ============== 重试中间件 ==============

// WithRetry 创建重试中间件
//
// 当请求失败时自动重试，使用指数退避策略。
// 退避时间为 backoff * 2^attempt，上限为 30 秒。
// 内部委托给 toolkit/util/retry.DoWithContext。
//
// 参数:
//   - maxRetries: 最大重试次数（不含首次请求）
//   - backoff: 初始退避时间
func WithRetry(maxRetries int, backoff time.Duration) Middleware {
	return func(next Provider) Provider {
		return &retryProvider{
			inner:      next,
			maxRetries: maxRetries,
			backoff:    backoff,
		}
	}
}

type retryProvider struct {
	inner      Provider
	maxRetries int
	backoff    time.Duration
}

func (p *retryProvider) Name() string { return p.inner.Name() }
func (p *retryProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *retryProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *retryProvider) retryOpts() []retry.Option {
	return []retry.Option{
		retry.Attempts(p.maxRetries + 1), // maxRetries + 1 = total attempts
		retry.Delay(p.backoff),
		retry.Multiplier(2),
		retry.MaxDelay(30 * time.Second),
		retry.RetryIf(func(err error) bool {
			return isRetryableError(err)
		}),
	}
}

func (p *retryProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	if OperationSafetyFromContext(ctx) == OperationSafetyNonIdempotent {
		return p.inner.Complete(ctx, req)
	}
	var resp *CompletionResponse
	err := retry.DoWithContext(ctx, func() error {
		var e error
		resp, e = p.inner.Complete(ctx, req)
		return e
	}, p.retryOpts()...)
	return resp, err
}

func (p *retryProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	if OperationSafetyFromContext(ctx) == OperationSafetyNonIdempotent {
		return p.inner.Stream(ctx, req)
	}
	var stream *streamx.Stream
	err := retry.DoWithContext(ctx, func() error {
		var e error
		stream, e = p.inner.Stream(ctx, req)
		return e
	}, p.retryOpts()...)
	return stream, err
}

// ============== 限流中间件 ==============

// WithRateLimit 创建限流中间件
//
// 使用令牌桶算法限制每秒请求数。
// 当请求超过限制时会等待直到有可用令牌或上下文取消。
//
// 参数:
//   - rps: 每秒允许的最大请求数
func WithRateLimit(rps float64) Middleware {
	return func(next Provider) Provider {
		return &rateLimitProvider{
			inner:    next,
			rps:      rps,
			tokens:   rps,
			lastTime: time.Now(),
		}
	}
}

type rateLimitProvider struct {
	inner    Provider
	rps      float64
	tokens   float64
	lastTime time.Time
	mu       sync.Mutex
}

func (p *rateLimitProvider) Name() string { return p.inner.Name() }
func (p *rateLimitProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *rateLimitProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *rateLimitProvider) wait(ctx context.Context) error {
	p.mu.Lock()
	now := time.Now()
	elapsed := now.Sub(p.lastTime).Seconds()
	p.tokens += elapsed * p.rps
	if p.tokens > p.rps {
		p.tokens = p.rps
	}
	p.lastTime = now

	if p.tokens >= 1 {
		p.tokens--
		p.mu.Unlock()
		return nil
	}

	// 计算需要等待的时间
	waitTime := time.Duration((1 - p.tokens) / p.rps * float64(time.Second))
	p.tokens = 0
	p.lastTime = now.Add(waitTime)
	p.mu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(waitTime):
		return nil
	}
}

func (p *rateLimitProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	if err := p.wait(ctx); err != nil {
		return nil, err
	}
	return p.inner.Complete(ctx, req)
}

func (p *rateLimitProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	if err := p.wait(ctx); err != nil {
		return nil, err
	}
	return p.inner.Stream(ctx, req)
}

// ============== 超时中间件 ==============

// WithTimeout 创建超时中间件
//
// 为每个请求设置超时时间。如果请求在超时前未完成，
// 将返回 context.DeadlineExceeded 错误。
//
// 参数:
//   - timeout: 请求超时时间
func WithTimeout(timeout time.Duration) Middleware {
	return func(next Provider) Provider {
		return &timeoutProvider{
			inner:   next,
			timeout: timeout,
		}
	}
}

type timeoutProvider struct {
	inner   Provider
	timeout time.Duration
}

func (p *timeoutProvider) Name() string { return p.inner.Name() }
func (p *timeoutProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *timeoutProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *timeoutProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	return p.inner.Complete(ctx, req)
}

func (p *timeoutProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	// 流式请求不使用 WithTimeout，因为：
	// 1. 流的生命周期由调用者控制，不应被中间件强制超时
	// 2. WithTimeout 创建的 cancel 函数无法在流关闭时调用，会导致 context 泄漏
	// 如果需要流超时控制，调用者应在自己的 context 中设置
	return p.inner.Stream(ctx, req)
}

// ============== 回调中间件 ==============

// WithCallback 创建回调中间件
//
// 在每次 LLM 请求的开始和结束时触发回调，
// 用于实现可观测性（追踪、日志、指标等）。
//
// 参数:
//   - callback: 回调接口实现
func WithCallback(callback Callback) Middleware {
	return func(next Provider) Provider {
		return &callbackProvider{
			inner:    next,
			callback: callback,
		}
	}
}

type callbackProvider struct {
	inner    Provider
	callback Callback
}

func (p *callbackProvider) Name() string { return p.inner.Name() }
func (p *callbackProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *callbackProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *callbackProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	// 使用栈分配的事件结构体，避免堆逃逸
	startEvent := CallbackStartEvent{
		Provider: p.inner.Name(),
		Request:  &req,
	}
	p.callback.OnStart(ctx, &startEvent)

	start := time.Now()
	resp, err := p.inner.Complete(ctx, req)

	endEvent := CallbackEndEvent{
		Provider:   startEvent.Provider,
		Request:    &req,
		Response:   resp,
		Error:      err,
		DurationMs: time.Since(start).Milliseconds(),
	}
	p.callback.OnEnd(ctx, &endEvent)

	return resp, err
}

func (p *callbackProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	startEvent := CallbackStartEvent{
		Provider: p.inner.Name(),
		Request:  &req,
	}
	p.callback.OnStart(ctx, &startEvent)

	start := time.Now()
	stream, err := p.inner.Stream(ctx, req)

	endEvent := CallbackEndEvent{
		Provider:   startEvent.Provider,
		Request:    &req,
		Error:      err,
		DurationMs: time.Since(start).Milliseconds(),
		Stream:     true,
	}
	p.callback.OnEnd(ctx, &endEvent)

	return stream, err
}

// ============== 缓存中间件 ==============

// Cache 定义 LLM 响应缓存接口
//
// 实现此接口以提供不同的缓存后端（内存、Redis 等）。
//
// 重要：实现必须是并发安全的，因为 WithCache 中间件
// 可能在多个 goroutine 中同时调用 Get/Set。
type Cache interface {
	// Get 获取缓存的响应
	//
	// 参数:
	//   - ctx: 上下文
	//   - key: 缓存键
	//
	// 返回:
	//   - *CompletionResponse: 缓存命中时返回响应，未命中返回 nil
	//   - error: 缓存访问错误
	Get(ctx context.Context, key string) (*CompletionResponse, error)

	// Set 缓存响应
	//
	// 参数:
	//   - ctx: 上下文
	//   - key: 缓存键
	//   - resp: 要缓存的响应
	//
	// 返回:
	//   - error: 缓存写入错误
	Set(ctx context.Context, key string, resp *CompletionResponse) error
}

// CacheKeyFunc 自定义缓存键生成函数
type CacheKeyFunc func(req *CompletionRequest) string

// WithCache 创建缓存中间件
//
// 对相同的请求返回缓存的响应，避免重复调用 LLM。
// 仅缓存非流式请求（Complete），流式请求（Stream）直接透传。
//
// 参数:
//   - cache: 缓存后端实现
//   - keyFn: 缓存键生成函数（可为 nil，使用默认键生成）
func WithCache(cache Cache, keyFn CacheKeyFunc) Middleware {
	return func(next Provider) Provider {
		return &cacheProvider{
			inner: next,
			cache: cache,
			keyFn: keyFn,
		}
	}
}

type cacheProvider struct {
	inner Provider
	cache Cache
	keyFn CacheKeyFunc
	// sf 复用 toolkit/lang/syncx.Singleflight 去重并发缓存未命中请求，
	// 零值即可使用（内部懒加载 map），语义与 golang.org/x/sync/singleflight 等价。
	sf syncx.Singleflight
}

func (p *cacheProvider) Name() string { return p.inner.Name() }
func (p *cacheProvider) Models() []ModelInfo {
	return p.inner.Models()
}
func (p *cacheProvider) CountTokens(messages []Message) (int, error) {
	return p.inner.CountTokens(messages)
}

func (p *cacheProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	if hasBeforeSendHook(ctx) {
		// A durable physical-call owner must observe the actual shared
		// transport boundary. Serving a semantic cache hit would skip its
		// before-send CAS and make a cached result indistinguishable from a
		// provider POST.
		return p.inner.Complete(ctx, req)
	}
	// 生成缓存键；空键表示请求不可缓存（如序列化失败），整体跳过缓存读写
	var key string
	if p.keyFn != nil {
		key = p.keyFn(&req)
	} else {
		key = defaultCacheKey(&req)
	}
	if key == "" {
		return p.inner.Complete(ctx, req)
	}

	// 查找缓存
	if cached, err := p.cache.Get(ctx, key); err == nil && cached != nil {
		return cached, nil
	}

	// Use singleflight to deduplicate concurrent cache-miss requests
	// with the same key, preventing thundering herd on the backend.
	v, err := p.sf.Do(key, func() (any, error) {
		// Double-check cache inside singleflight in case another
		// goroutine populated it between our first check and entering Do.
		if cached, err := p.cache.Get(ctx, key); err == nil && cached != nil {
			return cached, nil
		}

		resp, err := p.inner.Complete(ctx, req)
		if err != nil {
			return nil, err
		}

		// 写入缓存（忽略缓存写入错误）
		_ = p.cache.Set(ctx, key, resp)

		return resp, nil
	})
	if err != nil {
		return nil, err
	}

	return v.(*CompletionResponse), nil
}

func (p *cacheProvider) Stream(ctx context.Context, req CompletionRequest) (*streamx.Stream, error) {
	// 流式请求不走缓存
	return p.inner.Stream(ctx, req)
}

// defaultCacheKey 默认的缓存键生成
//
// 使用完整请求及控制平面策略作用域的稳定 JSON 表示，避免遗漏
// MultiContent、Metadata、ReasoningPolicyScope 等会影响上游语义的字段。
// 序列化失败时返回空串标记请求不可缓存：不能落回 %#v 伪键，
// 其中的指针地址会让同一请求每次生成不同键，缓存永不命中且条目堆积。
func defaultCacheKey(req *CompletionRequest) string {
	var scope ReasoningPolicyScope
	if req != nil {
		scope = req.ReasoningPolicyScope
	}
	cacheIdentity := struct {
		*CompletionRequest
		ReasoningPolicyScope ReasoningPolicyScope `json:"reasoning_policy_scope,omitempty"`
	}{
		CompletionRequest:    req,
		ReasoningPolicyScope: scope,
	}
	data, err := json.Marshal(cacheIdentity)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
