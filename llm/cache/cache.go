// Package cache 提供 LLM 响应缓存实现
//
// 本包实现了 llm.Cache 接口，提供内存缓存能力，
// 用于避免对相同请求重复调用 LLM，节省成本和延迟。
//
// 使用示例:
//
//	// 创建内存缓存
//	c := cache.NewMemoryCache(
//	    cache.WithMaxEntries(1000),
//	    cache.WithTTL(time.Hour),
//	)
//
//	// 通过中间件应用到 Provider
//	provider = llm.Chain(provider, llm.WithCache(c, nil))
package cache

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/toolkit/cache/local"
)

// ============== 内存缓存实现 ==============

// MemoryCache 基于内存的 LLM 响应缓存
//
// 存储、TTL 过期与 LRU 淘汰统一委托给 toolkit/cache/local 的本地缓存
// （通过 NewCacheNoCleanup 构造，不启动后台清理 goroutine，避免泄漏），
// 本类型仅保留 llm.Cache 接口契约所需的薄封装层：
//   - maxEntries ≤ 0 时禁用缓存：在封装层直接拦截，不委托给底层；
//   - 命中/未命中统计：由本封装层自行累计（底层不暴露命中率统计）。
//
// 线程安全，适用于单进程场景。
//
// 特性:
//   - LRU 淘汰：确定性淘汰（底层以 (accessedAt, seq) 二元组保证全序）
//   - TTL 过期：条目在指定时间后自动失效（惰性清理）；TTL ≤ 0 表示不过期
//   - 命中统计：提供命中率等缓存统计信息
//   - 零外部依赖（除 toolkit 基础库）：纯内存实现
type MemoryCache struct {
	// store 为底层本地缓存，负责实际的键值存储、TTL 过期与 LRU 淘汰。
	// 仅当 maxEntries > 0 时才会被构造与使用。
	store *local.Cache

	// maxEntries ≤ 0 表示禁用缓存：此时 store 为 nil，所有 Set 不存储、
	// 所有 Get 直接计为未命中。保留该字段用于在封装层拦截。
	maxEntries int

	// ttl 为条目过期时间，透传给底层 Set。ttl ≤ 0 表示不按 TTL 过期。
	ttl time.Duration

	// 命中/未命中统计，由封装层自行累计（底层不提供该统计）。
	// 使用原子计数，避免与底层读写共用一把锁、并保持统计读写的并发安全。
	hits   atomic.Int64
	misses atomic.Int64

	// mu 仅保护 Clear 与统计快照之间的一致性边界。
	mu sync.Mutex
}

// cacheEntry 缓存条目（封装层内部使用）
//
// 底层 toolkit 缓存以裸 any 形式存储该结构指针，避免对
// *llm.CompletionResponse 做序列化，保持指针标识一致。
type cacheEntry struct {
	response *llm.CompletionResponse
}

// MemoryCacheOption 内存缓存配置选项
type MemoryCacheOption func(*MemoryCache)

// WithMaxEntries 设置最大缓存条目数
//
// 超过此数量时使用 LRU 策略淘汰最旧的条目。
// 值 ≤ 0 时禁用缓存（所有 Set 操作不存储）。
// 默认值: 1000
func WithMaxEntries(n int) MemoryCacheOption {
	return func(c *MemoryCache) {
		c.maxEntries = n
	}
}

// WithTTL 设置缓存过期时间
//
// 超过此时间的条目会在访问时自动失效（惰性清理）。
// 默认值: 1 小时
func WithTTL(ttl time.Duration) MemoryCacheOption {
	return func(c *MemoryCache) {
		c.ttl = ttl
	}
}

// NewMemoryCache 创建内存缓存
//
// 默认配置:
//   - 最大条目数: 1000
//   - TTL: 1 小时
func NewMemoryCache(opts ...MemoryCacheOption) *MemoryCache {
	c := &MemoryCache{
		maxEntries: 1000,
		ttl:        time.Hour,
	}
	for _, opt := range opts {
		opt(c)
	}

	// maxEntries > 0 时才构造底层存储；≤ 0 时禁用缓存（store 保持 nil）。
	// 使用 NewCacheNoCleanup 避免底层启动后台清理 goroutine，防止泄漏；
	// 过期条目依赖惰性清理（读取时就地删除）。
	if c.maxEntries > 0 {
		c.store = local.NewCacheNoCleanup(c.maxEntries)
	}
	return c
}

// Get 获取缓存的响应
//
// 如果缓存命中且未过期，返回缓存的响应并更新 LRU 顺序。
// 如果缓存未命中或已过期，返回 nil。
func (c *MemoryCache) Get(_ context.Context, key string) (*llm.CompletionResponse, error) {
	// 禁用缓存时直接计为未命中
	if c.store == nil {
		c.misses.Add(1)
		return nil, nil
	}

	v, ok := c.store.Get(key)
	if !ok {
		c.misses.Add(1)
		return nil, nil
	}

	entry, ok := v.(*cacheEntry)
	if !ok {
		// 理论上不会发生：本封装层只写入 *cacheEntry。
		// 防御性处理为未命中，避免 panic。
		c.misses.Add(1)
		return nil, nil
	}

	c.hits.Add(1)
	return entry.response, nil
}

// Set 缓存响应
//
// 如果缓存已满（超过 MaxEntries），底层按 LRU 淘汰最久未使用的条目。
// 当 MaxEntries ≤ 0 时，不缓存任何条目。
func (c *MemoryCache) Set(_ context.Context, key string, resp *llm.CompletionResponse) error {
	// maxEntries ≤ 0 时禁用缓存
	if c.store == nil {
		return nil
	}

	// 以裸 any 形式存放 *cacheEntry，保持指针标识，不做序列化。
	// ttl ≤ 0 时底层语义为"永不过期"，与原实现 TTL=0 零值语义一致。
	c.store.Set(key, &cacheEntry{response: resp}, c.ttl)
	return nil
}

// Stats 返回缓存统计信息
func (c *MemoryCache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	hits := c.hits.Load()
	misses := c.misses.Load()
	total := hits + misses
	var hitRate float64
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}

	var size int
	if c.store != nil {
		size = c.store.Len()
	}

	return CacheStats{
		Size:    size,
		Hits:    hits,
		Misses:  misses,
		HitRate: hitRate,
	}
}

// Clear 清空缓存
func (c *MemoryCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.store != nil {
		c.store.Clear()
	}
	c.hits.Store(0)
	c.misses.Store(0)
}

// CacheStats 缓存统计信息
type CacheStats struct {
	// Size 当前缓存条目数
	Size int

	// Hits 缓存命中次数
	Hits int64

	// Misses 缓存未命中次数
	Misses int64

	// HitRate 缓存命中率 (0-1)
	HitRate float64
}

// 确保实现了 llm.Cache 接口
var _ llm.Cache = (*MemoryCache)(nil)
