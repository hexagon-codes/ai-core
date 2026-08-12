// Package router provides a multi-provider LLM router for intelligent request routing.
package router

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic" //nolint:depguard // used by atomic.Int64 in Router struct
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/toolkit/util/retry"
)

// Strategy 路由策略
type Strategy string

const (
	// StrategyRoundRobin 轮询策略
	StrategyRoundRobin Strategy = "round_robin"
	// StrategyRandom 随机策略
	StrategyRandom Strategy = "random"
	// StrategyLeastLatency 最低延迟策略
	StrategyLeastLatency Strategy = "least_latency"
	// StrategyLeastCost 最低成本策略
	StrategyLeastCost Strategy = "least_cost"
	// StrategyWeighted 加权策略
	StrategyWeighted Strategy = "weighted"
	// StrategyFallback 降级策略
	StrategyFallback Strategy = "fallback"
	// StrategyModelMatch 模型匹配策略
	StrategyModelMatch Strategy = "model_match"
)

// Router 多 Provider 路由器
type Router struct {
	providers    map[string]llm.Provider
	providerList []string
	weights      map[string]int
	modelMap     map[string]string // model -> provider
	strategy     Strategy
	fallback     string
	healthCheck  bool
	healthy      map[string]bool
	latencies    map[string]time.Duration
	mu           sync.RWMutex
	rrIndex      atomic.Int64
}

// Option 是 Router 的配置选项
type Option func(*Router)

// WithStrategy 设置路由策略
func WithStrategy(strategy Strategy) Option {
	return func(r *Router) {
		r.strategy = strategy
	}
}

// WithFallback 设置降级 Provider
func WithFallback(provider string) Option {
	return func(r *Router) {
		r.fallback = provider
	}
}

// WithWeights 设置 Provider 权重
func WithWeights(weights map[string]int) Option {
	return func(r *Router) {
		r.weights = weights
	}
}

// WithModelMap 设置模型到 Provider 的映射
func WithModelMap(modelMap map[string]string) Option {
	return func(r *Router) {
		r.modelMap = modelMap
	}
}

// WithHealthCheck 启用健康检查
func WithHealthCheck(enabled bool) Option {
	return func(r *Router) {
		r.healthCheck = enabled
	}
}

// New 创建路由器
func New(opts ...Option) *Router {
	r := &Router{
		providers:    make(map[string]llm.Provider),
		providerList: make([]string, 0),
		weights:      make(map[string]int),
		modelMap:     make(map[string]string),
		strategy:     StrategyRoundRobin,
		healthy:      make(map[string]bool),
		latencies:    make(map[string]time.Duration),
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// Register 注册 Provider
func (r *Router) Register(name string, provider llm.Provider) *Router {
	r.mu.Lock()
	defer r.mu.Unlock()

	// 检查是否已注册，避免 providerList 重复
	if _, exists := r.providers[name]; !exists {
		r.providerList = append(r.providerList, name)
	}

	r.providers[name] = provider
	r.healthy[name] = true

	// 默认权重为 1
	if _, ok := r.weights[name]; !ok {
		r.weights[name] = 1
	}

	// 自动映射模型
	for _, model := range provider.Models() {
		if _, exists := r.modelMap[model.ID]; !exists {
			r.modelMap[model.ID] = name
		}
	}

	return r
}

// Unregister 注销 Provider
func (r *Router) Unregister(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.providers, name)
	delete(r.healthy, name)
	delete(r.latencies, name)
	delete(r.weights, name)

	// 从列表中移除
	newList := make([]string, 0, len(r.providerList)-1)
	for _, n := range r.providerList {
		if n != name {
			newList = append(newList, n)
		}
	}
	r.providerList = newList

	// 清理 modelMap
	for model, prov := range r.modelMap {
		if prov == name {
			delete(r.modelMap, model)
		}
	}
}

// Name 返回路由器名称
func (r *Router) Name() string {
	return "router"
}

// Complete 执行补全请求
func (r *Router) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	provider, err := r.selectProvider(req)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	resp, err := provider.Complete(ctx, req)
	r.recordLatency(r.getProviderName(provider), time.Since(start))

	if err != nil {
		if llm.OperationSafetyFromContext(ctx) == llm.OperationSafetyNonIdempotent {
			return nil, err
		}
		// 尝试降级
		if fallbackProvider := r.getFallbackProvider(); fallbackProvider != nil {
			fbStart := time.Now()
			fbResp, fbErr := fallbackProvider.Complete(ctx, req)
			r.recordLatency(r.getProviderName(fallbackProvider), time.Since(fbStart))
			return fbResp, fbErr
		}
		return nil, err
	}

	return resp, nil
}

// Stream 执行流式补全请求
func (r *Router) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	provider, err := r.selectProvider(req)
	if err != nil {
		return nil, err
	}

	stream, err := provider.Stream(ctx, req)
	if err != nil {
		if llm.OperationSafetyFromContext(ctx) == llm.OperationSafetyNonIdempotent {
			return nil, err
		}
		// 尝试降级
		if fallbackProvider := r.getFallbackProvider(); fallbackProvider != nil {
			return fallbackProvider.Stream(ctx, req)
		}
		return nil, err
	}

	return stream, nil
}

// getFallbackProvider 安全获取降级 Provider
func (r *Router) getFallbackProvider() llm.Provider {
	if r.fallback == "" {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if p, ok := r.providers[r.fallback]; ok {
		return p
	}
	return nil
}

// Models 返回所有 Provider 的模型列表
func (r *Router) Models() []llm.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var models []llm.ModelInfo
	seen := make(map[string]bool)

	for _, provider := range r.providers {
		for _, model := range provider.Models() {
			if !seen[model.ID] {
				seen[model.ID] = true
				models = append(models, model)
			}
		}
	}

	return models
}

// CountTokens 计算 Token 数量
func (r *Router) CountTokens(messages []llm.Message) (int, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.providerList) == 0 {
		return 0, errors.New("no providers registered")
	}

	// 使用第一个 Provider 计算
	provider := r.providers[r.providerList[0]]
	return provider.CountTokens(messages)
}

// selectProvider 根据策略选择 Provider
func (r *Router) selectProvider(req llm.CompletionRequest) (llm.Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.providerList) == 0 {
		return nil, errors.New("no providers registered")
	}

	// 模型匹配优先
	if req.Model != "" {
		if provName, ok := r.modelMap[req.Model]; ok {
			if provider, ok := r.providers[provName]; ok {
				if !r.healthCheck || r.healthy[provName] {
					return provider, nil
				}
			}
		}
	}

	// 获取可用的 Provider 列表
	available := r.getHealthyProviders()
	if len(available) == 0 {
		return nil, errors.New("no healthy providers available")
	}

	switch r.strategy {
	case StrategyRoundRobin:
		return r.roundRobinSelect(available)
	case StrategyRandom:
		return r.randomSelect(available)
	case StrategyLeastLatency:
		return r.leastLatencySelect(available)
	case StrategyLeastCost:
		return r.leastCostSelect(available, req.Model)
	case StrategyWeighted:
		return r.weightedSelect(available)
	case StrategyFallback:
		return r.fallbackSelect(available)
	default:
		return r.roundRobinSelect(available)
	}
}

// getHealthyProviders 获取健康的 Provider 列表
func (r *Router) getHealthyProviders() []string {
	if !r.healthCheck {
		return r.providerList
	}

	var healthy []string
	for _, name := range r.providerList {
		if r.healthy[name] {
			healthy = append(healthy, name)
		}
	}
	return healthy
}

// roundRobinSelect 轮询选择
// 使用 atomic 操作递增 rrIndex，无需写锁
func (r *Router) roundRobinSelect(available []string) (llm.Provider, error) {
	n := int64(len(available))
	idx := r.rrIndex.Add(1) - 1
	// 取模选择，防止负值
	if idx < 0 {
		r.rrIndex.Store(0)
		idx = 0
	}
	name := available[idx%n]
	return r.providers[name], nil
}

// randomSelect 随机选择
func (r *Router) randomSelect(available []string) (llm.Provider, error) {
	name := available[rand.Intn(len(available))]
	return r.providers[name], nil
}

// leastLatencySelect 最低延迟选择
func (r *Router) leastLatencySelect(available []string) (llm.Provider, error) {
	var minLatency time.Duration
	var selected string

	for _, name := range available {
		latency := r.latencies[name]
		// 跳过未记录延迟的 Provider
		if latency == 0 {
			continue
		}
		if selected == "" || latency < minLatency {
			minLatency = latency
			selected = name
		}
	}

	// 如果都没有记录延迟，使用第一个
	if selected == "" {
		selected = available[0]
	}
	return r.providers[selected], nil
}

// leastCostSelect 最低成本选择
func (r *Router) leastCostSelect(available []string, model string) (llm.Provider, error) {
	var minCost float64
	var selected string

	for i, name := range available {
		provider := r.providers[name]
		for _, m := range provider.Models() {
			if model != "" && m.ID != model {
				continue
			}
			cost := m.InputCost + m.OutputCost
			if i == 0 || cost < minCost {
				minCost = cost
				selected = name
			}
		}
	}

	if selected == "" {
		selected = available[0]
	}
	return r.providers[selected], nil
}

// weightedSelect 加权选择
func (r *Router) weightedSelect(available []string) (llm.Provider, error) {
	totalWeight := 0
	for _, name := range available {
		totalWeight += r.weights[name]
	}

	// 防止总权重为 0 时 panic
	if totalWeight == 0 {
		return r.providers[available[0]], nil
	}

	target := rand.Intn(totalWeight)
	current := 0

	for _, name := range available {
		current += r.weights[name]
		if current > target {
			return r.providers[name], nil
		}
	}

	return r.providers[available[0]], nil
}

// fallbackSelect 降级选择
func (r *Router) fallbackSelect(available []string) (llm.Provider, error) {
	// 按注册顺序尝试
	for _, name := range r.providerList {
		if r.healthy[name] {
			return r.providers[name], nil
		}
	}
	return nil, errors.New("no healthy providers")
}

// recordLatency 记录延迟
func (r *Router) recordLatency(name string, latency time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// 使用指数移动平均
	if old, ok := r.latencies[name]; ok {
		r.latencies[name] = time.Duration(float64(old)*0.7 + float64(latency)*0.3)
	} else {
		r.latencies[name] = latency
	}
}

// getProviderName 获取 Provider 名称
func (r *Router) getProviderName(provider llm.Provider) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for name, p := range r.providers {
		if p == provider {
			return name
		}
	}
	return ""
}

// SetHealthy 设置 Provider 健康状态
func (r *Router) SetHealthy(name string, healthy bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.healthy[name] = healthy
}

// GetStats 获取路由统计
func (r *Router) GetStats() RouterStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := RouterStats{
		ProviderCount: len(r.providerList),
		Providers:     make(map[string]ProviderStats),
	}

	for _, name := range r.providerList {
		stats.Providers[name] = ProviderStats{
			Name:    name,
			Healthy: r.healthy[name],
			Latency: r.latencies[name],
			Weight:  r.weights[name],
		}
	}

	return stats
}

// RouterStats 路由器统计
type RouterStats struct {
	ProviderCount int                      `json:"provider_count"`
	Providers     map[string]ProviderStats `json:"providers"`
}

// ProviderStats Provider 统计
type ProviderStats struct {
	Name    string        `json:"name"`
	Healthy bool          `json:"healthy"`
	Latency time.Duration `json:"latency"`
	Weight  int           `json:"weight"`
}

// HealthChecker 健康检查器
type HealthChecker struct {
	router   *Router
	interval time.Duration
	stopCh   chan struct{}
	stopOnce sync.Once
}

// NewHealthChecker 创建健康检查器
func NewHealthChecker(router *Router, interval time.Duration) *HealthChecker {
	return &HealthChecker{
		router:   router,
		interval: interval,
		stopCh:   make(chan struct{}),
	}
}

// Start 启动健康检查
func (h *HealthChecker) Start(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(h.interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-h.stopCh:
				return
			case <-ticker.C:
				h.checkAll(ctx)
			}
		}
	}()
}

// Stop 停止健康检查
// 可以安全地多次调用
func (h *HealthChecker) Stop() {
	h.stopOnce.Do(func() {
		close(h.stopCh)
	})
}

// checkAll 检查所有 Provider
func (h *HealthChecker) checkAll(ctx context.Context) {
	h.router.mu.RLock()
	providers := make(map[string]llm.Provider)
	for k, v := range h.router.providers {
		providers[k] = v
	}
	h.router.mu.RUnlock()

	for name, provider := range providers {
		healthy := h.checkProvider(ctx, provider)
		h.router.SetHealthy(name, healthy)
	}
}

// checkProvider 检查单个 Provider
func (h *HealthChecker) checkProvider(ctx context.Context, provider llm.Provider) bool {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	// 简单的健康检查：尝试计算 token
	_, err := provider.CountTokens([]llm.Message{{Content: "test"}})
	return err == nil
}

// ============== 便捷构建器 ==============

// Builder 路由器构建器
type Builder struct {
	router *Router
}

// NewBuilder 创建构建器
func NewBuilder() *Builder {
	return &Builder{
		router: New(),
	}
}

// Add 添加 Provider
func (b *Builder) Add(name string, provider llm.Provider) *Builder {
	b.router.Register(name, provider)
	return b
}

// AddWithWeight 添加带权重的 Provider
func (b *Builder) AddWithWeight(name string, provider llm.Provider, weight int) *Builder {
	b.router.weights[name] = weight
	b.router.Register(name, provider)
	return b
}

// Strategy 设置策略
func (b *Builder) Strategy(strategy Strategy) *Builder {
	b.router.strategy = strategy
	return b
}

// Fallback 设置降级 Provider
func (b *Builder) Fallback(name string) *Builder {
	b.router.fallback = name
	return b
}

// MapModel 映射模型到 Provider
func (b *Builder) MapModel(model, provider string) *Builder {
	b.router.modelMap[model] = provider
	return b
}

// EnableHealthCheck 启用健康检查
func (b *Builder) EnableHealthCheck() *Builder {
	b.router.healthCheck = true
	return b
}

// Build 构建路由器
func (b *Builder) Build() *Router {
	return b.router
}

// 确保实现了 Provider 接口
var _ llm.Provider = (*Router)(nil)

// CreateDefaultRouter 创建包含常用 Provider 的默认路由器
func CreateDefaultRouter(providers map[string]llm.Provider) *Router {
	builder := NewBuilder().
		Strategy(StrategyModelMatch).
		EnableHealthCheck()

	// 排序 provider 名称以确保确定性行为
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		builder.Add(name, providers[name])
	}

	// 设置降级顺序：使用排序后的第一个 Provider
	if len(names) > 1 {
		builder.Fallback(names[0])
	}

	return builder.Build()
}

// RouteByCapability 根据能力路由
func RouteByCapability(router *Router, capability string) llm.Provider {
	router.mu.RLock()
	defer router.mu.RUnlock()

	for name, provider := range router.providers {
		if !router.healthy[name] {
			continue
		}
		for _, model := range provider.Models() {
			if model.HasFeature(capability) {
				return provider
			}
		}
	}
	return nil
}

// MultiProviderError 多 Provider 错误
type MultiProviderError struct {
	Errors map[string]error
}

func (e *MultiProviderError) Error() string {
	return fmt.Sprintf("all providers failed: %v", e.Errors)
}

// ExecuteWithRetry 带重试执行
//
// 每次尝试前检查 ctx，并在重试间使用指数退避（退避同样受 ctx 控制）；
// ctx 取消/超时立即返回，不再无谓地继续调用 Provider。
// 收集各次尝试的错误，全部失败后返回 MultiProviderError。
func ExecuteWithRetry(ctx context.Context, router *Router, req llm.CompletionRequest, maxRetries int) (*llm.CompletionResponse, error) {
	if maxRetries < 1 {
		maxRetries = 1
	}
	if llm.OperationSafetyFromContext(ctx) == llm.OperationSafetyNonIdempotent {
		maxRetries = 1
	}

	errs := make(map[string]error)
	var resp *llm.CompletionResponse
	attempt := 0

	err := retry.DoWithContext(ctx, func() error {
		attempt++
		r, e := router.Complete(ctx, req)
		if e != nil {
			errs[fmt.Sprintf("attempt_%d", attempt)] = e
			return e
		}
		resp = r
		return nil
	},
		retry.Attempts(maxRetries),
		retry.Delay(200*time.Millisecond),
		retry.Multiplier(2),
		retry.DelayType(retry.ExponentialBackoff),
		retry.MaxDelay(10*time.Second),
	)
	if err == nil {
		return resp, nil
	}

	// ctx 取消/超时：透传，便于调用方区分（区别于 Provider 业务错误）。
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}

	return nil, &MultiProviderError{Errors: errs}
}
