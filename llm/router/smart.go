// Package router 提供多 Provider LLM 路由功能
//
// SmartRouter 是智能路由器，根据任务类型、复杂度和约束条件自动选择最优模型。
// 它扩展了基础 Router，增加了任务感知的智能路由能力。
package router

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

// ============== 任务类型定义 ==============

// TaskType 任务类型
// 用于描述用户请求的任务性质，帮助路由器选择最适合的模型
type TaskType string

const (
	// TaskTypeChat 普通对话任务
	// 适用于日常聊天、问答等场景
	TaskTypeChat TaskType = "chat"

	// TaskTypeReasoning 推理任务
	// 适用于需要逻辑推理、问题分析的场景
	TaskTypeReasoning TaskType = "reasoning"

	// TaskTypeCoding 编程任务
	// 适用于代码生成、代码审查、调试等场景
	TaskTypeCoding TaskType = "coding"

	// TaskTypeAnalysis 数据分析任务
	// 适用于数据分析、报告生成等场景
	TaskTypeAnalysis TaskType = "analysis"

	// TaskTypeSummarization 摘要任务
	// 适用于文本摘要、内容提炼等场景
	TaskTypeSummarization TaskType = "summarization"

	// TaskTypeExtraction 信息提取任务
	// 适用于结构化数据提取、实体识别等场景
	TaskTypeExtraction TaskType = "extraction"

	// TaskTypeCreative 创意写作任务
	// 适用于创意写作、内容创作等场景
	TaskTypeCreative TaskType = "creative"

	// TaskTypeTranslation 翻译任务
	// 适用于多语言翻译场景
	TaskTypeTranslation TaskType = "translation"

	// TaskTypeMath 数学计算任务
	// 适用于数学问题求解、公式推导等场景
	TaskTypeMath TaskType = "math"

	// TaskTypeVision 视觉理解任务
	// 适用于图像理解、图表分析等场景
	TaskTypeVision TaskType = "vision"
)

// AllTaskTypes 返回所有任务类型
func AllTaskTypes() []TaskType {
	return []TaskType{
		TaskTypeChat,
		TaskTypeReasoning,
		TaskTypeCoding,
		TaskTypeAnalysis,
		TaskTypeSummarization,
		TaskTypeExtraction,
		TaskTypeCreative,
		TaskTypeTranslation,
		TaskTypeMath,
		TaskTypeVision,
	}
}

// ============== 任务复杂度定义 ==============

// TaskComplexity 任务复杂度
// 用于评估任务的难度级别，影响模型选择
type TaskComplexity string

const (
	// ComplexitySimple 简单任务
	// 如：简单问答、格式转换等
	ComplexitySimple TaskComplexity = "simple"

	// ComplexityMedium 中等任务
	// 如：常规编程、文本分析等
	ComplexityMedium TaskComplexity = "medium"

	// ComplexityComplex 复杂任务
	// 如：架构设计、深度分析等
	ComplexityComplex TaskComplexity = "complex"

	// ComplexityExpert 专家级任务
	// 如：前沿研究、高难度问题等
	ComplexityExpert TaskComplexity = "expert"
)

// ComplexityScore 返回复杂度对应的数值得分
// 用于计算路由评分
func (c TaskComplexity) Score() float64 {
	switch c {
	case ComplexitySimple:
		return 0.25
	case ComplexityMedium:
		return 0.5
	case ComplexityComplex:
		return 0.75
	case ComplexityExpert:
		return 1.0
	default:
		return 0.5
	}
}

// ============== 路由上下文定义 ==============

// RoutingContext 路由上下文
// 包含任务特征和约束条件，用于指导智能路由决策
type RoutingContext struct {
	// TaskType 任务类型
	TaskType TaskType

	// Complexity 任务复杂度
	Complexity TaskComplexity

	// RequiredCapabilities 必需的能力列表
	// 如 ["vision", "functions", "json_mode"]
	RequiredCapabilities []string

	// Constraints 路由约束条件
	Constraints RoutingConstraints

	// Hints 额外提示信息
	// 可用于传递自定义路由信息
	Hints map[string]any

	// Priority 优先级策略
	// 决定在多个因素间如何权衡
	Priority RoutingPriority
}

// RoutingConstraints 路由约束条件
// 用于限制模型选择范围
type RoutingConstraints struct {
	// MaxLatencyMs 最大允许延迟（毫秒）
	// 0 表示不限制
	MaxLatencyMs int

	// MaxCostPerRequest 单次请求最大成本（美元）
	// 0 表示不限制
	MaxCostPerRequest float64

	// MaxInputTokens 最大输入 Token 数
	// 0 表示不限制
	MaxInputTokens int

	// PreferredProviders 首选 Provider 列表
	// 优先从这些 Provider 中选择
	PreferredProviders []string

	// ExcludedProviders 排除的 Provider 列表
	ExcludedProviders []string

	// PreferredModels 首选模型列表
	PreferredModels []string

	// ExcludedModels 排除的模型列表
	ExcludedModels []string

	// RequireStreaming 是否要求流式输出
	RequireStreaming bool

	// RequireVision 是否要求视觉能力
	RequireVision bool

	// RequireFunctionCalling 是否要求函数调用能力
	RequireFunctionCalling bool

	// RequireJSONMode 是否要求 JSON 模式
	RequireJSONMode bool
}

// RoutingPriority 路由优先级策略
type RoutingPriority string

const (
	// PriorityQuality 质量优先
	// 选择任务得分最高的模型
	PriorityQuality RoutingPriority = "quality"

	// PriorityCost 成本优先
	// 选择成本最低且满足要求的模型
	PriorityCost RoutingPriority = "cost"

	// PriorityLatency 延迟优先
	// 选择响应最快的模型
	PriorityLatency RoutingPriority = "latency"

	// PriorityBalanced 均衡策略
	// 在质量、成本、延迟间取得平衡
	PriorityBalanced RoutingPriority = "balanced"
)

// NewRoutingContext 创建路由上下文
func NewRoutingContext(taskType TaskType, complexity TaskComplexity) *RoutingContext {
	return &RoutingContext{
		TaskType:   taskType,
		Complexity: complexity,
		Priority:   PriorityBalanced,
		Hints:      make(map[string]any),
	}
}

// WithCapabilities 设置必需能力
func (c *RoutingContext) WithCapabilities(caps ...string) *RoutingContext {
	c.RequiredCapabilities = append(c.RequiredCapabilities, caps...)
	return c
}

// WithMaxLatency 设置最大延迟
func (c *RoutingContext) WithMaxLatency(ms int) *RoutingContext {
	c.Constraints.MaxLatencyMs = ms
	return c
}

// WithMaxCost 设置最大成本
func (c *RoutingContext) WithMaxCost(cost float64) *RoutingContext {
	c.Constraints.MaxCostPerRequest = cost
	return c
}

// WithPriority 设置优先级策略
func (c *RoutingContext) WithPriority(priority RoutingPriority) *RoutingContext {
	c.Priority = priority
	return c
}

// WithPreferredProviders 设置首选 Provider
func (c *RoutingContext) WithPreferredProviders(providers ...string) *RoutingContext {
	c.Constraints.PreferredProviders = append(c.Constraints.PreferredProviders, providers...)
	return c
}

// RequireVision 设置视觉能力要求
func (c *RoutingContext) RequireVision() *RoutingContext {
	c.Constraints.RequireVision = true
	c.RequiredCapabilities = append(c.RequiredCapabilities, llm.FeatureVision)
	return c
}

// RequireFunctions 设置函数调用能力要求
func (c *RoutingContext) RequireFunctions() *RoutingContext {
	c.Constraints.RequireFunctionCalling = true
	c.RequiredCapabilities = append(c.RequiredCapabilities, llm.FeatureFunctions)
	return c
}

// ============== 路由决策定义 ==============

// RoutingDecision 路由决策结果
// 包含选中的模型及决策依据
type RoutingDecision struct {
	// Provider 选中的 Provider
	Provider llm.Provider

	// ProviderName Provider 名称
	ProviderName string

	// ModelID 选中的模型 ID
	ModelID string

	// ModelInfo 模型详细信息
	ModelInfo llm.ModelInfo

	// Reason 选择原因说明
	Reason string

	// Score 综合评分 (0-1)
	Score float64

	// Scores 各维度得分详情
	Scores ScoreBreakdown

	// EstimatedCost 预估成本（美元）
	EstimatedCost float64

	// EstimatedLatency 预估延迟（毫秒）
	EstimatedLatency int

	// Alternatives 备选模型列表
	Alternatives []AlternativeModel

	// DecidedAt 决策时间
	DecidedAt time.Time

	// Metadata 额外元数据
	Metadata map[string]any
}

// ScoreBreakdown 评分详情
type ScoreBreakdown struct {
	// TaskScore 任务匹配度得分
	TaskScore float64

	// ComplexityScore 复杂度匹配度得分
	ComplexityScore float64

	// CostScore 成本得分（越低越好）
	CostScore float64

	// LatencyScore 延迟得分（越低越好）
	LatencyScore float64

	// CapabilityScore 能力匹配度得分
	CapabilityScore float64

	// PreferenceScore 偏好得分
	PreferenceScore float64
}

// AlternativeModel 备选模型
type AlternativeModel struct {
	// ProviderName Provider 名称
	ProviderName string

	// ModelID 模型 ID
	ModelID string

	// Score 综合评分
	Score float64

	// Reason 未选择的原因
	Reason string
}

// ============== SmartRouter 实现 ==============

// SmartRouter 智能路由器
// 扩展基础 Router，增加任务感知的智能路由能力
type SmartRouter struct {
	// 继承基础路由器
	*Router

	// classifier 任务分类器
	classifier TaskClassifier

	// modelProfiles 模型能力档案
	modelProfiles map[string]*ModelProfile

	// profilesMu 保护 modelProfiles 的锁
	profilesMu sync.RWMutex

	// routingHistory 路由历史（用于学习优化）
	routingHistory []RoutingRecord

	// historyMu 保护 routingHistory 的锁
	historyMu sync.RWMutex

	// maxHistorySize 最大历史记录数
	maxHistorySize int

	// autoClassify 是否自动分类任务
	autoClassify bool
}

// RoutingRecord 路由记录
// 用于追踪路由决策和结果
type RoutingRecord struct {
	// Decision 路由决策
	Decision *RoutingDecision

	// Context 路由上下文
	Context *RoutingContext

	// Success 是否成功
	Success bool

	// ActualLatency 实际延迟
	ActualLatency time.Duration

	// ActualCost 实际成本
	ActualCost float64

	// Timestamp 记录时间
	Timestamp time.Time
}

// SmartRouterOption SmartRouter 配置选项
type SmartRouterOption func(*SmartRouter)

// WithClassifier 设置任务分类器
func WithClassifier(classifier TaskClassifier) SmartRouterOption {
	return func(r *SmartRouter) {
		r.classifier = classifier
	}
}

// WithAutoClassify 启用自动任务分类
func WithAutoClassify(enabled bool) SmartRouterOption {
	return func(r *SmartRouter) {
		r.autoClassify = enabled
	}
}

// WithMaxHistorySize 设置最大历史记录数
func WithMaxHistorySize(size int) SmartRouterOption {
	return func(r *SmartRouter) {
		r.maxHistorySize = size
	}
}

// WithModelProfiles 设置模型能力档案
func WithModelProfiles(profiles map[string]*ModelProfile) SmartRouterOption {
	return func(r *SmartRouter) {
		r.modelProfiles = profiles
	}
}

// NewSmartRouter 创建智能路由器
func NewSmartRouter(baseRouter *Router, opts ...SmartRouterOption) *SmartRouter {
	r := &SmartRouter{
		Router:         baseRouter,
		classifier:     NewRuleBasedClassifier(),
		modelProfiles:  make(map[string]*ModelProfile),
		routingHistory: make([]RoutingRecord, 0),
		maxHistorySize: 1000,
		autoClassify:   true,
	}

	// 应用配置选项
	for _, opt := range opts {
		opt(r)
	}

	// 初始化默认模型档案
	r.initDefaultProfiles()

	return r
}

// initDefaultProfiles 初始化默认模型能力档案
func (r *SmartRouter) initDefaultProfiles() {
	// 加载所有默认档案
	for modelID, profile := range GetDefaultProfiles() {
		r.modelProfiles[modelID] = profile
	}
}

// RegisterProfile 注册模型能力档案
func (r *SmartRouter) RegisterProfile(modelID string, profile *ModelProfile) {
	r.profilesMu.Lock()
	defer r.profilesMu.Unlock()
	r.modelProfiles[modelID] = profile
}

// GetProfile 获取模型能力档案
func (r *SmartRouter) GetProfile(modelID string) *ModelProfile {
	r.profilesMu.RLock()
	defer r.profilesMu.RUnlock()
	return r.modelProfiles[modelID]
}

// Route 智能路由
// 根据请求内容和路由上下文选择最优模型
func (r *SmartRouter) Route(ctx context.Context, req llm.CompletionRequest, routingCtx *RoutingContext) (*RoutingDecision, error) {
	// 如果没有提供路由上下文，尝试自动分类
	if routingCtx == nil {
		routingCtx = r.autoClassifyRequest(ctx, req)
	}

	// 获取所有候选模型
	candidates := r.getCandidates(routingCtx)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("没有满足条件的候选模型")
	}

	// 成本/延迟约束是硬约束，不能只作为评分项降低权重。
	filtered := candidates[:0]
	for _, c := range candidates {
		if r.meetsHardConstraints(c, routingCtx, req) {
			filtered = append(filtered, c)
		}
	}
	candidates = filtered
	if len(candidates) == 0 {
		return nil, fmt.Errorf("没有满足成本或延迟约束的候选模型")
	}

	// 计算每个候选的得分
	scoredCandidates := make([]scoredCandidate, 0, len(candidates))
	for _, c := range candidates {
		score := r.calculateScore(c, routingCtx, req)
		scoredCandidates = append(scoredCandidates, scoredCandidate{
			candidate: c,
			score:     score,
		})
	}

	// 按得分排序
	sort.Slice(scoredCandidates, func(i, j int) bool {
		return scoredCandidates[i].score.Total > scoredCandidates[j].score.Total
	})

	// 选择得分最高的模型
	best := scoredCandidates[0]

	// 构建备选列表
	alternatives := make([]AlternativeModel, 0, min(3, len(scoredCandidates)-1))
	for i := 1; i < len(scoredCandidates) && i <= 3; i++ {
		alt := scoredCandidates[i]
		alternatives = append(alternatives, AlternativeModel{
			ProviderName: alt.candidate.providerName,
			ModelID:      alt.candidate.modelID,
			Score:        alt.score.Total,
			Reason:       fmt.Sprintf("得分 %.2f，低于最佳模型", alt.score.Total),
		})
	}

	// 构建决策结果
	decision := &RoutingDecision{
		Provider:     best.candidate.provider,
		ProviderName: best.candidate.providerName,
		ModelID:      best.candidate.modelID,
		ModelInfo:    best.candidate.modelInfo,
		Reason:       r.buildDecisionReason(best, routingCtx),
		Score:        best.score.Total,
		Scores: ScoreBreakdown{
			TaskScore:       best.score.TaskScore,
			ComplexityScore: best.score.ComplexityScore,
			CostScore:       best.score.CostScore,
			LatencyScore:    best.score.LatencyScore,
			CapabilityScore: best.score.CapabilityScore,
			PreferenceScore: best.score.PreferenceScore,
		},
		EstimatedCost:    r.estimateCost(best.candidate, req),
		EstimatedLatency: r.estimateLatency(best.candidate),
		Alternatives:     alternatives,
		DecidedAt:        time.Now(),
		Metadata:         make(map[string]any),
	}

	return decision, nil
}

// CompleteWithRouting 带路由的补全请求
// 自动选择最优模型并执行请求
func (r *SmartRouter) CompleteWithRouting(ctx context.Context, req llm.CompletionRequest, routingCtx *RoutingContext) (*llm.CompletionResponse, *RoutingDecision, error) {
	// 执行路由决策
	decision, err := r.Route(ctx, req, routingCtx)
	if err != nil {
		return nil, nil, fmt.Errorf("路由决策失败: %w", err)
	}

	// 设置选中的模型
	req.Model = decision.ModelID

	// 执行请求
	start := time.Now()
	resp, err := decision.Provider.Complete(ctx, req)
	elapsed := time.Since(start)

	// 记录路由历史
	r.recordRouting(decision, routingCtx, err == nil, elapsed, resp)

	if err != nil {
		return nil, decision, fmt.Errorf("请求执行失败: %w", err)
	}

	return resp, decision, nil
}

// StreamWithRouting 带路由的流式补全请求
func (r *SmartRouter) StreamWithRouting(ctx context.Context, req llm.CompletionRequest, routingCtx *RoutingContext) (*streamx.Stream, *RoutingDecision, error) {
	// 确保要求流式能力
	if routingCtx != nil {
		routingCtx.Constraints.RequireStreaming = true
		routingCtx.RequiredCapabilities = append(routingCtx.RequiredCapabilities, llm.FeatureStreaming)
	}

	// 执行路由决策
	decision, err := r.Route(ctx, req, routingCtx)
	if err != nil {
		return nil, nil, fmt.Errorf("路由决策失败: %w", err)
	}

	// 设置选中的模型
	req.Model = decision.ModelID

	// 执行流式请求
	stream, err := decision.Provider.Stream(ctx, req)
	if err != nil {
		return nil, decision, fmt.Errorf("流式请求执行失败: %w", err)
	}

	return stream, decision, nil
}

// ============== 内部方法 ==============

// candidate 候选模型
type candidate struct {
	provider     llm.Provider
	providerName string
	modelID      string
	modelInfo    llm.ModelInfo
	profile      *ModelProfile
}

// scoredCandidate 带得分的候选模型
type scoredCandidate struct {
	candidate candidate
	score     candidateScore
}

// candidateScore 候选得分
type candidateScore struct {
	Total           float64
	TaskScore       float64
	ComplexityScore float64
	CostScore       float64
	LatencyScore    float64
	CapabilityScore float64
	PreferenceScore float64
}

// getCandidates 获取所有候选模型
func (r *SmartRouter) getCandidates(routingCtx *RoutingContext) []candidate {
	r.mu.RLock()
	defer r.mu.RUnlock()

	candidates := make([]candidate, 0)

	for name, provider := range r.providers {
		// 检查 Provider 是否被排除
		if r.isProviderExcluded(name, routingCtx) {
			continue
		}

		// 检查健康状态
		if r.healthCheck && !r.healthy[name] {
			continue
		}

		// 遍历 Provider 的所有模型
		for _, model := range provider.Models() {
			// 检查模型是否被排除
			if r.isModelExcluded(model.ID, routingCtx) {
				continue
			}

			// 检查能力要求
			if !r.meetsCapabilityRequirements(model, routingCtx) {
				continue
			}

			// 获取模型档案
			profile := r.getProfileLocked(model.ID)

			candidates = append(candidates, candidate{
				provider:     provider,
				providerName: name,
				modelID:      model.ID,
				modelInfo:    model,
				profile:      profile,
			})
		}
	}

	return candidates
}

// isProviderExcluded 检查 Provider 是否被排除
func (r *SmartRouter) isProviderExcluded(name string, ctx *RoutingContext) bool {
	if ctx == nil || len(ctx.Constraints.ExcludedProviders) == 0 {
		return false
	}
	for _, excluded := range ctx.Constraints.ExcludedProviders {
		if excluded == name {
			return true
		}
	}
	return false
}

// isModelExcluded 检查模型是否被排除
func (r *SmartRouter) isModelExcluded(modelID string, ctx *RoutingContext) bool {
	if ctx == nil || len(ctx.Constraints.ExcludedModels) == 0 {
		return false
	}
	for _, excluded := range ctx.Constraints.ExcludedModels {
		if excluded == modelID {
			return true
		}
	}
	return false
}

// meetsCapabilityRequirements 检查模型是否满足能力要求
func (r *SmartRouter) meetsCapabilityRequirements(model llm.ModelInfo, ctx *RoutingContext) bool {
	if ctx == nil {
		return true
	}

	// 检查必需能力
	for _, cap := range ctx.RequiredCapabilities {
		if !model.HasFeature(cap) {
			return false
		}
	}

	// 检查特定约束
	if ctx.Constraints.RequireVision && !model.HasFeature(llm.FeatureVision) {
		return false
	}
	if ctx.Constraints.RequireFunctionCalling && !model.HasFeature(llm.FeatureFunctions) {
		return false
	}
	if ctx.Constraints.RequireStreaming && !model.HasFeature(llm.FeatureStreaming) {
		return false
	}
	if ctx.Constraints.RequireJSONMode && !model.HasFeature(llm.FeatureJSON) {
		return false
	}

	return true
}

// getProfileLocked 获取模型档案（需要外部持有锁或使用 profilesMu）
func (r *SmartRouter) getProfileLocked(modelID string) *ModelProfile {
	r.profilesMu.RLock()
	defer r.profilesMu.RUnlock()
	return r.modelProfiles[modelID]
}

// meetsHardConstraints 检查候选是否满足显式成本/延迟上限。
// 未知延迟不作为拒绝依据，避免把“尚无历史数据”误判为超限。
func (r *SmartRouter) meetsHardConstraints(c candidate, ctx *RoutingContext, req llm.CompletionRequest) bool {
	if ctx == nil {
		return true
	}
	if ctx.Constraints.MaxCostPerRequest > 0 && r.estimateCost(c, req) > ctx.Constraints.MaxCostPerRequest {
		return false
	}
	if ctx.Constraints.MaxLatencyMs > 0 {
		if latency, known := r.knownLatency(c); known && latency > ctx.Constraints.MaxLatencyMs {
			return false
		}
	}
	return true
}

func (r *SmartRouter) knownLatency(c candidate) (int, bool) {
	if c.profile != nil && c.profile.AverageLatencyMs > 0 {
		return c.profile.AverageLatencyMs, true
	}
	r.mu.RLock()
	latency := r.latencies[c.providerName]
	r.mu.RUnlock()
	if latency > 0 {
		return int(latency.Milliseconds()), true
	}
	return 0, false
}

// calculateScore 计算候选模型的得分
func (r *SmartRouter) calculateScore(c candidate, ctx *RoutingContext, req llm.CompletionRequest) candidateScore {
	score := candidateScore{}

	// 1. 任务匹配度得分 (0-1)
	score.TaskScore = r.calculateTaskScore(c, ctx)

	// 2. 复杂度匹配度得分 (0-1)
	score.ComplexityScore = r.calculateComplexityScore(c, ctx)

	// 3. 成本得分 (0-1，越低越好需要反转)
	score.CostScore = r.calculateCostScore(c, ctx, req)

	// 4. 延迟得分 (0-1，越低越好需要反转)
	score.LatencyScore = r.calculateLatencyScore(c, ctx)

	// 5. 能力匹配度得分 (0-1)
	score.CapabilityScore = r.calculateCapabilityScore(c, ctx)

	// 6. 偏好得分 (0-1)
	score.PreferenceScore = r.calculatePreferenceScore(c, ctx)

	// 根据优先级策略计算总分
	score.Total = r.calculateTotalScore(score, ctx)

	return score
}

// calculateTaskScore 计算任务匹配度得分
func (r *SmartRouter) calculateTaskScore(c candidate, ctx *RoutingContext) float64 {
	if c.profile == nil || ctx == nil || ctx.TaskType == "" {
		return 0.5 // 默认中等得分
	}

	if score, ok := c.profile.TaskScores[ctx.TaskType]; ok {
		return score
	}
	return 0.5
}

// calculateComplexityScore 计算复杂度匹配度得分
func (r *SmartRouter) calculateComplexityScore(c candidate, ctx *RoutingContext) float64 {
	if c.profile == nil || ctx == nil || ctx.Complexity == "" {
		return 0.5
	}

	if score, ok := c.profile.ComplexityScores[ctx.Complexity]; ok {
		return score
	}
	return 0.5
}

// calculateCostScore 计算成本得分
func (r *SmartRouter) calculateCostScore(c candidate, ctx *RoutingContext, req llm.CompletionRequest) float64 {
	// 估算请求成本
	estimatedCost := r.estimateCost(c, req)

	// 检查是否超出约束
	if ctx != nil && ctx.Constraints.MaxCostPerRequest > 0 {
		if estimatedCost > ctx.Constraints.MaxCostPerRequest {
			return 0.0 // 超出预算，得分为 0
		}
	}

	// 成本越低得分越高（反转）
	// 假设最高成本为 $0.1 每请求
	maxCost := 0.1
	if estimatedCost >= maxCost {
		return 0.1
	}
	return 1.0 - (estimatedCost / maxCost)
}

// calculateLatencyScore 计算延迟得分
func (r *SmartRouter) calculateLatencyScore(c candidate, ctx *RoutingContext) float64 {
	// 获取历史延迟
	r.mu.RLock()
	latency := r.latencies[c.providerName]
	r.mu.RUnlock()

	// 如果有档案中的延迟信息，优先使用
	if c.profile != nil && c.profile.AverageLatencyMs > 0 {
		latency = time.Duration(c.profile.AverageLatencyMs) * time.Millisecond
	}

	// 如果没有延迟数据，返回默认得分
	if latency == 0 {
		return 0.5
	}

	// 检查是否超出约束
	if ctx != nil && ctx.Constraints.MaxLatencyMs > 0 {
		if int(latency.Milliseconds()) > ctx.Constraints.MaxLatencyMs {
			return 0.0 // 超出延迟限制，得分为 0
		}
	}

	// 延迟越低得分越高
	// 假设最大可接受延迟为 10 秒
	maxLatency := 10000.0 // 毫秒
	latencyMs := float64(latency.Milliseconds())
	if latencyMs >= maxLatency {
		return 0.1
	}
	return 1.0 - (latencyMs / maxLatency)
}

// calculateCapabilityScore 计算能力匹配度得分
func (r *SmartRouter) calculateCapabilityScore(c candidate, ctx *RoutingContext) float64 {
	if ctx == nil || len(ctx.RequiredCapabilities) == 0 {
		return 1.0 // 没有特殊要求，满分
	}

	matched := 0
	for _, cap := range ctx.RequiredCapabilities {
		if c.modelInfo.HasFeature(cap) {
			matched++
		}
	}

	if len(ctx.RequiredCapabilities) == 0 {
		return 1.0
	}
	return float64(matched) / float64(len(ctx.RequiredCapabilities))
}

// calculatePreferenceScore 计算偏好得分
func (r *SmartRouter) calculatePreferenceScore(c candidate, ctx *RoutingContext) float64 {
	if ctx == nil {
		return 0.5
	}

	score := 0.5 // 默认得分

	// 首选 Provider 加分
	for _, pref := range ctx.Constraints.PreferredProviders {
		if pref == c.providerName {
			score += 0.3
			break
		}
	}

	// 首选模型加分
	for _, pref := range ctx.Constraints.PreferredModels {
		if pref == c.modelID {
			score += 0.2
			break
		}
	}

	// 限制在 0-1 范围内
	if score > 1.0 {
		score = 1.0
	}
	return score
}

// calculateTotalScore 根据优先级策略计算总分
func (r *SmartRouter) calculateTotalScore(score candidateScore, ctx *RoutingContext) float64 {
	priority := PriorityBalanced
	if ctx != nil && ctx.Priority != "" {
		priority = ctx.Priority
	}

	switch priority {
	case PriorityQuality:
		// 质量优先：任务得分权重最高
		return score.TaskScore*0.4 +
			score.ComplexityScore*0.25 +
			score.CapabilityScore*0.2 +
			score.CostScore*0.05 +
			score.LatencyScore*0.05 +
			score.PreferenceScore*0.05

	case PriorityCost:
		// 成本优先：成本得分权重最高
		return score.CostScore*0.5 +
			score.TaskScore*0.2 +
			score.ComplexityScore*0.1 +
			score.CapabilityScore*0.1 +
			score.LatencyScore*0.05 +
			score.PreferenceScore*0.05

	case PriorityLatency:
		// 延迟优先：延迟得分权重最高
		return score.LatencyScore*0.5 +
			score.TaskScore*0.2 +
			score.ComplexityScore*0.1 +
			score.CapabilityScore*0.1 +
			score.CostScore*0.05 +
			score.PreferenceScore*0.05

	case PriorityBalanced:
		fallthrough
	default:
		// 均衡策略
		return score.TaskScore*0.25 +
			score.ComplexityScore*0.15 +
			score.CostScore*0.2 +
			score.LatencyScore*0.15 +
			score.CapabilityScore*0.15 +
			score.PreferenceScore*0.1
	}
}

// estimateCost 估算请求成本
func (r *SmartRouter) estimateCost(c candidate, req llm.CompletionRequest) float64 {
	// 估算输入 Token 数
	inputTokens := 0
	for _, msg := range req.Messages {
		inputTokens += len(msg.Content) / 4 // 粗略估算
	}

	// 估算输出 Token 数（使用 MaxTokens 或默认值）
	outputTokens := req.MaxTokens
	if outputTokens == 0 {
		outputTokens = 500 // 默认假设
	}

	// 计算成本（每百万 Token）
	inputCost := float64(inputTokens) * c.modelInfo.InputCost / 1000000
	outputCost := float64(outputTokens) * c.modelInfo.OutputCost / 1000000

	return inputCost + outputCost
}

// estimateLatency 估算延迟
func (r *SmartRouter) estimateLatency(c candidate) int {
	// 优先使用档案中的数据
	if c.profile != nil && c.profile.AverageLatencyMs > 0 {
		return c.profile.AverageLatencyMs
	}

	// 使用历史数据
	r.mu.RLock()
	latency := r.latencies[c.providerName]
	r.mu.RUnlock()

	if latency > 0 {
		return int(latency.Milliseconds())
	}

	// 默认值
	return 1000
}

// buildDecisionReason 构建决策原因说明
func (r *SmartRouter) buildDecisionReason(best scoredCandidate, ctx *RoutingContext) string {
	taskType := "通用"
	if ctx != nil && ctx.TaskType != "" {
		taskType = string(ctx.TaskType)
	}

	return fmt.Sprintf(
		"选择 %s (%s) 用于%s任务，综合得分 %.2f (任务匹配: %.2f, 复杂度匹配: %.2f, 成本: %.2f, 延迟: %.2f)",
		best.candidate.modelID,
		best.candidate.providerName,
		taskType,
		best.score.Total,
		best.score.TaskScore,
		best.score.ComplexityScore,
		best.score.CostScore,
		best.score.LatencyScore,
	)
}

// autoClassifyRequest 自动分类请求
func (r *SmartRouter) autoClassifyRequest(ctx context.Context, req llm.CompletionRequest) *RoutingContext {
	if !r.autoClassify || r.classifier == nil {
		return nil
	}

	taskType, complexity := r.classifier.Classify(ctx, req)
	return NewRoutingContext(taskType, complexity)
}

// recordRouting 记录路由历史
func (r *SmartRouter) recordRouting(decision *RoutingDecision, ctx *RoutingContext, success bool, elapsed time.Duration, resp *llm.CompletionResponse) {
	r.historyMu.Lock()
	defer r.historyMu.Unlock()

	// 计算实际成本
	var actualCost float64
	if resp != nil {
		profile := r.GetProfile(decision.ModelID)
		if profile != nil {
			actualCost = float64(resp.Usage.PromptTokens)*profile.InputCostPerMillion/1000000 +
				float64(resp.Usage.CompletionTokens)*profile.OutputCostPerMillion/1000000
		}
	}

	record := RoutingRecord{
		Decision:      decision,
		Context:       ctx,
		Success:       success,
		ActualLatency: elapsed,
		ActualCost:    actualCost,
		Timestamp:     time.Now(),
	}

	r.routingHistory = append(r.routingHistory, record)

	// 限制历史记录大小
	if len(r.routingHistory) > r.maxHistorySize {
		r.routingHistory = r.routingHistory[len(r.routingHistory)-r.maxHistorySize:]
	}
}

// GetRoutingHistory 获取路由历史
func (r *SmartRouter) GetRoutingHistory() []RoutingRecord {
	r.historyMu.RLock()
	defer r.historyMu.RUnlock()

	result := make([]RoutingRecord, len(r.routingHistory))
	copy(result, r.routingHistory)
	return result
}

// GetRoutingStats 获取路由统计
func (r *SmartRouter) GetRoutingStats() SmartRoutingStats {
	r.historyMu.RLock()
	defer r.historyMu.RUnlock()

	stats := SmartRoutingStats{
		TotalRequests:  len(r.routingHistory),
		ModelUsage:     make(map[string]int),
		ProviderUsage:  make(map[string]int),
		TaskTypeUsage:  make(map[TaskType]int),
		AverageLatency: make(map[string]time.Duration),
		AverageCost:    make(map[string]float64),
		SuccessRate:    make(map[string]float64),
		latencySums:    make(map[string]time.Duration),
		costSums:       make(map[string]float64),
		successCounts:  make(map[string]int),
		requestCounts:  make(map[string]int),
	}

	for _, record := range r.routingHistory {
		if record.Decision == nil {
			continue
		}

		modelID := record.Decision.ModelID
		providerName := record.Decision.ProviderName

		// 统计使用次数
		stats.ModelUsage[modelID]++
		stats.ProviderUsage[providerName]++

		if record.Context != nil {
			stats.TaskTypeUsage[record.Context.TaskType]++
		}

		// 统计延迟
		stats.latencySums[modelID] += record.ActualLatency
		stats.requestCounts[modelID]++

		// 统计成本
		stats.costSums[modelID] += record.ActualCost

		// 统计成功率
		if record.Success {
			stats.successCounts[modelID]++
			stats.SuccessfulRequests++
		}
	}

	// 计算平均值
	for modelID, count := range stats.requestCounts {
		if count > 0 {
			stats.AverageLatency[modelID] = stats.latencySums[modelID] / time.Duration(count)
			stats.AverageCost[modelID] = stats.costSums[modelID] / float64(count)
			stats.SuccessRate[modelID] = float64(stats.successCounts[modelID]) / float64(count)
		}
	}

	return stats
}

// SmartRoutingStats 智能路由统计
type SmartRoutingStats struct {
	// TotalRequests 总请求数
	TotalRequests int

	// SuccessfulRequests 成功请求数
	SuccessfulRequests int

	// ModelUsage 各模型使用次数
	ModelUsage map[string]int

	// ProviderUsage 各 Provider 使用次数
	ProviderUsage map[string]int

	// TaskTypeUsage 各任务类型使用次数
	TaskTypeUsage map[TaskType]int

	// AverageLatency 各模型平均延迟
	AverageLatency map[string]time.Duration

	// AverageCost 各模型平均成本
	AverageCost map[string]float64

	// SuccessRate 各模型成功率
	SuccessRate map[string]float64

	// 内部统计字段
	latencySums   map[string]time.Duration
	costSums      map[string]float64
	successCounts map[string]int
	requestCounts map[string]int
}

// min 返回两个整数中的较小值
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
