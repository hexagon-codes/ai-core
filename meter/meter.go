package meter

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Record 表示单次 API 请求的记录
// 包含请求的所有关键信息，用于统计和审计
type Record struct {
	// Model 使用的模型名称，如 "gpt-4"、"claude-3-opus"
	Model string `json:"model"`
	// InputTokens 输入/提示词消耗的 Token 数
	InputTokens int `json:"input_tokens"`
	// OutputTokens 输出/生成内容消耗的 Token 数
	OutputTokens int `json:"output_tokens"`
	// Timestamp 请求的时间戳
	Timestamp time.Time `json:"timestamp"`
	// Latency 请求延迟/响应时间
	Latency time.Duration `json:"latency,omitempty"`
	// Success 请求是否成功
	Success bool `json:"success"`
	// Error 错误信息（如果请求失败）
	Error string `json:"error,omitempty"`
}

// Stats 是聚合的统计信息
// 包含请求数、Token 数、成本和延迟等指标
type Stats struct {
	// TotalRequests 总请求数
	TotalRequests int64 `json:"total_requests"`
	// SuccessRequests 成功请求数
	SuccessRequests int64 `json:"success_requests"`
	// FailedRequests 失败请求数
	FailedRequests int64 `json:"failed_requests"`
	// InputTokens 总输入 Token 数
	InputTokens int64 `json:"input_tokens"`
	// OutputTokens 总输出 Token 数
	OutputTokens int64 `json:"output_tokens"`
	// TotalTokens 总 Token 数（输入+输出）
	TotalTokens int64 `json:"total_tokens"`
	// EstimatedCost 预估总成本（美元）
	EstimatedCost float64 `json:"estimated_cost"`
	// AvgLatency 平均延迟
	AvgLatency time.Duration `json:"avg_latency"`
	// MinLatency 最小延迟
	MinLatency time.Duration `json:"min_latency"`
	// MaxLatency 最大延迟
	MaxLatency time.Duration `json:"max_latency"`
}

// ModelStats 是按模型分组的统计信息
type ModelStats struct {
	// Model 模型名称
	Model string `json:"model"`
	// Stats 该模型的统计数据
	Stats
}

// Pricing 定义模型的 Token 价格
type Pricing struct {
	// InputPrice 输入 Token 价格（美元/百万 Token）
	InputPrice float64
	// OutputPrice 输出 Token 价格（美元/百万 Token）
	OutputPrice float64
}

// defaultPricing 是预定义的主流模型定价表
// 数据来源：各厂商官方定价页面（2024年数据）
// 注意：价格可能随时调整，请以官方为准
var defaultPricing = map[string]Pricing{
	"gpt-4":           {InputPrice: 30.0, OutputPrice: 60.0},
	"gpt-4-turbo":     {InputPrice: 10.0, OutputPrice: 30.0},
	"gpt-4o":          {InputPrice: 2.5, OutputPrice: 10.0},
	"gpt-4o-mini":     {InputPrice: 0.15, OutputPrice: 0.6},
	"gpt-3.5-turbo":   {InputPrice: 0.5, OutputPrice: 1.5},
	"claude-3-opus":   {InputPrice: 15.0, OutputPrice: 75.0},
	"claude-3-sonnet": {InputPrice: 3.0, OutputPrice: 15.0},
	"claude-3-haiku":  {InputPrice: 0.25, OutputPrice: 1.25},
	"gemini-pro":      {InputPrice: 0.5, OutputPrice: 1.5},
	"deepseek":        {InputPrice: 0.14, OutputPrice: 0.28},
}

// DefaultMaxRecords 默认最大记录数
// 超过此限制时，会自动删除最旧的记录
const DefaultMaxRecords = 100000

// Meter 是 AI API 用量统计器
// 记录所有请求并提供统计分析功能
//
// 主要功能：
//   - 记录每次 API 请求的 Token 消耗
//   - 统计成功/失败请求数
//   - 计算延迟指标（平均/最小/最大）
//   - 估算 API 调用成本
//   - 按模型分组统计
//   - 时间窗口统计
//   - 自动清理旧记录（防止内存无限增长）
//
// 线程安全，可在并发环境中使用
type Meter struct {
	records    []Record           // 详细记录列表
	pricing    map[string]Pricing // 模型定价表
	maxRecords int                // 最大记录数，0 表示无限制
	mu         sync.RWMutex       // 保护 records 的读写锁

	// 快速计数器（原子操作，避免锁竞争）
	totalRequests   atomic.Int64  // 总请求数
	successRequests atomic.Int64  // 成功请求数
	inputTokens     atomic.Int64  // 总输入 Token
	outputTokens    atomic.Int64  // 总输出 Token
	totalLatency    atomic.Int64  // 总延迟（纳秒）
	latencyCount    atomic.Int64  // 有延迟记录的请求数（用于计算平均值）
	totalCost       atomic.Uint64 // 累计成本（美元，按 float64 bits 存储，避免微美元取整丢失低额成本）
}

// addCost 以 float64 精度原子累加成本（CAS 循环，避免单条记录取整丢失 < 1 微美元的成本）。
func (m *Meter) addCost(delta float64) {
	for {
		old := m.totalCost.Load()
		newVal := math.Float64bits(math.Float64frombits(old) + delta)
		if m.totalCost.CompareAndSwap(old, newVal) {
			return
		}
	}
}

// loadCost 读取以 float64 bits 存储的累计成本。
func (m *Meter) loadCost() float64 {
	return math.Float64frombits(m.totalCost.Load())
}

// New 创建新的用量统计器
// 使用默认的模型定价表（会创建副本，不影响原始数据）
// 默认最大记录数为 DefaultMaxRecords
func New() *Meter {
	return NewWithOptions(nil, DefaultMaxRecords)
}

// NewWithOptions 创建带选项的用量统计器
// pricing: 自定义定价表（nil 使用默认）
// maxRecords: 最大记录数（0 表示无限制）
func NewWithOptions(pricing map[string]Pricing, maxRecords int) *Meter {
	// 创建定价表的副本，避免修改全局默认值
	p := make(map[string]Pricing, len(defaultPricing))
	for k, v := range defaultPricing {
		p[k] = v
	}
	// 合并自定义定价
	for k, v := range pricing {
		p[k] = v
	}
	return &Meter{
		records:    make([]Record, 0),
		pricing:    p,
		maxRecords: maxRecords,
	}
}

// NewWithPricing 创建带自定义定价的统计器
// 自定义定价会合并到默认定价表中（覆盖同名模型）
func NewWithPricing(pricing map[string]Pricing) *Meter {
	return NewWithOptions(pricing, DefaultMaxRecords)
}

// SetPricing 设置或更新指定模型的定价
// 可用于添加新模型或覆盖默认定价
func (m *Meter) SetPricing(model string, pricing Pricing) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pricing[model] = pricing
}

// Record 记录一次成功的 API 请求
// 这是最常用的记录方法，适用于不需要延迟信息的场景
func (m *Meter) Record(model string, inputTokens, outputTokens int) {
	m.RecordWithDetails(model, inputTokens, outputTokens, 0, true, "")
}

// RecordWithLatency 记录带延迟信息的成功请求
// 延迟信息用于计算平均响应时间等指标
func (m *Meter) RecordWithLatency(model string, inputTokens, outputTokens int, latency time.Duration) {
	m.RecordWithDetails(model, inputTokens, outputTokens, latency, true, "")
}

// RecordError 记录失败的 API 请求
// 失败请求只记录输入 Token（因为没有输出）
func (m *Meter) RecordError(model string, inputTokens int, err error) {
	errStr := ""
	if err != nil {
		errStr = err.Error()
	}
	m.RecordWithDetails(model, inputTokens, 0, 0, false, errStr)
}

// RecordWithDetails 记录请求的完整信息
// 这是底层方法，其他 Record* 方法都调用此方法
// 当记录数超过 maxRecords 时，自动删除最旧的 10% 记录
//
// 注意：原子计数器（TotalRequests 等）统计的是所有历史请求的累计值，
// 不会因为记录被清理而减少。如果需要精确匹配 records 的统计，
// 请使用 StatsInWindow() 或遍历 Records() 自行计算。
//
// 线程安全
func (m *Meter) RecordWithDetails(model string, inputTokens, outputTokens int, latency time.Duration, success bool, errStr string) {
	record := Record{
		Model:        model,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		Timestamp:    time.Now(),
		Latency:      latency,
		Success:      success,
		Error:        errStr,
	}

	m.mu.Lock()
	m.records = append(m.records, record)
	// 自动清理：当超过限制时，删除最旧的 10% 记录
	// 原子计数器不减少，保持历史累计统计
	if m.maxRecords > 0 && len(m.records) > m.maxRecords {
		deleteCount := m.maxRecords / 10
		if deleteCount < 1 {
			deleteCount = 1
		}
		m.records = m.records[deleteCount:]
	}
	m.mu.Unlock()

	// 更新计数器（累计统计，不受清理影响）
	m.totalRequests.Add(1)
	if success {
		m.successRequests.Add(1)
	}
	m.inputTokens.Add(int64(inputTokens))
	m.outputTokens.Add(int64(outputTokens))
	if latency > 0 {
		m.totalLatency.Add(int64(latency))
		m.latencyCount.Add(1)
	}

	// 累计成本（以 float64 精度累加，与 records 清理无关）。
	// 不做单条取整，使 Stats() 全局成本与 StatsByModel() 按模型成本一致。
	m.mu.RLock()
	if pricing, ok := m.pricing[model]; ok {
		cost := float64(inputTokens)/1_000_000*pricing.InputPrice +
			float64(outputTokens)/1_000_000*pricing.OutputPrice
		m.addCost(cost)
	}
	m.mu.RUnlock()
}

// Stats 返回所有请求的聚合统计信息
// 包含请求数、Token 数、成本和延迟等指标
//
// 注意：成本和延迟的详细统计（MinLatency/MaxLatency）需要遍历 records，
// 当记录被清理后，这些值只反映剩余记录的情况。
// 如需精确的历史累计统计，请使用原子计数器相关字段。
//
// 线程安全
func (m *Meter) Stats() Stats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := Stats{
		TotalRequests:   m.totalRequests.Load(),
		SuccessRequests: m.successRequests.Load(),
		FailedRequests:  m.totalRequests.Load() - m.successRequests.Load(),
		InputTokens:     m.inputTokens.Load(),
		OutputTokens:    m.outputTokens.Load(),
		TotalTokens:     m.inputTokens.Load() + m.outputTokens.Load(),
	}

	// Use cumulative atomic counter for cost (consistent with TotalRequests
	// even after records are pruned). Stored as float64 bits.
	stats.EstimatedCost = m.loadCost()

	// 计算平均延迟（使用原子计数器，反映全部历史）
	latencyCount := m.latencyCount.Load()
	if latencyCount > 0 {
		stats.AvgLatency = time.Duration(m.totalLatency.Load() / latencyCount)
	}

	// 计算延迟详细统计（遍历剩余记录）
	for _, r := range m.records {
		if r.Latency > 0 {
			if stats.MinLatency == 0 || r.Latency < stats.MinLatency {
				stats.MinLatency = r.Latency
			}
			if r.Latency > stats.MaxLatency {
				stats.MaxLatency = r.Latency
			}
		}
	}

	return stats
}

// StatsByModel 返回指定模型的统计信息
// 只统计匹配模型名称的请求
func (m *Meter) StatsByModel(model string) Stats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.statsByModelLocked(model)
}

// statsByModelLocked 是 StatsByModel 的内部实现，调用者必须已持有锁
func (m *Meter) statsByModelLocked(model string) Stats {
	var stats Stats
	var totalLatency time.Duration
	var latencyCount int

	for _, r := range m.records {
		if r.Model != model {
			continue
		}

		stats.TotalRequests++
		if r.Success {
			stats.SuccessRequests++
		} else {
			stats.FailedRequests++
		}
		stats.InputTokens += int64(r.InputTokens)
		stats.OutputTokens += int64(r.OutputTokens)

		if r.Latency > 0 {
			totalLatency += r.Latency
			latencyCount++
			if stats.MinLatency == 0 || r.Latency < stats.MinLatency {
				stats.MinLatency = r.Latency
			}
			if r.Latency > stats.MaxLatency {
				stats.MaxLatency = r.Latency
			}
		}
	}

	stats.TotalTokens = stats.InputTokens + stats.OutputTokens

	// 计算成本
	if pricing, ok := m.pricing[model]; ok {
		stats.EstimatedCost = float64(stats.InputTokens)/1_000_000*pricing.InputPrice +
			float64(stats.OutputTokens)/1_000_000*pricing.OutputPrice
	}

	// 计算平均延迟
	if latencyCount > 0 {
		stats.AvgLatency = totalLatency / time.Duration(latencyCount)
	}

	return stats
}

// AllModelStats 返回按模型分组的统计信息列表
// 结果按请求数降序排列
func (m *Meter) AllModelStats() []ModelStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 收集所有模型
	models := make(map[string]bool)
	for _, r := range m.records {
		models[r.Model] = true
	}

	var result []ModelStats
	for model := range models {
		// 使用内部方法避免重复加锁导致死锁
		stats := m.statsByModelLocked(model)
		result = append(result, ModelStats{
			Model: model,
			Stats: stats,
		})
	}

	// 按请求数排序
	sort.Slice(result, func(i, j int) bool {
		return result[i].TotalRequests > result[j].TotalRequests
	})

	return result
}

// Records 返回所有原始记录的副本
// 用于导出、审计或自定义分析
func (m *Meter) Records() []Record {
	m.mu.RLock()
	defer m.mu.RUnlock()

	records := make([]Record, len(m.records))
	copy(records, m.records)
	return records
}

// RecordsSince 返回指定时间之后的所有记录
// 用于增量导出或时间窗口分析
func (m *Meter) RecordsSince(since time.Time) []Record {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var records []Record
	for _, r := range m.records {
		if r.Timestamp.After(since) {
			records = append(records, r)
		}
	}
	return records
}

// Clear 清空所有记录和计数器
// 重置统计器到初始状态
func (m *Meter) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.records = m.records[:0]
	m.totalRequests.Store(0)
	m.successRequests.Store(0)
	m.inputTokens.Store(0)
	m.outputTokens.Store(0)
	m.totalLatency.Store(0)
	m.latencyCount.Store(0)
	m.totalCost.Store(0)
}

// 注：totalCost 存的是 float64 bits；0.0 的 bits 恰为 0，故 Store(0) 即重置为 0.0。

// Report 生成人类可读的文本报告
// 包含总体统计和按模型分组的详细信息
func (m *Meter) Report() string {
	stats := m.Stats()
	modelStats := m.AllModelStats()

	var sb strings.Builder
	sb.WriteString("=== AI API Usage Report ===\n\n")

	sb.WriteString("Overall Statistics:\n")
	fmt.Fprintf(&sb, "  Total Requests:   %d\n", stats.TotalRequests)
	if stats.TotalRequests > 0 {
		fmt.Fprintf(&sb, "  Success:          %d (%.1f%%)\n",
			stats.SuccessRequests,
			float64(stats.SuccessRequests)/float64(stats.TotalRequests)*100)
	} else {
		fmt.Fprintf(&sb, "  Success:          %d\n", stats.SuccessRequests)
	}
	fmt.Fprintf(&sb, "  Failed:           %d\n", stats.FailedRequests)
	fmt.Fprintf(&sb, "  Input Tokens:     %d\n", stats.InputTokens)
	fmt.Fprintf(&sb, "  Output Tokens:    %d\n", stats.OutputTokens)
	fmt.Fprintf(&sb, "  Total Tokens:     %d\n", stats.TotalTokens)
	fmt.Fprintf(&sb, "  Estimated Cost:   $%.4f\n", stats.EstimatedCost)
	if stats.AvgLatency > 0 {
		fmt.Fprintf(&sb, "  Avg Latency:      %v\n", stats.AvgLatency)
		fmt.Fprintf(&sb, "  Min Latency:      %v\n", stats.MinLatency)
		fmt.Fprintf(&sb, "  Max Latency:      %v\n", stats.MaxLatency)
	}

	if len(modelStats) > 0 {
		sb.WriteString("\nBy Model:\n")
		for _, ms := range modelStats {
			fmt.Fprintf(&sb, "  %s:\n", ms.Model)
			fmt.Fprintf(&sb, "    Requests: %d, Tokens: %d, Cost: $%.4f\n",
				ms.TotalRequests, ms.TotalTokens, ms.EstimatedCost)
		}
	}

	return sb.String()
}

// JSON 返回 JSON 格式的统计数据
// 便于存储或通过 API 传输
func (m *Meter) JSON() ([]byte, error) {
	data := struct {
		Stats      Stats        `json:"stats"`
		ByModel    []ModelStats `json:"by_model"`
		RecordedAt time.Time    `json:"recorded_at"`
	}{
		Stats:      m.Stats(),
		ByModel:    m.AllModelStats(),
		RecordedAt: time.Now(),
	}
	return json.Marshal(data)
}

// ============== 时间窗口统计 ==============

// WindowStats 表示特定时间窗口内的统计信息
type WindowStats struct {
	// Window 时间窗口大小
	Window time.Duration
	// Stats 窗口内的统计数据
	Stats Stats
	// Start 窗口开始时间
	Start time.Time
	// End 窗口结束时间
	End time.Time
}

// StatsInWindow 返回最近指定时间窗口内的统计
// 用于实时监控和告警场景
//
// 示例：
//
//	// 获取最近 1 分钟的统计
//	ws := m.StatsInWindow(time.Minute)
//	fmt.Printf("最近1分钟请求数: %d\n", ws.Stats.TotalRequests)
func (m *Meter) StatsInWindow(window time.Duration) WindowStats {
	now := time.Now()
	since := now.Add(-window)

	records := m.RecordsSince(since)

	var stats Stats
	var totalLatency time.Duration
	var latencyCount int

	for _, r := range records {
		stats.TotalRequests++
		if r.Success {
			stats.SuccessRequests++
		} else {
			stats.FailedRequests++
		}
		stats.InputTokens += int64(r.InputTokens)
		stats.OutputTokens += int64(r.OutputTokens)

		if r.Latency > 0 {
			totalLatency += r.Latency
			latencyCount++
			if stats.MinLatency == 0 || r.Latency < stats.MinLatency {
				stats.MinLatency = r.Latency
			}
			if r.Latency > stats.MaxLatency {
				stats.MaxLatency = r.Latency
			}
		}
	}

	stats.TotalTokens = stats.InputTokens + stats.OutputTokens

	// 计算成本
	m.mu.RLock()
	for _, r := range records {
		if pricing, ok := m.pricing[r.Model]; ok {
			stats.EstimatedCost += float64(r.InputTokens)/1_000_000*pricing.InputPrice +
				float64(r.OutputTokens)/1_000_000*pricing.OutputPrice
		}
	}
	m.mu.RUnlock()

	if latencyCount > 0 {
		stats.AvgLatency = totalLatency / time.Duration(latencyCount)
	}

	return WindowStats{
		Window: window,
		Stats:  stats,
		Start:  since,
		End:    now,
	}
}

// ============== 全局统计器 ==============

// globalMeter 是应用级别的全局统计器
// 用于跨模块的统一统计
var globalMeter = New()

// Global 返回全局统计器实例
// 用于需要直接访问统计器的场景
func Global() *Meter {
	return globalMeter
}

// RecordGlobal 向全局统计器记录一次请求
// 这是最常用的全局记录方法
func RecordGlobal(model string, inputTokens, outputTokens int) {
	globalMeter.Record(model, inputTokens, outputTokens)
}

// StatsGlobal 返回全局统计信息
func StatsGlobal() Stats {
	return globalMeter.Stats()
}

// ============== Tracker 请求追踪器 ==============

// Tracker 是单次请求的追踪器
// 自动记录请求开始时间，计算延迟
// 适用于需要精确延迟统计的场景
//
// 示例：
//
//	tracker := meter.NewTracker("gpt-4").SetInputTokens(100)
//	// ... 执行 API 调用 ...
//	tracker.Done(50)  // 记录成功，自动计算延迟
type Tracker struct {
	meter       *Meter      // 关联的统计器
	model       string      // 模型名称
	inputTokens int         // 输入 Token 数
	startTime   time.Time   // 请求开始时间
	done        atomic.Bool // 防止多次调用 Done/Error
}

// NewTracker 创建请求追踪器
// 自动记录当前时间作为请求开始时间
func (m *Meter) NewTracker(model string) *Tracker {
	return &Tracker{
		meter:     m,
		model:     model,
		startTime: time.Now(),
	}
}

// SetInputTokens 设置输入 Token 数
// 支持链式调用
func (t *Tracker) SetInputTokens(n int) *Tracker {
	t.inputTokens = n
	return t
}

// Done 完成追踪并记录成功请求
// 自动计算从创建 Tracker 到调用 Done 的延迟
// 多次调用是安全的，只有首次调用会记录
func (t *Tracker) Done(outputTokens int) {
	if !t.done.CompareAndSwap(false, true) {
		return // 已经调用过，忽略重复调用
	}
	latency := time.Since(t.startTime)
	t.meter.RecordWithLatency(t.model, t.inputTokens, outputTokens, latency)
}

// Error 完成追踪并记录失败请求
// 多次调用是安全的，只有首次调用会记录
func (t *Tracker) Error(err error) {
	if !t.done.CompareAndSwap(false, true) {
		return
	}
	t.meter.RecordError(t.model, t.inputTokens, err)
}
