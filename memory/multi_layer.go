package memory

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// MemoryLayer 记忆层类型
type MemoryLayer string

const (
	// LayerWorking 工作记忆层（当前对话上下文）
	LayerWorking MemoryLayer = "working"

	// LayerShortTerm 短期记忆层（最近几轮对话）
	LayerShortTerm MemoryLayer = "short_term"

	// LayerLongTerm 长期记忆层（持久化存储，语义检索）
	LayerLongTerm MemoryLayer = "long_term"
)

// MultiLayerMemory 多层记忆系统
// 模拟人类记忆的分层结构：工作记忆 -> 短期记忆 -> 长期记忆
type MultiLayerMemory struct {
	// 工作记忆（当前对话上下文，容量小，访问快）
	working *BufferMemory

	// 短期记忆（最近几轮对话，带摘要压缩）
	shortTerm *SummaryMemory

	// 长期记忆（向量语义检索）
	longTerm *VectorMemory

	// 配置
	config MultiLayerConfig

	// 互斥锁（保护 working/shortTerm/longTerm 的快速读写操作）
	mu sync.RWMutex

	// transferMu 专用于 Transfer 操作，避免持有 mu 期间调用外部服务（embedder 等）
	// Transfer 持有 transferMu 时不阻塞 mu 保护的读操作
	transferMu sync.Mutex

	// 统计
	stats MultiLayerStats
}

// MultiLayerConfig 多层记忆配置
type MultiLayerConfig struct {
	// Working 工作记忆配置
	Working WorkingConfig

	// ShortTerm 短期记忆配置
	ShortTerm ShortTermConfig

	// LongTerm 长期记忆配置
	LongTerm LongTermConfig

	// TransferPolicy 记忆转移策略
	TransferPolicy TransferPolicy
}

// WorkingConfig 工作记忆配置
type WorkingConfig struct {
	// Capacity 容量（条目数）
	Capacity int
}

// ShortTermConfig 短期记忆配置
type ShortTermConfig struct {
	// MaxEntries 最大条目数
	MaxEntries int

	// KeepRecent 保留最近条目数
	KeepRecent int

	// Summarizer 摘要生成器
	Summarizer Summarizer
}

// LongTermConfig 长期记忆配置
type LongTermConfig struct {
	// Embedder 向量嵌入器
	Embedder Embedder

	// Store 向量存储
	Store VectorStore

	// MinScore 最小相似度
	MinScore float32

	// TopK 检索数量
	TopK int
}

// TransferPolicy 记忆转移策略
type TransferPolicy struct {
	// AutoTransfer 自动转移
	AutoTransfer bool

	// WorkingToShortTermThreshold 工作记忆到短期记忆的阈值
	WorkingToShortTermThreshold int

	// ShortTermToLongTermThreshold 短期记忆到长期记忆的阈值
	ShortTermToLongTermThreshold int
}

// MultiLayerStats 多层记忆统计
type MultiLayerStats struct {
	// WorkingCount 工作记忆条目数
	WorkingCount int

	// ShortTermCount 短期记忆条目数
	ShortTermCount int

	// LongTermCount 长期记忆条目数
	LongTermCount int

	// TransferCount 转移次数
	TransferCount int

	// LastTransferTime 最后转移时间
	LastTransferTime time.Time
}

// DefaultMultiLayerConfig 返回默认配置
func DefaultMultiLayerConfig() MultiLayerConfig {
	return MultiLayerConfig{
		Working: WorkingConfig{
			Capacity: 10,
		},
		ShortTerm: ShortTermConfig{
			MaxEntries: 50,
			KeepRecent: 10,
		},
		LongTerm: LongTermConfig{
			MinScore: 0.75,
			TopK:     5,
		},
		TransferPolicy: TransferPolicy{
			AutoTransfer:                 true,
			WorkingToShortTermThreshold:  8,
			ShortTermToLongTermThreshold: 40,
		},
	}
}

// MultiLayerOption 配置选项
type MultiLayerOption func(*MultiLayerMemory)

// WithMultiLayerConfig 设置配置
func WithMultiLayerConfig(config MultiLayerConfig) MultiLayerOption {
	return func(m *MultiLayerMemory) {
		m.config = config
	}
}

// WithSummarizer 设置摘要器
func WithSummarizer(s Summarizer) MultiLayerOption {
	return func(m *MultiLayerMemory) {
		m.config.ShortTerm.Summarizer = s
	}
}

// WithEmbedder 设置嵌入器
func WithEmbedder(e Embedder) MultiLayerOption {
	return func(m *MultiLayerMemory) {
		m.config.LongTerm.Embedder = e
	}
}

// WithLongTermStore 设置长期记忆存储
func WithLongTermStore(store VectorStore) MultiLayerOption {
	return func(m *MultiLayerMemory) {
		m.config.LongTerm.Store = store
	}
}

// NewMultiLayerMemory 创建多层记忆
func NewMultiLayerMemory(opts ...MultiLayerOption) *MultiLayerMemory {
	m := &MultiLayerMemory{
		config: DefaultMultiLayerConfig(),
	}

	for _, opt := range opts {
		opt(m)
	}

	// 初始化工作记忆
	m.working = NewBuffer(m.config.Working.Capacity)

	// 初始化短期记忆
	if m.config.ShortTerm.Summarizer != nil {
		m.shortTerm = NewSummaryMemory(
			m.config.ShortTerm.Summarizer,
			WithMaxEntries(m.config.ShortTerm.MaxEntries),
			WithKeepRecent(m.config.ShortTerm.KeepRecent),
		)
	}

	// 初始化长期记忆
	if m.config.LongTerm.Embedder != nil {
		m.longTerm = NewVectorMemory(
			m.config.LongTerm.Embedder,
			WithMinScore(m.config.LongTerm.MinScore),
			WithVectorStore(m.config.LongTerm.Store),
		)
	}

	return m
}

// Save 保存记忆条目到工作记忆
func (m *MultiLayerMemory) Save(ctx context.Context, entry Entry) error {
	m.mu.Lock()

	// 保存到工作记忆
	if err := m.working.Save(ctx, entry); err != nil {
		m.mu.Unlock()
		return err
	}

	// 检查是否需要转移
	needTransfer := m.config.TransferPolicy.AutoTransfer &&
		m.working.Stats().EntryCount >= m.config.TransferPolicy.WorkingToShortTermThreshold
	m.mu.Unlock()

	// 在释放锁后执行转移（转移可能调用外部服务）
	if needTransfer {
		m.Transfer(ctx)
	}

	return nil
}

// SaveBatch 批量保存记忆条目
func (m *MultiLayerMemory) SaveBatch(ctx context.Context, entries []Entry) error {
	for _, entry := range entries {
		if err := m.Save(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

// Get 根据 ID 获取记忆条目
func (m *MultiLayerMemory) Get(ctx context.Context, id string) (*Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 先从工作记忆查找
	if entry, err := m.working.Get(ctx, id); err == nil {
		return entry, nil
	} else if !errors.Is(err, ErrNotFound) {
		return nil, err
	}

	// 从短期记忆查找
	if m.shortTerm != nil {
		if entry, err := m.shortTerm.Get(ctx, id); err == nil {
			return entry, nil
		} else if !errors.Is(err, ErrNotFound) {
			return nil, err
		}
	}

	// 从长期记忆查找
	if m.longTerm != nil {
		if entry, err := m.longTerm.Get(ctx, id); err == nil {
			return entry, nil
		} else if !errors.Is(err, ErrNotFound) {
			return nil, err
		}
	}

	return nil, ErrNotFound
}

// Search 搜索记忆条目
// 综合所有层的搜索结果
func (m *MultiLayerMemory) Search(ctx context.Context, query SearchQuery) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var allEntries []Entry
	var errs []error

	// 从工作记忆搜索
	if entries, err := m.working.Search(ctx, query); err != nil {
		errs = append(errs, fmt.Errorf("working memory: %w", err))
	} else {
		for i := range entries {
			entries[i].Metadata = addLayerMeta(entries[i].Metadata, LayerWorking)
		}
		allEntries = append(allEntries, entries...)
	}

	// 从短期记忆搜索
	if m.shortTerm != nil {
		if entries, err := m.shortTerm.Search(ctx, query); err != nil {
			errs = append(errs, fmt.Errorf("short-term memory: %w", err))
		} else {
			for i := range entries {
				entries[i].Metadata = addLayerMeta(entries[i].Metadata, LayerShortTerm)
			}
			allEntries = append(allEntries, entries...)
		}
	}

	// 从长期记忆搜索（语义检索）
	if m.longTerm != nil && (query.Query != "" || len(query.Embedding) > 0) {
		if entries, err := m.longTerm.Search(ctx, query); err != nil {
			errs = append(errs, fmt.Errorf("long-term memory: %w", err))
		} else {
			for i := range entries {
				entries[i].Metadata = addLayerMeta(entries[i].Metadata, LayerLongTerm)
			}
			allEntries = append(allEntries, entries...)
		}
	}

	// 应用限制
	if query.Limit > 0 && len(allEntries) > query.Limit {
		allEntries = allEntries[:query.Limit]
	}

	if len(errs) > 0 {
		return allEntries, errors.Join(errs...)
	}
	return allEntries, nil
}

// SearchLayer 在指定层搜索
func (m *MultiLayerMemory) SearchLayer(ctx context.Context, layer MemoryLayer, query SearchQuery) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	switch layer {
	case LayerWorking:
		return m.working.Search(ctx, query)
	case LayerShortTerm:
		if m.shortTerm != nil {
			return m.shortTerm.Search(ctx, query)
		}
	case LayerLongTerm:
		if m.longTerm != nil {
			return m.longTerm.Search(ctx, query)
		}
	}

	return nil, fmt.Errorf("layer %s not available", layer)
}

// Delete 删除指定 ID 的记忆条目
func (m *MultiLayerMemory) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	found := false

	if err := m.working.Delete(ctx, id); err == nil {
		found = true
	} else if !errors.Is(err, ErrNotFound) {
		return err
	}

	if m.shortTerm != nil {
		if err := m.shortTerm.Delete(ctx, id); err == nil {
			found = true
		} else if !errors.Is(err, ErrNotFound) {
			return err
		}
	}

	if m.longTerm != nil {
		if err := m.longTerm.Delete(ctx, id); err == nil {
			found = true
		} else if !errors.Is(err, ErrNotFound) {
			return err
		}
	}

	if !found {
		return ErrNotFound
	}
	return nil
}

// Clear 清空所有记忆
func (m *MultiLayerMemory) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errs []error
	if err := m.working.Clear(ctx); err != nil {
		errs = append(errs, fmt.Errorf("working memory: %w", err))
	}
	if m.shortTerm != nil {
		if err := m.shortTerm.Clear(ctx); err != nil {
			errs = append(errs, fmt.Errorf("short-term memory: %w", err))
		}
	}
	if m.longTerm != nil {
		if err := m.longTerm.Clear(ctx); err != nil {
			errs = append(errs, fmt.Errorf("long-term memory: %w", err))
		}
	}

	m.stats = MultiLayerStats{}
	return errors.Join(errs...)
}

// ClearLayer 清空指定层
func (m *MultiLayerMemory) ClearLayer(ctx context.Context, layer MemoryLayer) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	switch layer {
	case LayerWorking:
		return m.working.Clear(ctx)
	case LayerShortTerm:
		if m.shortTerm != nil {
			return m.shortTerm.Clear(ctx)
		}
	case LayerLongTerm:
		if m.longTerm != nil {
			return m.longTerm.Clear(ctx)
		}
	}

	return nil
}

// Stats 返回记忆统计信息
func (m *MultiLayerMemory) Stats() MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := m.working.Stats()

	if m.shortTerm != nil {
		shortStats := m.shortTerm.Stats()
		stats.EntryCount += shortStats.EntryCount
	}

	if m.longTerm != nil {
		longStats := m.longTerm.Stats()
		stats.EntryCount += longStats.EntryCount
	}

	return stats
}

// MultiStats 返回多层统计信息
func (m *MultiLayerMemory) MultiStats() MultiLayerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := m.stats
	stats.WorkingCount = m.working.Stats().EntryCount
	if m.shortTerm != nil {
		stats.ShortTermCount = m.shortTerm.Stats().EntryCount
	}
	if m.longTerm != nil {
		stats.LongTermCount = m.longTerm.Stats().EntryCount
	}

	return stats
}

// GetContext 获取完整上下文
// 返回工作记忆 + 短期摘要 + 相关长期记忆
func (m *MultiLayerMemory) GetContext(ctx context.Context, query string) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var context []Entry

	// 添加短期记忆摘要
	if m.shortTerm != nil && m.shortTerm.GetSummary() != "" {
		context = append(context, Entry{
			ID:      "summary",
			Role:    "system",
			Content: fmt.Sprintf("对话历史摘要: %s", m.shortTerm.GetSummary()),
		})
	}

	// 添加相关长期记忆
	if m.longTerm != nil && query != "" {
		relevantMemories, err := m.longTerm.SemanticSearch(ctx, query, m.config.LongTerm.TopK)
		if err != nil {
			return nil, fmt.Errorf("long-term memory: %w", err)
		}
		context = append(context, relevantMemories...)
	}

	// 添加工作记忆
	context = append(context, m.working.Entries()...)

	return context, nil
}

// GetWorkingMemory 获取工作记忆
func (m *MultiLayerMemory) GetWorkingMemory() []Entry {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.working.Entries()
}

// transferWorkingToShortTerm 将工作记忆转移到短期记忆
func (m *MultiLayerMemory) transferWorkingToShortTerm(ctx context.Context) error {
	if m.shortTerm == nil {
		return nil
	}

	entries := m.working.Entries()
	// 保留最近几条在工作记忆
	keepCount := m.config.Working.Capacity / 2
	if keepCount < 2 {
		keepCount = 2
	}

	if len(entries) <= keepCount {
		return nil
	}

	// 转移较旧的条目到短期记忆
	toTransfer := entries[:len(entries)-keepCount]
	for _, entry := range toTransfer {
		if err := m.shortTerm.Save(ctx, entry); err != nil {
			return err
		}
	}

	// 清理工作记忆，保留最近的
	m.working.Clear(ctx)
	for _, entry := range entries[len(entries)-keepCount:] {
		m.working.Save(ctx, entry)
	}

	m.updateTransferStats()

	return nil
}

// transferShortTermToLongTerm 将短期记忆转移到长期记忆
// Called with transferMu held but NOT mu, so slow external calls
// (embedder, summarizer) do not block other read operations.
func (m *MultiLayerMemory) transferShortTermToLongTerm(ctx context.Context) error {
	if m.longTerm == nil || m.shortTerm == nil {
		return nil
	}

	entries := m.shortTerm.Entries()
	keepCount := m.config.ShortTerm.KeepRecent

	if len(entries) <= keepCount {
		return nil
	}

	// 转移较旧的条目到长期记忆（可能调用 embedder，慢速网络调用）
	toTransfer := entries[:len(entries)-keepCount]
	if err := m.longTerm.SaveBatch(ctx, toTransfer); err != nil {
		return err
	}

	// ForceSummarize 内部会按需加锁，不会死锁
	m.shortTerm.ForceSummarize(ctx)

	// Acquire mu briefly to update transfer stats
	m.mu.Lock()
	m.updateTransferStats()
	m.mu.Unlock()

	return nil
}

// updateTransferStats 更新转移统计（调用者必须持有 m.mu 锁）
func (m *MultiLayerMemory) updateTransferStats() {
	m.stats.TransferCount++
	m.stats.LastTransferTime = time.Now()
}

// Transfer 手动触发记忆转移
// Uses transferMu to serialize transfers without blocking reads on mu.
// The mu lock is acquired only for short data-copy phases, not during
// slow external calls (embedder, summarizer).
func (m *MultiLayerMemory) Transfer(ctx context.Context) error {
	m.transferMu.Lock()
	defer m.transferMu.Unlock()

	m.mu.Lock()
	err := m.transferWorkingToShortTerm(ctx)
	m.mu.Unlock()
	if err != nil {
		return err
	}

	// transferShortTermToLongTerm may call embedder (slow network call)
	// and summarizer. We do NOT hold mu during these calls so that
	// Get/Search/Save on other goroutines are not blocked.
	return m.transferShortTermToLongTerm(ctx)
}

// Recall 从长期记忆回忆相关内容
func (m *MultiLayerMemory) Recall(ctx context.Context, query string, topK int) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.longTerm == nil {
		return nil, fmt.Errorf("long term memory not configured")
	}

	return m.longTerm.SemanticSearch(ctx, query, topK)
}

// SaveToLongTerm 直接保存到长期记忆
func (m *MultiLayerMemory) SaveToLongTerm(ctx context.Context, entry Entry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.longTerm == nil {
		return fmt.Errorf("long term memory not configured")
	}

	return m.longTerm.Save(ctx, entry)
}

// addLayerMeta 添加层元数据
// 返回一个新的 map 副本，避免修改原始 Entry 的 Metadata
func addLayerMeta(metadata map[string]any, layer MemoryLayer) map[string]any {
	result := make(map[string]any, len(metadata)+1)
	for k, v := range metadata {
		result[k] = v
	}
	result["_layer"] = string(layer)
	return result
}

// 确保实现了接口
var _ Memory = (*MultiLayerMemory)(nil)
