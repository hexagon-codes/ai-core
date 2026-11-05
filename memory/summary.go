package memory

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ErrSummaryInProgress 表示已有摘要任务在执行；并发调用方应稍后重试，
// 避免在自定义 Summarizer 回调记忆方法时形成不可重入死锁。
var ErrSummaryInProgress = errors.New("memory: summary already in progress")

// Summarizer 摘要生成器接口
// 通常由 LLM Provider 实现
type Summarizer interface {
	// Summarize 生成摘要
	Summarize(ctx context.Context, content string) (string, error)
}

// SummaryMemory 自动摘要记忆
// 当记忆条目超过阈值时，自动将旧记忆压缩为摘要
type SummaryMemory struct {
	// 底层存储
	buffer *BufferMemory

	// 摘要生成器
	summarizer Summarizer

	// 配置
	config SummaryConfig

	// 当前摘要
	summary     string
	summaryTime time.Time

	// 互斥锁
	mu sync.RWMutex

	// summarizeMu 串行化摘要任务，避免多个外部摘要调用互相覆盖结果。
	summarizeMu sync.Mutex
}

// SummaryConfig 摘要配置
type SummaryConfig struct {
	// MaxEntries 触发摘要的最大条目数
	MaxEntries int

	// MaxTokens 触发摘要的最大 Token 数（估算）
	MaxTokens int

	// KeepRecent 保留最近的条目数
	KeepRecent int

	// SummaryPrompt 自定义摘要提示词
	SummaryPrompt string

	// ProgressiveSummary 渐进式摘要（将新内容与旧摘要合并）
	ProgressiveSummary bool

	// BufferCapacity 底层缓冲区容量
	BufferCapacity int
}

// DefaultSummaryConfig 返回默认摘要配置
func DefaultSummaryConfig() SummaryConfig {
	return SummaryConfig{
		MaxEntries:         20,
		MaxTokens:          4000,
		KeepRecent:         5,
		SummaryPrompt:      defaultSummaryPrompt,
		ProgressiveSummary: true,
		BufferCapacity:     100,
	}
}

const defaultSummaryPrompt = `请将以下对话历史压缩成一个简洁的摘要，保留关键信息和上下文：

%s

请用一段话总结以上内容，突出重要的事实、决策和结论。`

const progressiveSummaryPrompt = `你正在更新一份对话摘要。

当前摘要：
%s

新对话内容：
%s

请将新内容融入现有摘要，生成一份更新后的完整摘要。保持简洁，只保留关键信息。`

// SummaryOption 是 SummaryMemory 的配置选项
type SummaryOption func(*SummaryMemory)

// WithSummaryConfig 设置摘要配置
func WithSummaryConfig(config SummaryConfig) SummaryOption {
	return func(m *SummaryMemory) {
		m.config = config
	}
}

// WithMaxEntries 设置最大条目数
func WithMaxEntries(n int) SummaryOption {
	return func(m *SummaryMemory) {
		m.config.MaxEntries = n
	}
}

// WithKeepRecent 设置保留最近条目数
func WithKeepRecent(n int) SummaryOption {
	return func(m *SummaryMemory) {
		m.config.KeepRecent = n
	}
}

// WithProgressiveSummary 设置渐进式摘要
func WithProgressiveSummary(enabled bool) SummaryOption {
	return func(m *SummaryMemory) {
		m.config.ProgressiveSummary = enabled
	}
}

// NewSummaryMemory 创建摘要记忆
func NewSummaryMemory(summarizer Summarizer, opts ...SummaryOption) *SummaryMemory {
	m := &SummaryMemory{
		summarizer: summarizer,
		config:     DefaultSummaryConfig(),
	}

	for _, opt := range opts {
		opt(m)
	}

	m.buffer = NewBuffer(m.config.BufferCapacity)
	return m
}

// Save 保存记忆条目
func (m *SummaryMemory) Save(ctx context.Context, entry Entry) error {
	// 与 doSummarize 的 Clear/回写临界区使用同一把锁，避免并发 Save 的条目
	// 在“快照后、清理前”被丢弃。
	m.mu.Lock()
	err := m.buffer.Save(ctx, entry)
	shouldSummarize := err == nil && m.shouldSummarize()
	m.mu.Unlock()
	if err != nil {
		return err
	}

	// 摘要会调用外部服务，必须在锁外执行；summarizeMu 负责串行化摘要任务。
	if shouldSummarize {
		if err := m.doSummarize(ctx); err != nil {
			// 摘要失败不影响保存
			_ = err
		}
	}

	return nil
}

// SaveBatch 批量保存记忆条目
func (m *SummaryMemory) SaveBatch(ctx context.Context, entries []Entry) error {
	for _, entry := range entries {
		if err := m.Save(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

// Get 根据 ID 获取记忆条目
func (m *SummaryMemory) Get(ctx context.Context, id string) (*Entry, error) {
	return m.buffer.Get(ctx, id)
}

// Search 搜索记忆条目
func (m *SummaryMemory) Search(ctx context.Context, query SearchQuery) ([]Entry, error) {
	return m.buffer.Search(ctx, query)
}

// Delete 删除指定 ID 的记忆条目
func (m *SummaryMemory) Delete(ctx context.Context, id string) error {
	return m.buffer.Delete(ctx, id)
}

// Clear 清空所有记忆
func (m *SummaryMemory) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.summary = ""
	m.summaryTime = time.Time{}
	return m.buffer.Clear(ctx)
}

// Stats 返回记忆统计信息
func (m *SummaryMemory) Stats() MemoryStats {
	stats := m.buffer.Stats()
	m.mu.RLock()
	summary := m.summary
	m.mu.RUnlock()
	if summary != "" {
		// 估算摘要的 token 数
		stats.TokenCount += len(summary) / 4
	}
	return stats
}

// GetSummary 返回当前摘要
func (m *SummaryMemory) GetSummary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.summary
}

// GetContext 返回完整上下文（摘要 + 最近记忆）
func (m *SummaryMemory) GetContext() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var parts []string

	if m.summary != "" {
		parts = append(parts, fmt.Sprintf("[历史摘要] %s", m.summary))
	}

	entries := m.buffer.Entries()
	for _, entry := range entries {
		parts = append(parts, fmt.Sprintf("[%s] %s", entry.Role, entry.Content))
	}

	return strings.Join(parts, "\n")
}

// GetContextEntries 返回上下文条目（摘要作为系统消息 + 最近记忆）
func (m *SummaryMemory) GetContextEntries() []Entry {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var entries []Entry

	if m.summary != "" {
		entries = append(entries, Entry{
			ID:        "summary",
			Role:      "system",
			Content:   fmt.Sprintf("对话历史摘要: %s", m.summary),
			CreatedAt: m.summaryTime,
		})
	}

	entries = append(entries, m.buffer.Entries()...)
	return entries
}

// Entries 返回所有条目
func (m *SummaryMemory) Entries() []Entry {
	return m.buffer.Entries()
}

// shouldSummarize 检查是否需要触发摘要
func (m *SummaryMemory) shouldSummarize() bool {
	stats := m.buffer.Stats()

	// 按条目数判断
	if m.config.MaxEntries > 0 && stats.EntryCount > m.config.MaxEntries {
		return true
	}

	// 按 Token 数判断
	if m.config.MaxTokens > 0 && stats.TokenCount > m.config.MaxTokens {
		return true
	}

	return false
}

// doSummarize 执行摘要
// 注意：调用此方法时不应持有 m.mu 锁，因为它会调用外部服务
func (m *SummaryMemory) doSummarize(ctx context.Context) error {
	if !m.summarizeMu.TryLock() {
		return ErrSummaryInProgress
	}
	defer m.summarizeMu.Unlock()

	if m.summarizer == nil {
		return fmt.Errorf("summary memory: summarizer is nil")
	}

	entries := m.buffer.Entries()
	if len(entries) <= m.config.KeepRecent {
		return nil
	}

	// 记录快照时的条目数，用于后续检测并发修改
	snapshotLen := len(entries)

	// 需要摘要的条目
	toSummarize := entries[:len(entries)-m.config.KeepRecent]
	recentEntries := entries[len(entries)-m.config.KeepRecent:]

	// 构建待摘要内容
	var content strings.Builder
	for _, entry := range toSummarize {
		content.WriteString(fmt.Sprintf("%s: %s\n", entry.Role, entry.Content))
	}

	// 获取当前摘要（需要锁保护读取）
	m.mu.RLock()
	currentSummary := m.summary
	progressiveSummary := m.config.ProgressiveSummary
	summaryPrompt := m.config.SummaryPrompt
	m.mu.RUnlock()

	// 生成摘要（不持有锁，因为调用外部服务）
	var prompt string
	if progressiveSummary && currentSummary != "" {
		prompt = fmt.Sprintf(progressiveSummaryPrompt, currentSummary, content.String())
	} else {
		prompt = fmt.Sprintf(summaryPrompt, content.String())
	}

	newSummary, err := m.summarizer.Summarize(ctx, prompt)
	if err != nil {
		return fmt.Errorf("summarize failed: %w", err)
	}

	// Hold the lock during the entire Clear + re-insert phase to prevent
	// concurrent Save from losing data between Clear and re-insert (TOCTOU fix).
	m.mu.Lock()
	defer m.mu.Unlock()

	// 更新摘要
	m.summary = newSummary
	m.summaryTime = time.Now()

	// 清理旧条目，只保留最近的
	// 重新读取当前 entries，保留摘要期间新增的条目
	currentEntries := m.buffer.Entries()
	m.buffer.Clear(ctx)

	// 保留快照时标记的 recentEntries
	for _, entry := range recentEntries {
		m.buffer.Save(ctx, entry)
	}
	// 保留摘要期间新增的条目（快照之后新写入的）
	if len(currentEntries) > snapshotLen {
		for _, entry := range currentEntries[snapshotLen:] {
			m.buffer.Save(ctx, entry)
		}
	}

	return nil
}

// ForceSummarize 强制执行摘要
// 注意：此方法不持有 m.mu 锁，因为 doSummarize 内部会按需加锁
func (m *SummaryMemory) ForceSummarize(ctx context.Context) error {
	return m.doSummarize(ctx)
}

// SetSummary 设置摘要（用于恢复状态）
func (m *SummaryMemory) SetSummary(summary string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.summary = summary
	m.summaryTime = time.Now()
}

// SimpleSummarizer 简单的基于函数的摘要器
type SimpleSummarizer struct {
	fn func(ctx context.Context, content string) (string, error)
}

// NewSimpleSummarizer 创建简单摘要器
func NewSimpleSummarizer(fn func(ctx context.Context, content string) (string, error)) *SimpleSummarizer {
	return &SimpleSummarizer{fn: fn}
}

// Summarize 生成摘要
func (s *SimpleSummarizer) Summarize(ctx context.Context, content string) (string, error) {
	return s.fn(ctx, content)
}

// 确保实现了接口
var _ Memory = (*SummaryMemory)(nil)
var _ Summarizer = (*SimpleSummarizer)(nil)
