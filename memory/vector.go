package memory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/hexagon-codes/toolkit/util/idgen"
)

// Embedder 向量嵌入接口
// 通常由 LLM Provider 的 Embed 方法实现
type Embedder interface {
	// Embed 生成文本的向量嵌入
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// VectorStore 向量存储接口
// 可以由内存实现或外部向量数据库（Qdrant、Milvus 等）实现
type VectorStore interface {
	// Add 添加向量
	Add(ctx context.Context, id string, embedding []float32, metadata map[string]any) error

	// Search 搜索相似向量
	Search(ctx context.Context, embedding []float32, topK int) ([]VectorResult, error)

	// Delete 删除向量
	Delete(ctx context.Context, id string) error

	// Clear 清空所有向量
	Clear(ctx context.Context) error

	// Count 返回向量数量
	Count(ctx context.Context) (int, error)
}

// VectorResult 向量搜索结果
type VectorResult struct {
	// ID 向量 ID
	ID string `json:"id"`

	// Score 相似度分数 (0-1, 越大越相似)
	Score float32 `json:"score"`

	// Metadata 元数据
	Metadata map[string]any `json:"metadata,omitempty"`
}

// VectorMemory 向量记忆
// 使用向量嵌入进行语义搜索
type VectorMemory struct {
	// 条目存储
	entries map[string]*Entry
	// order 记录条目插入顺序，用于 Capacity 淘汰。
	order []string
	mu    sync.RWMutex

	// 向量嵌入器
	embedder Embedder

	// 向量存储
	store VectorStore

	// 配置
	config VectorConfig
}

// VectorConfig 向量记忆配置
type VectorConfig struct {
	// Dimension 向量维度
	Dimension int

	// Capacity 最大容量
	Capacity int

	// MinScore 最小相似度阈值
	MinScore float32

	// DefaultTopK 默认返回数量
	DefaultTopK int
}

// DefaultVectorConfig 返回默认配置
func DefaultVectorConfig() VectorConfig {
	return VectorConfig{
		Dimension:   1536, // OpenAI text-embedding-3-small
		Capacity:    10000,
		MinScore:    0.7,
		DefaultTopK: 10,
	}
}

// VectorOption 是 VectorMemory 的配置选项
type VectorOption func(*VectorMemory)

// WithVectorConfig 设置向量配置
func WithVectorConfig(config VectorConfig) VectorOption {
	return func(m *VectorMemory) {
		m.config = config
	}
}

// WithDimension 设置向量维度
func WithDimension(dim int) VectorOption {
	return func(m *VectorMemory) {
		m.config.Dimension = dim
	}
}

// WithMinScore 设置最小相似度
func WithMinScore(score float32) VectorOption {
	return func(m *VectorMemory) {
		m.config.MinScore = score
	}
}

// WithVectorStore 设置向量存储
func WithVectorStore(store VectorStore) VectorOption {
	return func(m *VectorMemory) {
		m.store = store
	}
}

// NewVectorMemory 创建向量记忆
func NewVectorMemory(embedder Embedder, opts ...VectorOption) *VectorMemory {
	m := &VectorMemory{
		entries:  make(map[string]*Entry),
		embedder: embedder,
		config:   DefaultVectorConfig(),
	}

	for _, opt := range opts {
		opt(m)
	}

	// 如果没有设置向量存储，使用内存存储
	if m.store == nil {
		m.store = NewMemoryVectorStore(m.config.Dimension)
	}

	return m
}

// Save 保存记忆条目
func (m *VectorMemory) Save(ctx context.Context, entry Entry) error {
	if entry.ID == "" {
		entry.ID = generateVectorID()
	}
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now()
	}

	// 在获取锁之前生成向量嵌入（避免持锁调用外部服务导致阻塞）
	if len(entry.Embedding) == 0 && entry.Content != "" {
		if m.embedder == nil {
			return fmt.Errorf("vector memory: embedder is nil")
		}
		embeddings, err := m.embedder.Embed(ctx, []string{entry.Content})
		if err != nil {
			return fmt.Errorf("generate embedding: %w", err)
		}
		if len(embeddings) > 0 && len(embeddings[0]) > 0 {
			entry.Embedding = embeddings[0]
		} else {
			return fmt.Errorf("embedder returned empty embedding for content")
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	_, existed := m.entries[entry.ID]

	// 先写向量库，成功后再发布内存条目，避免 Add 失败时内存与向量库不一致。
	if len(entry.Embedding) > 0 {
		metadata := map[string]any{
			"id":         entry.ID,
			"role":       entry.Role,
			"created_at": entry.CreatedAt.Unix(),
		}
		for k, v := range entry.Metadata {
			metadata[k] = v
		}
		if err := m.store.Add(ctx, entry.ID, entry.Embedding, metadata); err != nil {
			return fmt.Errorf("add to vector store: %w", err)
		}
	}

	m.entries[entry.ID] = &entry
	if !existed {
		m.order = append(m.order, entry.ID)
	}
	m.evictLocked(ctx)
	return nil
}

// evictLocked 按 Capacity 淘汰最旧条目；调用者必须持有 m.mu。
func (m *VectorMemory) evictLocked(ctx context.Context) {
	if m.config.Capacity <= 0 {
		return
	}
	for len(m.order) > m.config.Capacity {
		oldest := m.order[0]
		if err := m.store.Delete(ctx, oldest); err != nil {
			// 删除失败时保留条目和顺序，等待下次 Save 重试，避免内存与向量库分叉。
			return
		}
		delete(m.entries, oldest)
		m.order = m.order[1:]
	}
}

// SaveBatch 批量保存记忆条目
func (m *VectorMemory) SaveBatch(ctx context.Context, entries []Entry) error {
	// 批量生成嵌入
	var textsToEmbed []string
	var entryIndices []int

	for i, entry := range entries {
		if len(entry.Embedding) == 0 && entry.Content != "" {
			textsToEmbed = append(textsToEmbed, entry.Content)
			entryIndices = append(entryIndices, i)
		}
	}

	if len(textsToEmbed) > 0 {
		if m.embedder == nil {
			return fmt.Errorf("vector memory: embedder is nil")
		}
		embeddings, err := m.embedder.Embed(ctx, textsToEmbed)
		if err != nil {
			return fmt.Errorf("batch generate embeddings: %w", err)
		}
		for i, idx := range entryIndices {
			if i < len(embeddings) {
				entries[idx].Embedding = embeddings[i]
			}
		}
	}

	// 逐个保存
	for i := range entries {
		if err := m.Save(ctx, entries[i]); err != nil {
			return err
		}
	}

	return nil
}

// Get 根据 ID 获取记忆条目
func (m *VectorMemory) Get(ctx context.Context, id string) (*Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, ok := m.entries[id]
	if !ok {
		return nil, ErrNotFound
	}
	cloned := cloneEntry(entry)
	return &cloned, nil
}

// Search 搜索记忆条目
func (m *VectorMemory) Search(ctx context.Context, query SearchQuery) ([]Entry, error) {
	topK := query.Limit
	if topK <= 0 {
		topK = m.config.DefaultTopK
		if topK <= 0 {
			topK = 10
		}
	}

	// 在获取锁之前生成查询向量（避免持锁调用外部服务导致阻塞）
	var embedding []float32
	if len(query.Embedding) > 0 {
		embedding = query.Embedding
	} else if query.Query != "" {
		if m.embedder == nil {
			return nil, fmt.Errorf("vector memory: embedder is nil")
		}
		embeddings, err := m.embedder.Embed(ctx, []string{query.Query})
		if err != nil {
			return nil, fmt.Errorf("generate query embedding: %w", err)
		}
		if len(embeddings) > 0 {
			embedding = embeddings[0]
		}
	}

	offset := query.Offset
	if offset < 0 {
		offset = 0
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(embedding) > 0 {
		// 向量搜索。先多取 offset 条，过滤后再统一分页，避免 offset 被
		// 底层 store 的 topK 截断。
		fetchLimit := topK + offset
		results, err := m.store.Search(ctx, embedding, fetchLimit)
		if err != nil {
			return nil, fmt.Errorf("vector search: %w", err)
		}

		var entries []Entry
		for _, result := range results {
			if result.Score < m.config.MinScore {
				continue
			}
			if entry, ok := m.entries[result.ID]; ok {
				// 应用过滤器
				if !matchQuery(*entry, query) {
					continue
				}
				entryCopy := cloneEntry(entry)
				// 添加分数到元数据
				if entryCopy.Metadata == nil {
					entryCopy.Metadata = make(map[string]any)
				}
				entryCopy.Metadata["_score"] = result.Score
				entries = append(entries, entryCopy)
			}
		}
		if query.OrderBy != "" && !strings.EqualFold(query.OrderBy, "score") {
			sortEntries(entries, query.OrderBy, query.OrderDesc)
		}
		return paginateEntries(entries, query), nil
	}

	// 普通搜索（按时间排序）
	var entries []Entry
	for _, entry := range m.entries {
		if !matchQuery(*entry, query) {
			continue
		}
		entries = append(entries, cloneEntry(entry))
	}

	sortEntries(entries, query.OrderBy, query.OrderDesc)
	return paginateEntries(entries, query), nil
}

// sortEntries 按 SearchQuery.OrderBy 排序；空值保持既有的 created_at 默认行为。
func sortEntries(entries []Entry, orderBy string, desc bool) {
	switch strings.ToLower(strings.TrimSpace(orderBy)) {
	case "updated_at":
		if desc {
			sort.Slice(entries, func(i, j int) bool { return entries[i].UpdatedAt.After(entries[j].UpdatedAt) })
		} else {
			sort.Slice(entries, func(i, j int) bool { return entries[i].UpdatedAt.Before(entries[j].UpdatedAt) })
		}
	default:
		if desc {
			sort.Slice(entries, func(i, j int) bool { return entries[i].CreatedAt.After(entries[j].CreatedAt) })
		} else {
			sort.Slice(entries, func(i, j int) bool { return entries[i].CreatedAt.Before(entries[j].CreatedAt) })
		}
	}
}

// paginateEntries 应用 Offset/Limit，返回共享底层切片的只读窗口。
func paginateEntries(entries []Entry, query SearchQuery) []Entry {
	start := query.Offset
	if start < 0 {
		start = 0
	}
	if start >= len(entries) {
		return nil
	}
	end := len(entries)
	if query.Limit > 0 && start+query.Limit < end {
		end = start + query.Limit
	}
	return entries[start:end]
}

// SemanticSearch 语义搜索
func (m *VectorMemory) SemanticSearch(ctx context.Context, query string, topK int) ([]Entry, error) {
	return m.Search(ctx, SearchQuery{
		Query: query,
		Limit: topK,
	})
}

// Delete 删除指定 ID 的记忆条目
func (m *VectorMemory) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.entries[id]; !ok {
		return ErrNotFound
	}
	if err := m.store.Delete(ctx, id); err != nil {
		return err
	}
	delete(m.entries, id)
	for i, key := range m.order {
		if key == id {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	return nil
}

// Clear 清空所有记忆
func (m *VectorMemory) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.store.Clear(ctx); err != nil {
		return err
	}
	m.entries = make(map[string]*Entry)
	m.order = nil
	return nil
}

// Stats 返回记忆统计信息
func (m *VectorMemory) Stats() MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := MemoryStats{
		EntryCount: len(m.entries),
	}

	var oldest, newest time.Time
	for _, entry := range m.entries {
		if oldest.IsZero() || entry.CreatedAt.Before(oldest) {
			oldest = entry.CreatedAt
		}
		if newest.IsZero() || entry.CreatedAt.After(newest) {
			newest = entry.CreatedAt
		}
	}

	if !oldest.IsZero() {
		stats.OldestEntry = &oldest
	}
	if !newest.IsZero() {
		stats.NewestEntry = &newest
	}

	return stats
}

// vectorIDCounter 用于生成唯一向量 ID 的原子计数器
// generateVectorID 生成向量记忆 ID
// 复用 toolkit NanoID（抗碰撞），不自行用计数器+随机字节重造。
func generateVectorID() string {
	return "vec-" + idgen.NanoID()
}

// ============== 内存向量存储 ==============

// cloneMetadata 复制元数据 map，避免共享可变引用。
func cloneMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	cloned := make(map[string]any, len(metadata))
	for key, value := range metadata {
		cloned[key] = value
	}
	return cloned
}

// cloneEntry 返回条目副本，避免调用方绕过锁修改内部状态。
func cloneEntry(entry *Entry) Entry {
	if entry == nil {
		return Entry{}
	}
	cloned := *entry
	cloned.Embedding = append([]float32(nil), entry.Embedding...)
	if entry.Metadata != nil {
		cloned.Metadata = make(map[string]any, len(entry.Metadata))
		for key, value := range entry.Metadata {
			cloned.Metadata[key] = value
		}
	}
	if entry.Media != nil {
		cloned.Media = make([]MediaRef, len(entry.Media))
		for i, media := range entry.Media {
			cloned.Media[i] = media
			if media.Metadata != nil {
				cloned.Media[i].Metadata = make(map[string]any, len(media.Metadata))
				for key, value := range media.Metadata {
					cloned.Media[i].Metadata[key] = value
				}
			}
		}
	}
	return cloned
}

// MemoryVectorStore 内存向量存储
type MemoryVectorStore struct {
	vectors   map[string]vectorEntry
	dimension int
	mu        sync.RWMutex
}

type vectorEntry struct {
	embedding []float32
	metadata  map[string]any
}

// NewMemoryVectorStore 创建内存向量存储
func NewMemoryVectorStore(dimension int) *MemoryVectorStore {
	return &MemoryVectorStore{
		vectors:   make(map[string]vectorEntry),
		dimension: dimension,
	}
}

// Add 添加向量
func (s *MemoryVectorStore) Add(ctx context.Context, id string, embedding []float32, metadata map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.vectors[id] = vectorEntry{
		embedding: append([]float32(nil), embedding...),
		metadata:  cloneMetadata(metadata),
	}
	return nil
}

// Search 搜索相似向量
func (s *MemoryVectorStore) Search(ctx context.Context, query []float32, topK int) ([]VectorResult, error) {
	// 检查 context 是否已取消
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	var results []VectorResult

	checkInterval := 1000 // 每 1000 个向量检查一次 context
	count := 0
	for id, entry := range s.vectors {
		// 定期检查 context 是否已取消
		count++
		if count%checkInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}

		score := cosineSimilarity(query, entry.embedding)
		results = append(results, VectorResult{
			ID:       id,
			Score:    score,
			Metadata: cloneMetadata(entry.metadata),
		})
	}

	// 按分数降序排序
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// 返回 topK
	if topK > 0 && topK < len(results) {
		results = results[:topK]
	}

	return results, nil
}

// Delete 删除向量
func (s *MemoryVectorStore) Delete(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.vectors, id)
	return nil
}

// Clear 清空所有向量
func (s *MemoryVectorStore) Clear(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.vectors = make(map[string]vectorEntry)
	return nil
}

// Count 返回向量数量
func (s *MemoryVectorStore) Count(ctx context.Context) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.vectors), nil
}

// cosineSimilarity 计算余弦相似度
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)))
}

// ============== 简单嵌入器实现 ==============

// SimpleEmbedder 简单的基于函数的嵌入器
type SimpleEmbedder struct {
	fn func(ctx context.Context, texts []string) ([][]float32, error)
}

// NewSimpleEmbedder 创建简单嵌入器
func NewSimpleEmbedder(fn func(ctx context.Context, texts []string) ([][]float32, error)) *SimpleEmbedder {
	return &SimpleEmbedder{fn: fn}
}

// Embed 生成向量嵌入
func (e *SimpleEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return e.fn(ctx, texts)
}

// 确保实现了接口
var _ Memory = (*VectorMemory)(nil)
var _ VectorStore = (*MemoryVectorStore)(nil)
var _ Embedder = (*SimpleEmbedder)(nil)
