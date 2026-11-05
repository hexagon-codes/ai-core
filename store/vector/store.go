// Package vector 提供向量存储抽象
//
// VectorStore 提供向量存储的统一接口，支持多种后端实现（Qdrant、Milvus、Pinecone 等）。
//
// 使用示例:
//
//	store := vector.NewMemoryStore(1536)
//	defer store.Close()
//
//	// 添加文档
//	docs := []vector.Document{
//	    {ID: "1", Content: "Hello", Embedding: embedding},
//	}
//	store.Add(ctx, docs)
//
//	// 搜索相似文档
//	results, _ := store.Search(ctx, queryEmbedding, 10)
package vector

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"sort"
	"sync"
	"time"
)

// Store 向量存储接口
//
// Store 定义了向量存储的基本操作，包括添加、搜索、获取、删除等。
// 不同的后端实现（如 Qdrant、Milvus）需要实现此接口。
type Store interface {
	// Add 添加文档
	//
	// 参数:
	//   - ctx: 上下文
	//   - docs: 要添加的文档列表
	//
	// 返回:
	//   - error: 添加失败时返回错误
	Add(ctx context.Context, docs []Document) error

	// Search 搜索相似文档
	//
	// 参数:
	//   - ctx: 上下文
	//   - query: 查询向量
	//   - k: 返回的最大文档数
	//   - opts: 可选的搜索配置
	//
	// 返回:
	//   - []Document: 相似文档列表，按相似度降序排列
	//   - error: 搜索失败时返回错误
	Search(ctx context.Context, query []float32, k int, opts ...SearchOption) ([]Document, error)

	// Get 根据 ID 获取文档
	//
	// 参数:
	//   - ctx: 上下文
	//   - id: 文档 ID
	//
	// 返回:
	//   - *Document: 找到的文档，不存在则返回 nil
	//   - error: 获取失败时返回错误
	Get(ctx context.Context, id string) (*Document, error)

	// Delete 删除文档
	//
	// 参数:
	//   - ctx: 上下文
	//   - ids: 要删除的文档 ID 列表
	//
	// 返回:
	//   - error: 删除失败时返回错误
	Delete(ctx context.Context, ids []string) error

	// Clear 清空存储
	//
	// 返回:
	//   - error: 清空失败时返回错误
	Clear(ctx context.Context) error

	// Count 返回文档数量
	//
	// 返回:
	//   - int: 文档总数
	//   - error: 获取数量失败时返回错误
	Count(ctx context.Context) (int, error)

	// Close 关闭存储
	//
	// 应在程序退出前调用，释放资源
	Close() error
}

// Document 文档
//
// Document 表示存储在向量数据库中的一个文档，
// 包含内容、向量嵌入和元数据。
type Document struct {
	// ID 文档唯一标识
	ID string `json:"id"`

	// Content 文档内容
	Content string `json:"content"`

	// Embedding 文档向量
	Embedding []float32 `json:"embedding,omitempty"`

	// Metadata 元数据
	Metadata map[string]any `json:"metadata,omitempty"`

	// Score 相似度分数（仅在搜索结果中有效）
	Score float32 `json:"score,omitempty"`

	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at,omitempty"`

	// UpdatedAt 更新时间
	UpdatedAt time.Time `json:"updated_at,omitempty"`
}

// SearchConfig 搜索配置
type SearchConfig struct {
	// Filter 元数据过滤条件
	Filter map[string]any

	// MinScore 最小相似度分数
	MinScore float32

	// IncludeEmbedding 是否返回向量
	IncludeEmbedding bool

	// IncludeMetadata 是否返回元数据
	IncludeMetadata bool
}

// SearchOption 搜索选项
type SearchOption func(*SearchConfig)

// WithFilter 设置过滤条件
func WithFilter(filter map[string]any) SearchOption {
	return func(c *SearchConfig) {
		c.Filter = filter
	}
}

// WithMinScore 设置最小分数
func WithMinScore(score float32) SearchOption {
	return func(c *SearchConfig) {
		c.MinScore = score
	}
}

// WithEmbedding 设置是否返回向量
func WithEmbedding(include bool) SearchOption {
	return func(c *SearchConfig) {
		c.IncludeEmbedding = include
	}
}

// WithMetadata 设置是否返回元数据
func WithMetadata(include bool) SearchOption {
	return func(c *SearchConfig) {
		c.IncludeMetadata = include
	}
}

// ============== 内存实现 ==============

// MemoryStore 内存向量存储
//
// MemoryStore 是一个简单的内存向量存储实现，
// 适用于开发测试和小规模数据场景。
type MemoryStore struct {
	docs      map[string]Document
	mu        sync.RWMutex
	dimension int
}

// NewMemoryStore 创建内存向量存储
//
// 参数:
//   - dimension: 向量维度
//
// 返回:
//   - *MemoryStore: 内存向量存储实例
func NewMemoryStore(dimension int) *MemoryStore {
	return &MemoryStore{
		docs:      make(map[string]Document),
		dimension: dimension,
	}
}

// Add 添加文档
func (s *MemoryStore) Add(ctx context.Context, docs []Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	for _, doc := range docs {
		doc.CreatedAt = now
		doc.UpdatedAt = now
		s.docs[doc.ID] = doc
	}

	return nil
}

// Search 搜索相似文档
func (s *MemoryStore) Search(ctx context.Context, query []float32, k int, opts ...SearchOption) ([]Document, error) {
	if k <= 0 {
		return nil, fmt.Errorf("vector: k must be positive")
	}
	// 检查 context 是否已取消
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	cfg := &SearchConfig{
		IncludeMetadata: true,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// 计算所有文档的相似度
	type scored struct {
		doc   Document
		score float32
	}

	var results []scored
	checkInterval := 1000 // 每 1000 个文档检查一次 context
	count := 0
	for _, doc := range s.docs {
		// 定期检查 context 是否已取消
		count++
		if count%checkInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}

		if len(doc.Embedding) == 0 {
			continue
		}

		// 余弦相似度
		score := cosineSimilarity(query, doc.Embedding)

		if cfg.MinScore > 0 && score < cfg.MinScore {
			continue
		}

		// 检查过滤条件
		if cfg.Filter != nil && !matchFilter(doc.Metadata, cfg.Filter) {
			continue
		}

		doc.Score = score
		if !cfg.IncludeEmbedding {
			doc.Embedding = nil
		}
		if !cfg.IncludeMetadata {
			doc.Metadata = nil
		}

		results = append(results, scored{doc: doc, score: score})
	}

	// 按分数排序（降序）- 使用标准库 O(n log n)
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	// 返回前 k 个
	if k > len(results) {
		k = len(results)
	}

	docs := make([]Document, k)
	for i := 0; i < k; i++ {
		docs[i] = results[i].doc
	}

	return docs, nil
}

// Get 根据 ID 获取文档
func (s *MemoryStore) Get(ctx context.Context, id string) (*Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if doc, ok := s.docs[id]; ok {
		return &doc, nil
	}
	return nil, nil
}

// Delete 删除文档
func (s *MemoryStore) Delete(ctx context.Context, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.docs, id)
	}
	return nil
}

// Clear 清空存储
func (s *MemoryStore) Clear(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.docs = make(map[string]Document)
	return nil
}

// Count 返回文档数量
func (s *MemoryStore) Count(ctx context.Context) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docs), nil
}

// Close 关闭存储
func (s *MemoryStore) Close() error {
	return nil
}

// Dimension 返回向量维度
func (s *MemoryStore) Dimension() int {
	return s.dimension
}

// 确保实现了 Store 接口
var _ Store = (*MemoryStore)(nil)

// ============== 工具函数 ==============

// cosineSimilarity 计算余弦相似度
// 使用 float64 进行中间计算以获得更好的精度
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

// matchFilter 检查元数据是否匹配过滤条件
func matchFilter(metadata, filter map[string]any) bool {
	if metadata == nil {
		return len(filter) == 0
	}

	for k, v := range filter {
		if mv, ok := metadata[k]; !ok || !equalFilterValue(mv, v) {
			return false
		}
	}
	return true
}

// equalFilterValue 比较过滤值，避免对 slice/map 等不可比较类型直接使用
// 接口 != 导致运行时 panic。
func equalFilterValue(a, b any) bool {
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false
	}
	return reflect.DeepEqual(a, b)
}
