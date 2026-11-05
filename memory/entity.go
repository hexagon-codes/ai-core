package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// EntityType 实体类型
type EntityType string

const (
	// EntityTypePerson 人物实体
	EntityTypePerson EntityType = "person"

	// EntityTypePlace 地点实体
	EntityTypePlace EntityType = "place"

	// EntityTypeOrganization 组织实体
	EntityTypeOrganization EntityType = "organization"

	// EntityTypeConcept 概念实体
	EntityTypeConcept EntityType = "concept"

	// EntityTypeEvent 事件实体
	EntityTypeEvent EntityType = "event"

	// EntityTypeProduct 产品实体
	EntityTypeProduct EntityType = "product"

	// EntityTypeOther 其他实体
	EntityTypeOther EntityType = "other"
)

// Entity 实体结构
// 表示对话中提取的实体信息
type Entity struct {
	// Name 实体名称（作为唯一标识）
	Name string `json:"name"`

	// Type 实体类型
	Type EntityType `json:"type"`

	// Description 实体描述
	Description string `json:"description"`

	// Attributes 实体属性（如职位、特征等）
	Attributes map[string]any `json:"attributes,omitempty"`

	// Relations 与其他实体的关系
	Relations []EntityRelation `json:"relations,omitempty"`

	// FirstMentioned 首次提及时间
	FirstMentioned time.Time `json:"first_mentioned"`

	// LastMentioned 最后提及时间
	LastMentioned time.Time `json:"last_mentioned"`

	// MentionCount 提及次数
	MentionCount int `json:"mention_count"`

	// Sources 来源消息 ID 列表
	Sources []string `json:"sources,omitempty"`
}

// EntityRelation 实体关系
type EntityRelation struct {
	// Type 关系类型（如 "works_at", "knows", "located_in" 等）
	Type string `json:"type"`

	// TargetName 目标实体名称
	TargetName string `json:"target_name"`

	// Description 关系描述
	Description string `json:"description,omitempty"`

	// Context 关系上下文（从哪段对话中提取）
	Context string `json:"context,omitempty"`

	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at"`
}

// EntityExtractor 实体提取器接口
// 通常由 LLM Provider 实现
type EntityExtractor interface {
	// Extract 从文本中提取实体和关系
	Extract(ctx context.Context, text string) (*ExtractionResult, error)
}

// ExtractionResult 实体提取结果
type ExtractionResult struct {
	// Entities 提取的实体列表
	Entities []ExtractedEntity `json:"entities"`

	// Relations 提取的关系列表
	Relations []ExtractedRelation `json:"relations"`
}

// ExtractedEntity 提取的实体
type ExtractedEntity struct {
	Name        string         `json:"name"`
	Type        EntityType     `json:"type"`
	Description string         `json:"description"`
	Attributes  map[string]any `json:"attributes,omitempty"`
}

// ExtractedRelation 提取的关系
type ExtractedRelation struct {
	SourceName  string `json:"source_name"`
	TargetName  string `json:"target_name"`
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// EntityMemory 实体记忆
// 自动从对话中提取实体和关系，构建实体知识库
//
// 工作原理：
//  1. 保存消息到底层缓冲区
//  2. 异步调用实体提取器提取实体和关系
//  3. 更新实体库（新增或合并）
//  4. 提供基于实体的上下文检索
//
// 使用示例：
//
//	extractor := memory.NewLLMEntityExtractor(llmProvider)
//	mem := memory.NewEntityMemory(extractor,
//	    memory.WithEntityBufferCapacity(100),
//	)
type EntityMemory struct {
	// entities 实体存储（name -> entity）
	entities map[string]*Entity

	// buffer 底层消息缓冲
	buffer *BufferMemory

	// extractor 实体提取器
	extractor EntityExtractor

	// config 配置
	config EntityConfig

	// mu 保护并发访问
	mu sync.RWMutex

	// extractionQueue 提取队列（批量提取优化）
	extractionQueue []Entry
	queueMu         sync.Mutex

	// asyncCtx/asyncCancel 控制异步提取的生命周期；Close 时取消。
	asyncCtx    context.Context
	asyncCancel context.CancelFunc
	closeOnce   sync.Once

	// generation 在 Clear/Close 时递增，用于丢弃在途提取的过期结果。
	generation uint64
}

// EntityConfig 实体记忆配置
type EntityConfig struct {
	// BufferCapacity 底层缓冲区容量
	BufferCapacity int

	// AsyncExtraction 是否异步提取（不阻塞保存）
	AsyncExtraction bool

	// BatchSize 批量提取大小（多少条消息触发一次提取）
	BatchSize int

	// MaxEntities 最大实体数量
	MaxEntities int

	// MaxRelationsPerEntity 每个实体最大关系数量
	MaxRelationsPerEntity int

	// MaxSourcesPerEntity 每个实体最大来源数量
	MaxSourcesPerEntity int
}

// DefaultEntityConfig 返回默认配置
func DefaultEntityConfig() EntityConfig {
	return EntityConfig{
		BufferCapacity:        200,
		AsyncExtraction:       true,
		BatchSize:             5,
		MaxEntities:           1000,
		MaxRelationsPerEntity: 100,
		MaxSourcesPerEntity:   500,
	}
}

// EntityOption 配置选项
type EntityOption func(*EntityMemory)

// WithEntityConfig 设置实体记忆配置
func WithEntityConfig(config EntityConfig) EntityOption {
	return func(m *EntityMemory) {
		m.config = config
	}
}

// WithEntityBufferCapacity 设置缓冲区容量
func WithEntityBufferCapacity(capacity int) EntityOption {
	return func(m *EntityMemory) {
		m.config.BufferCapacity = capacity
	}
}

// WithAsyncExtraction 设置异步提取
func WithAsyncExtraction(async bool) EntityOption {
	return func(m *EntityMemory) {
		m.config.AsyncExtraction = async
	}
}

// WithBatchSize 设置批量提取大小
func WithBatchSize(size int) EntityOption {
	return func(m *EntityMemory) {
		m.config.BatchSize = size
	}
}

// NewEntityMemory 创建实体记忆
//
// 参数：
//   - extractor: 实体提取器，通常由 LLM 实现
//   - opts: 配置选项
func NewEntityMemory(extractor EntityExtractor, opts ...EntityOption) *EntityMemory {
	m := &EntityMemory{
		entities:        make(map[string]*Entity),
		extractor:       extractor,
		config:          DefaultEntityConfig(),
		extractionQueue: make([]Entry, 0),
	}

	for _, opt := range opts {
		opt(m)
	}

	m.buffer = NewBuffer(m.config.BufferCapacity)
	m.asyncCtx, m.asyncCancel = context.WithCancel(context.Background())
	return m
}

// Save 保存记忆条目
// 保存后会触发实体提取（同步或异步）
func (m *EntityMemory) Save(ctx context.Context, entry Entry) error {
	if entry.ID == "" {
		entry.ID = defaultIDGen()
	}
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now()
	}

	// 保存到缓冲区
	m.mu.Lock()
	generation := m.generation
	err := m.buffer.Save(ctx, entry)
	m.mu.Unlock()
	if err != nil {
		return err
	}

	// 添加到提取队列
	m.queueMu.Lock()
	m.extractionQueue = append(m.extractionQueue, entry)
	shouldExtract := len(m.extractionQueue) >= m.config.BatchSize
	var entriesToExtract []Entry
	if shouldExtract {
		entriesToExtract = make([]Entry, len(m.extractionQueue))
		copy(entriesToExtract, m.extractionQueue)
		m.extractionQueue = m.extractionQueue[:0]
	}
	m.queueMu.Unlock()

	// 触发提取
	if shouldExtract && len(entriesToExtract) > 0 {
		if m.config.AsyncExtraction {
			ctx := m.asyncCtx
			go m.extractEntities(ctx, entriesToExtract, generation)
		} else {
			if err := m.extractEntities(ctx, entriesToExtract, generation); err != nil {
				// 提取失败不影响保存
				_ = err
			}
		}
	}

	return nil
}

// Close 取消仍在进行的异步实体提取。多次调用安全。
func (m *EntityMemory) Close() {
	if m == nil {
		return
	}
	m.closeOnce.Do(func() {
		m.mu.Lock()
		m.generation++
		m.mu.Unlock()
		if m.asyncCancel != nil {
			m.asyncCancel()
		}
	})
}

// SaveBatch 批量保存记忆条目
func (m *EntityMemory) SaveBatch(ctx context.Context, entries []Entry) error {
	for _, entry := range entries {
		if err := m.Save(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

// Get 根据 ID 获取记忆条目
func (m *EntityMemory) Get(ctx context.Context, id string) (*Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.buffer.Get(ctx, id)
}

// Search 搜索记忆条目
func (m *EntityMemory) Search(ctx context.Context, query SearchQuery) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.buffer.Search(ctx, query)
}

// Delete 删除指定 ID 的记忆条目
func (m *EntityMemory) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.buffer.Delete(ctx, id)
}

// Clear 清空所有记忆（包括实体）
func (m *EntityMemory) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.generation++
	m.entities = make(map[string]*Entity)
	m.queueMu.Lock()
	m.extractionQueue = m.extractionQueue[:0]
	m.queueMu.Unlock()
	return m.buffer.Clear(ctx)
}

// Stats 返回记忆统计信息
func (m *EntityMemory) Stats() MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.buffer.Stats()
}

// EntityStats 返回实体统计信息
func (m *EntityMemory) EntityStats() EntityMemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := EntityMemoryStats{
		TotalEntities: len(m.entities),
		TypeCounts:    make(map[EntityType]int),
	}

	var totalRelations int
	for _, entity := range m.entities {
		stats.TypeCounts[entity.Type]++
		totalRelations += len(entity.Relations)
	}
	stats.TotalRelations = totalRelations

	return stats
}

// EntityMemoryStats 实体记忆统计
type EntityMemoryStats struct {
	// TotalEntities 实体总数
	TotalEntities int `json:"total_entities"`

	// TotalRelations 关系总数
	TotalRelations int `json:"total_relations"`

	// TypeCounts 各类型实体数量
	TypeCounts map[EntityType]int `json:"type_counts"`
}

// GetEntity 获取指定名称的实体
func (m *EntityMemory) GetEntity(name string) (*Entity, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	entity, ok := m.entities[normalizeEntityName(name)]
	return cloneEntity(entity), ok
}

// GetEntities 获取所有实体
func (m *EntityMemory) GetEntities() []*Entity {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entities := make([]*Entity, 0, len(m.entities))
	for _, entity := range m.entities {
		entities = append(entities, cloneEntity(entity))
	}
	return entities
}

// SearchEntities 按名称模糊搜索实体
func (m *EntityMemory) SearchEntities(query string) []*Entity {
	m.mu.RLock()
	defer m.mu.RUnlock()

	query = strings.ToLower(query)
	var results []*Entity

	for _, entity := range m.entities {
		if strings.Contains(strings.ToLower(entity.Name), query) ||
			strings.Contains(strings.ToLower(entity.Description), query) {
			results = append(results, cloneEntity(entity))
		}
	}

	return results
}

// GetEntitiesByType 按类型获取实体
func (m *EntityMemory) GetEntitiesByType(entityType EntityType) []*Entity {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []*Entity
	for _, entity := range m.entities {
		if entity.Type == entityType {
			results = append(results, cloneEntity(entity))
		}
	}
	return results
}

// GetRelatedEntities 获取与指定实体相关的其他实体
func (m *EntityMemory) GetRelatedEntities(name string) []*Entity {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entity, ok := m.entities[normalizeEntityName(name)]
	if !ok {
		return nil
	}

	var related []*Entity
	seen := make(map[string]bool)
	seen[normalizeEntityName(name)] = true

	for _, rel := range entity.Relations {
		normalizedTarget := normalizeEntityName(rel.TargetName)
		if !seen[normalizedTarget] {
			if target, ok := m.entities[normalizedTarget]; ok {
				related = append(related, cloneEntity(target))
				seen[normalizedTarget] = true
			}
		}
	}

	return related
}

// GetEntityContext 获取实体相关的上下文
// 返回包含指定实体的消息
func (m *EntityMemory) GetEntityContext(ctx context.Context, entityNames ...string) ([]Entry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 收集所有相关的消息 ID
	sourceIDs := make(map[string]bool)
	for _, name := range entityNames {
		if entity, ok := m.entities[normalizeEntityName(name)]; ok {
			for _, srcID := range entity.Sources {
				sourceIDs[srcID] = true
			}
		}
	}

	// 获取相关消息
	var entries []Entry
	allEntries := m.buffer.Entries()
	for _, entry := range allEntries {
		if sourceIDs[entry.ID] {
			entries = append(entries, entry)
		}
	}

	return entries, nil
}

// GetContextWithEntities 获取包含实体信息的上下文
// 返回摘要形式的实体知识
func (m *EntityMemory) GetContextWithEntities(entityNames ...string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var sb strings.Builder
	sb.WriteString("已知实体信息：\n")

	for _, name := range entityNames {
		if entity, ok := m.entities[normalizeEntityName(name)]; ok {
			sb.WriteString(fmt.Sprintf("\n【%s】(%s)\n", entity.Name, entity.Type))
			if entity.Description != "" {
				sb.WriteString(fmt.Sprintf("  描述: %s\n", entity.Description))
			}

			// 添加属性
			if len(entity.Attributes) > 0 {
				sb.WriteString("  属性: ")
				var attrs []string
				for k, v := range entity.Attributes {
					attrs = append(attrs, fmt.Sprintf("%s=%v", k, v))
				}
				sb.WriteString(strings.Join(attrs, ", "))
				sb.WriteString("\n")
			}

			// 添加关系
			if len(entity.Relations) > 0 {
				sb.WriteString("  关系:\n")
				for _, rel := range entity.Relations {
					sb.WriteString(fmt.Sprintf("    - %s %s\n", rel.Type, rel.TargetName))
				}
			}
		}
	}

	return sb.String()
}

// extractEntities 提取实体
func (m *EntityMemory) extractEntities(ctx context.Context, entries []Entry, generation uint64) error {
	if m.extractor == nil {
		return nil
	}

	// 构建待提取文本
	var sb strings.Builder
	entryIDs := make([]string, 0, len(entries))
	for _, entry := range entries {
		sb.WriteString(fmt.Sprintf("[%s] %s\n", entry.Role, entry.Content))
		entryIDs = append(entryIDs, entry.ID)
	}

	// 调用提取器
	result, err := m.extractor.Extract(ctx, sb.String())
	if err != nil {
		return fmt.Errorf("提取实体失败: %w", err)
	}

	if result == nil {
		return nil
	}

	// 更新实体库
	m.mu.Lock()
	defer m.mu.Unlock()

	// Clear/Close 已发生过：丢弃在途提取结果，避免清空后重新填充。
	if generation != m.generation {
		return nil
	}

	now := time.Now()

	// 处理提取的实体
	for _, extracted := range result.Entities {
		normalizedName := normalizeEntityName(extracted.Name)
		if normalizedName == "" {
			continue
		}

		if existing, ok := m.entities[normalizedName]; ok {
			// 更新已有实体
			existing.LastMentioned = now
			existing.MentionCount++
			if extracted.Description != "" && len(extracted.Description) > len(existing.Description) {
				existing.Description = extracted.Description
			}
			// 合并属性
			for k, v := range extracted.Attributes {
				existing.Attributes[k] = v
			}
			// 添加来源（限制最大数量，保留最新的）
			existing.Sources = append(existing.Sources, entryIDs...)
			if m.config.MaxSourcesPerEntity > 0 && len(existing.Sources) > m.config.MaxSourcesPerEntity {
				existing.Sources = existing.Sources[len(existing.Sources)-m.config.MaxSourcesPerEntity:]
			}
		} else {
			// 创建新实体
			if m.config.MaxEntities > 0 && len(m.entities) >= m.config.MaxEntities {
				continue
			}
			m.entities[normalizedName] = &Entity{
				Name:           extracted.Name,
				Type:           extracted.Type,
				Description:    extracted.Description,
				Attributes:     extracted.Attributes,
				Relations:      make([]EntityRelation, 0),
				FirstMentioned: now,
				LastMentioned:  now,
				MentionCount:   1,
				Sources:        entryIDs,
			}
			if m.entities[normalizedName].Attributes == nil {
				m.entities[normalizedName].Attributes = make(map[string]any)
			}
		}
	}

	// 处理提取的关系
	for _, rel := range result.Relations {
		sourceName := normalizeEntityName(rel.SourceName)
		if entity, ok := m.entities[sourceName]; ok {
			// 检查关系数量限制
			if m.config.MaxRelationsPerEntity > 0 && len(entity.Relations) >= m.config.MaxRelationsPerEntity {
				continue
			}
			// 检查关系是否已存在
			exists := false
			for _, existingRel := range entity.Relations {
				if existingRel.Type == rel.Type &&
					normalizeEntityName(existingRel.TargetName) == normalizeEntityName(rel.TargetName) {
					exists = true
					break
				}
			}
			if !exists {
				entity.Relations = append(entity.Relations, EntityRelation{
					Type:        rel.Type,
					TargetName:  rel.TargetName,
					Description: rel.Description,
					CreatedAt:   now,
				})
			}
		}
	}

	return nil
}

// FlushExtractionQueue 刷新提取队列（立即处理）
func (m *EntityMemory) FlushExtractionQueue(ctx context.Context) error {
	m.mu.RLock()
	generation := m.generation
	m.mu.RUnlock()

	m.queueMu.Lock()
	if len(m.extractionQueue) == 0 {
		m.queueMu.Unlock()
		return nil
	}
	entriesToExtract := make([]Entry, len(m.extractionQueue))
	copy(entriesToExtract, m.extractionQueue)
	m.extractionQueue = m.extractionQueue[:0]
	m.queueMu.Unlock()

	return m.extractEntities(ctx, entriesToExtract, generation)
}

// AddEntity 手动添加实体
func (m *EntityMemory) AddEntity(entity *Entity) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if entity == nil {
		return
	}
	normalizedName := normalizeEntityName(entity.Name)
	if normalizedName == "" {
		return
	}
	if _, exists := m.entities[normalizedName]; !exists &&
		m.config.MaxEntities > 0 && len(m.entities) >= m.config.MaxEntities {
		return
	}

	now := time.Now()
	if entity.FirstMentioned.IsZero() {
		entity.FirstMentioned = now
	}
	if entity.LastMentioned.IsZero() {
		entity.LastMentioned = now
	}
	if entity.Attributes == nil {
		entity.Attributes = make(map[string]any)
	}
	if entity.Relations == nil {
		entity.Relations = make([]EntityRelation, 0)
	}

	m.entities[normalizedName] = cloneEntity(entity)
}

// AddRelation 手动添加实体关系
func (m *EntityMemory) AddRelation(sourceName, targetName, relationType, description string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	normalizedSource := normalizeEntityName(sourceName)
	entity, ok := m.entities[normalizedSource]
	if !ok {
		return fmt.Errorf("源实体 %s 不存在", sourceName)
	}
	if m.config.MaxRelationsPerEntity > 0 && len(entity.Relations) >= m.config.MaxRelationsPerEntity {
		return fmt.Errorf("实体 %s 关系数量已达到上限", sourceName)
	}

	entity.Relations = append(entity.Relations, EntityRelation{
		Type:        relationType,
		TargetName:  targetName,
		Description: description,
		CreatedAt:   time.Now(),
	})

	return nil
}

// normalizeEntityName 规范化实体名称
func normalizeEntityName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

// cloneEntity 返回实体的独立副本，避免调用方绕过锁修改内部状态。
func cloneEntity(entity *Entity) *Entity {
	if entity == nil {
		return nil
	}
	cloned := *entity
	if entity.Attributes != nil {
		cloned.Attributes = make(map[string]any, len(entity.Attributes))
		for key, value := range entity.Attributes {
			cloned.Attributes[key] = value
		}
	}
	cloned.Relations = append([]EntityRelation(nil), entity.Relations...)
	cloned.Sources = append([]string(nil), entity.Sources...)
	return &cloned
}

// ============== LLM 实体提取器 ==============

// LLMEntityExtractor 基于 LLM 的实体提取器
type LLMEntityExtractor struct {
	// complete 是 LLM 完成函数
	complete func(ctx context.Context, prompt string) (string, error)

	// promptTemplate 提取提示词模板
	promptTemplate string
}

// LLMEntityExtractorOption 配置选项
type LLMEntityExtractorOption func(*LLMEntityExtractor)

// WithEntityExtractionPrompt 设置自定义提取提示词
func WithEntityExtractionPrompt(prompt string) LLMEntityExtractorOption {
	return func(e *LLMEntityExtractor) {
		e.promptTemplate = prompt
	}
}

// 默认实体提取提示词
const defaultEntityExtractionPrompt = `请从以下对话内容中提取实体和关系。

对话内容：
%s

请提取：
1. 人物（姓名、职位、特征）
2. 地点（名称、类型）
3. 组织（名称、类型）
4. 概念/术语
5. 产品/服务
6. 事件
7. 实体之间的关系

请以 JSON 格式输出，格式如下：
{
  "entities": [
    {
      "name": "实体名称",
      "type": "person|place|organization|concept|event|product|other",
      "description": "实体描述",
      "attributes": {"key": "value"}
    }
  ],
  "relations": [
    {
      "source_name": "源实体名称",
      "target_name": "目标实体名称",
      "type": "关系类型（如 works_at, knows, located_in, belongs_to 等）",
      "description": "关系描述"
    }
  ]
}

只输出 JSON，不要其他内容。如果没有提取到实体，返回空数组。`

// NewLLMEntityExtractor 创建 LLM 实体提取器
//
// 参数：
//   - complete: LLM 完成函数，接收提示词返回响应
//   - opts: 配置选项
func NewLLMEntityExtractor(
	complete func(ctx context.Context, prompt string) (string, error),
	opts ...LLMEntityExtractorOption,
) *LLMEntityExtractor {
	e := &LLMEntityExtractor{
		complete:       complete,
		promptTemplate: defaultEntityExtractionPrompt,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Extract 从文本中提取实体和关系
func (e *LLMEntityExtractor) Extract(ctx context.Context, text string) (*ExtractionResult, error) {
	if e.complete == nil {
		return nil, fmt.Errorf("LLM complete function not set")
	}

	prompt := fmt.Sprintf(e.promptTemplate, text)
	response, err := e.complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM 调用失败: %w", err)
	}

	// 解析 JSON 响应
	// 尝试提取 JSON 部分（LLM 可能会添加额外文本）
	jsonStr := extractJSON(response)
	if jsonStr == "" {
		return &ExtractionResult{}, nil
	}

	var result ExtractionResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		// 解析失败返回空结果，不报错
		return &ExtractionResult{}, nil
	}

	return &result, nil
}

// extractJSON 从响应中提取 JSON 部分
func extractJSON(text string) string {
	// 查找第一个 { 和最后一个 }
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start == -1 || end == -1 || end <= start {
		return ""
	}
	return text[start : end+1]
}

// ============== 简单实体提取器 ==============

// SimpleEntityExtractor 简单的基于函数的实体提取器
type SimpleEntityExtractor struct {
	fn func(ctx context.Context, text string) (*ExtractionResult, error)
}

// NewSimpleEntityExtractor 创建简单实体提取器
func NewSimpleEntityExtractor(
	fn func(ctx context.Context, text string) (*ExtractionResult, error),
) *SimpleEntityExtractor {
	return &SimpleEntityExtractor{fn: fn}
}

// Extract 提取实体
func (e *SimpleEntityExtractor) Extract(ctx context.Context, text string) (*ExtractionResult, error) {
	return e.fn(ctx, text)
}

// 确保实现了接口
var _ Memory = (*EntityMemory)(nil)
var _ EntityExtractor = (*LLMEntityExtractor)(nil)
var _ EntityExtractor = (*SimpleEntityExtractor)(nil)
