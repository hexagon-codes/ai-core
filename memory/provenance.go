package memory

import "time"

// 内容模态常量。Entry.Modality 与 MediaRef.Type 共用语义（MediaRef 不取 text）。
const (
	ModalityText       = "text"
	ModalityImage      = "image"
	ModalityAudio      = "audio"
	ModalityVideo      = "video"
	ModalityMultimodal = "multimodal"
)

// MediaRef 是一条非文本内容的引用。
//
// 记忆系统只保存引用与元信息，不内联二进制数据；消费方按需根据 URI 取回原始资源。
type MediaRef struct {
	// Type 媒体类型：image/audio/video（与 Modality 常量一致，不含 text）
	Type string `json:"type"`
	// URI 资源定位，如 https://...、s3://...、file://...、data:...
	URI string `json:"uri"`
	// MIME 媒体 MIME 类型，如 image/png、audio/wav（可选）
	MIME string `json:"mime,omitempty"`
	// Metadata 媒体级元数据（尺寸、时长、采样率等，可选）
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewMediaRef 创建媒体引用。
func NewMediaRef(typ, uri, mime string) MediaRef {
	return MediaRef{Type: typ, URI: uri, MIME: mime}
}

// NewDerivedEntry 基于父条目派生一条新条目，建立溯源（lineage）关系：
//   - ParentID 指向 parent.ID
//   - Version 在父版本基础上 +1（父未版本化时 Version 为 0，派生条目即为 1）
//   - CauseBy 记录产生该条目的动作/事件
//
// 不修改 parent。调用方如需将旧条目标记过时，另行 parent.MarkStale() 并 Save。
func NewDerivedEntry(parent Entry, role, content, causeBy string) Entry {
	return Entry{
		Role:      role,
		Content:   content,
		ParentID:  parent.ID,
		Version:   parent.Version + 1,
		CauseBy:   causeBy,
		CreatedAt: time.Now(),
	}
}

// MarkStale 将条目标记为过时并更新 UpdatedAt。
// 过时条目仍可被检索（保留可追溯性），但可经 SearchQuery.ExcludeStale 或
// FilterActive 排除在上下文组装之外。
func (e *Entry) MarkStale() {
	e.Stale = true
	e.UpdatedAt = time.Now()
}

// IsStale 返回条目是否已标记过时。
func (e Entry) IsStale() bool { return e.Stale }

// WithMedia 追加媒体引用并返回更新后的条目（链式），同时按内容自动推断 Modality。
func (e Entry) WithMedia(refs ...MediaRef) Entry {
	e.Media = append(e.Media, refs...)
	e.Modality = inferModality(e)
	return e
}

// inferModality 根据 Content 与 Media 推断条目模态：
//   - 无媒体 → text
//   - 有媒体且有文本 → multimodal
//   - 仅单一类型媒体、无文本 → 该媒体类型
//   - 多种类型媒体、无文本 → multimodal
func inferModality(e Entry) string {
	if len(e.Media) == 0 {
		return ModalityText
	}
	if e.Content != "" {
		return ModalityMultimodal
	}
	first := e.Media[0].Type
	if first == "" {
		return ModalityMultimodal
	}
	for _, m := range e.Media[1:] {
		if m.Type != first {
			return ModalityMultimodal
		}
	}
	return first
}

// FilterActive 返回非过时（!Stale）的条目，保持原有顺序。
//
// 适用于任意 Memory 实现的检索结果后处理，是 SearchQuery.ExcludeStale 在
// 调用方一侧的等价工具（部分 Memory 实现可能不在 Search 内处理 ExcludeStale）。
func FilterActive(entries []Entry) []Entry {
	out := make([]Entry, 0, len(entries))
	for _, e := range entries {
		if !e.Stale {
			out = append(out, e)
		}
	}
	return out
}

// BuildLineage 在给定条目集合内，从 leafID 沿 ParentID 回溯，返回从根到叶的溯源链。
//
//   - 集合中找不到某个 ParentID 时在该处停止（链可能不完整）；
//   - leafID 不在集合中时返回 nil；
//   - 自带防环保护（ParentID 形成环时提前终止）。
//
// 用于回答"这条记忆是如何一步步演变而来"。
func BuildLineage(entries []Entry, leafID string) []Entry {
	index := make(map[string]Entry, len(entries))
	for _, e := range entries {
		index[e.ID] = e
	}

	cur, ok := index[leafID]
	if !ok {
		return nil
	}

	var chain []Entry
	seen := make(map[string]bool)
	for {
		if seen[cur.ID] { // 防环
			break
		}
		seen[cur.ID] = true
		chain = append(chain, cur)

		if cur.ParentID == "" {
			break
		}
		parent, ok := index[cur.ParentID]
		if !ok {
			break
		}
		cur = parent
	}

	// 反转为根 → 叶
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}
	return chain
}
