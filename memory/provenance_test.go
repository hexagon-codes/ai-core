package memory

import (
	"context"
	"testing"
)

// TestNewDerivedEntry 验证派生条目建立溯源关系：ParentID 指向父、Version+1、记录 CauseBy。
func TestNewDerivedEntry(t *testing.T) {
	parent := Entry{ID: "p1", Role: "assistant", Content: "旧答案", Version: 0}
	d := NewDerivedEntry(parent, "assistant", "修正后的答案", "reflection")

	if d.ParentID != "p1" {
		t.Errorf("ParentID = %q, want p1", d.ParentID)
	}
	if d.Version != 1 {
		t.Errorf("父未版本化(0)时派生 Version = %d, want 1", d.Version)
	}
	if d.CauseBy != "reflection" {
		t.Errorf("CauseBy = %q, want reflection", d.CauseBy)
	}
	if d.CreatedAt.IsZero() {
		t.Error("派生条目应设置 CreatedAt")
	}

	// 已版本化父：Version 累加
	d2 := NewDerivedEntry(Entry{ID: "p2", Version: 3}, "assistant", "x", "tool:edit")
	if d2.Version != 4 {
		t.Errorf("父 Version=3 时派生 Version = %d, want 4", d2.Version)
	}
}

// TestMarkStale 验证标记过时设置 Stale + UpdatedAt，且不影响其它字段。
func TestMarkStale(t *testing.T) {
	e := Entry{ID: "1", Content: "x"}
	if e.IsStale() {
		t.Error("新条目不应为 stale")
	}
	e.MarkStale()
	if !e.IsStale() || !e.Stale {
		t.Error("MarkStale 后应为 stale")
	}
	if e.UpdatedAt.IsZero() {
		t.Error("MarkStale 应更新 UpdatedAt")
	}
}

// TestWithMedia_Modality 验证多模态推断逻辑。
func TestWithMedia_Modality(t *testing.T) {
	img := NewMediaRef(ModalityImage, "https://x/a.png", "image/png")
	aud := NewMediaRef(ModalityAudio, "https://x/a.wav", "audio/wav")

	cases := []struct {
		name string
		in   Entry
		refs []MediaRef
		want string
	}{
		{"文本+图片→multimodal", Entry{Content: "看图"}, []MediaRef{img}, ModalityMultimodal},
		{"仅单图→image", Entry{}, []MediaRef{img}, ModalityImage},
		{"多类型→multimodal", Entry{}, []MediaRef{img, aud}, ModalityMultimodal},
		{"双图同类→image", Entry{}, []MediaRef{img, img}, ModalityImage},
	}
	for _, c := range cases {
		got := c.in.WithMedia(c.refs...)
		if got.Modality != c.want {
			t.Errorf("%s: Modality = %q, want %q", c.name, got.Modality, c.want)
		}
		if len(got.Media) != len(c.refs) {
			t.Errorf("%s: Media 数量 = %d, want %d", c.name, len(got.Media), len(c.refs))
		}
	}
}

// TestFilterActive 验证排除过时条目并保持顺序。
func TestFilterActive(t *testing.T) {
	entries := []Entry{
		{ID: "1"},
		{ID: "2", Stale: true},
		{ID: "3"},
		{ID: "4", Stale: true},
	}
	got := FilterActive(entries)
	if len(got) != 2 || got[0].ID != "1" || got[1].ID != "3" {
		t.Errorf("FilterActive = %v, want [1 3]", got)
	}
}

// TestBuildLineage 验证沿 ParentID 回溯出根→叶链，并处理缺失/环。
func TestBuildLineage(t *testing.T) {
	entries := []Entry{
		{ID: "a", ParentID: ""},
		{ID: "b", ParentID: "a"},
		{ID: "c", ParentID: "b"},
	}
	chain := BuildLineage(entries, "c")
	if len(chain) != 3 || chain[0].ID != "a" || chain[1].ID != "b" || chain[2].ID != "c" {
		t.Fatalf("lineage = %v, want [a b c]", ids(chain))
	}

	if BuildLineage(entries, "missing") != nil {
		t.Error("叶不存在应返回 nil")
	}

	// 防环：x↔y 互为父，不应死循环
	cyclic := []Entry{{ID: "x", ParentID: "y"}, {ID: "y", ParentID: "x"}}
	got := BuildLineage(cyclic, "x")
	if len(got) != 2 {
		t.Errorf("环形 lineage 应终止且去重, got %v", ids(got))
	}
}

// TestSearch_ProvenanceFilters 验证 BufferMemory.Search 支持 ExcludeStale 与 CauseBy 过滤。
func TestSearch_ProvenanceFilters(t *testing.T) {
	ctx := context.Background()
	m := NewBuffer(10)
	_ = m.Save(ctx, Entry{ID: "1", Role: "user", Content: "a"})
	_ = m.Save(ctx, Entry{ID: "2", Role: "user", Content: "b", Stale: true})
	_ = m.Save(ctx, Entry{ID: "3", Role: "user", Content: "c", CauseBy: "reflection"})

	active, _ := m.Search(ctx, SearchQuery{ExcludeStale: true})
	if len(active) != 2 {
		t.Errorf("ExcludeStale 应过滤掉 stale, got %d 条", len(active))
	}
	for _, e := range active {
		if e.Stale {
			t.Error("ExcludeStale 结果不应含 stale 条目")
		}
	}

	byCause, _ := m.Search(ctx, SearchQuery{CauseBy: "reflection"})
	if len(byCause) != 1 || byCause[0].ID != "3" {
		t.Errorf("CauseBy 过滤应只返回条目 3, got %v", ids(byCause))
	}
}

func ids(entries []Entry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.ID
	}
	return out
}
