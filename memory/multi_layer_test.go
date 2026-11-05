package memory

import (
	"context"
	"testing"
)

func TestMultiLayerMemory_Save(t *testing.T) {
	mem := NewMultiLayerMemory()

	ctx := context.Background()

	// 保存到工作记忆
	err := mem.Save(ctx, NewUserEntry("Test message"))
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}

	stats := mem.MultiStats()
	if stats.WorkingCount != 1 {
		t.Errorf("expected 1 working entry, got %d", stats.WorkingCount)
	}
}

func TestMultiLayerMemory_Get(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	entry := NewUserEntry("Test")
	entry.ID = "test-id"
	mem.Save(ctx, entry)

	found, err := mem.Get(ctx, "test-id")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if found == nil {
		t.Fatal("expected to find entry")
	}

	if found.Content != "Test" {
		t.Errorf("expected 'Test', got '%s'", found.Content)
	}
}

func TestMultiLayerMemory_Search(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("Hello"))
	mem.Save(ctx, NewAssistantEntry("Hi"))

	entries, err := mem.Search(ctx, SearchQuery{Roles: []string{"user"}})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(entries) != 1 {
		t.Errorf("expected 1 entry, got %d", len(entries))
	}
}

func TestMultiLayerMemory_SearchLayer(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("Working memory entry"))

	entries, err := mem.SearchLayer(ctx, LayerWorking, SearchQuery{})
	if err != nil {
		t.Fatalf("search layer failed: %v", err)
	}

	if len(entries) != 1 {
		t.Errorf("expected 1 entry, got %d", len(entries))
	}
}

func TestMultiLayerMemory_Clear(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("Test"))
	mem.Clear(ctx)

	stats := mem.Stats()
	if stats.EntryCount != 0 {
		t.Errorf("expected 0 entries after clear, got %d", stats.EntryCount)
	}
}

func TestMultiLayerMemory_ClearLayer(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("Test"))
	mem.ClearLayer(ctx, LayerWorking)

	stats := mem.MultiStats()
	if stats.WorkingCount != 0 {
		t.Errorf("expected 0 working entries, got %d", stats.WorkingCount)
	}
}

func TestMultiLayerMemory_GetWorkingMemory(t *testing.T) {
	mem := NewMultiLayerMemory()
	ctx := context.Background()

	mem.Save(ctx, NewUserEntry("Entry 1"))
	mem.Save(ctx, NewUserEntry("Entry 2"))

	entries := mem.GetWorkingMemory()
	if len(entries) != 2 {
		t.Errorf("expected 2 entries, got %d", len(entries))
	}
}

func TestMultiLayerMemory_WithSummarizer(t *testing.T) {
	summarizer := &mockSummarizer{}
	mem := NewMultiLayerMemory(WithSummarizer(summarizer))

	ctx := context.Background()

	// 保存多个条目
	for i := 0; i < 5; i++ {
		mem.Save(ctx, NewUserEntry("Test message"))
	}

	stats := mem.MultiStats()
	if stats.WorkingCount == 0 {
		t.Error("expected some working entries")
	}
}

func TestMultiLayerMemory_WithEmbedder(t *testing.T) {
	embedder := newMockEmbedder(8)
	mem := NewMultiLayerMemory(WithEmbedder(embedder))

	ctx := context.Background()

	// 直接保存到长期记忆
	err := mem.SaveToLongTerm(ctx, NewUserEntry("Long term entry"))
	if err != nil {
		t.Fatalf("save to long term failed: %v", err)
	}

	stats := mem.MultiStats()
	if stats.LongTermCount != 1 {
		t.Errorf("expected 1 long term entry, got %d", stats.LongTermCount)
	}
}
