package memory

import (
	"context"
	"testing"
)

func TestBufferMemorySaveUpdatesSameID(t *testing.T) {
	t.Parallel()

	buffer := NewBuffer(4)
	ctx := context.Background()
	if err := buffer.Save(ctx, Entry{ID: "same", Content: "old"}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if err := buffer.Save(ctx, Entry{ID: "same", Content: "new"}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	entry, err := buffer.Get(ctx, "same")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if entry.Content != "new" {
		t.Fatalf("Get() = %q, want new", entry.Content)
	}
	if stats := buffer.Stats(); stats.EntryCount != 1 {
		t.Fatalf("EntryCount = %d, want 1", stats.EntryCount)
	}
}

func TestBufferMemoryReadsReturnCopies(t *testing.T) {
	t.Parallel()

	buffer := NewBuffer(4)
	ctx := context.Background()
	if err := buffer.Save(ctx, Entry{
		ID:        "1",
		Content:   "x",
		Metadata:  map[string]any{"k": "v"},
		Embedding: []float32{1},
	}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	entry, err := buffer.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	entry.Metadata["k"] = "mutated"
	entry.Embedding[0] = 9

	again, err := buffer.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if again.Metadata["k"] != "v" || again.Embedding[0] != 1 {
		t.Fatalf("Get() returned shared data: %+v", again)
	}
}

func TestBufferMemoryLastRejectsNegative(t *testing.T) {
	t.Parallel()

	buffer := NewBuffer(4)
	if got := buffer.Last(-1); got != nil {
		t.Fatalf("Last(-1) = %v, want nil", got)
	}
}
