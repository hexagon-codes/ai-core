package vector

import (
	"context"
	"testing"
)

func TestMemoryStoreSearchFilterWithCompositeValues(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore(2)
	ctx := context.Background()
	if err := store.Add(ctx, []Document{{
		ID:        "doc-1",
		Embedding: []float32{1, 0},
		Metadata:  map[string]any{"tags": []string{"a", "b"}},
	}}); err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	results, err := store.Search(ctx, []float32{1, 0}, 1, WithFilter(map[string]any{"tags": []string{"a", "b"}}))
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Search() returned %d documents, want 1", len(results))
	}

	results, err = store.Search(ctx, []float32{1, 0}, 1, WithFilter(map[string]any{"tags": []string{"other"}}))
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("Search() returned %d documents for non-matching filter, want 0", len(results))
	}
}
