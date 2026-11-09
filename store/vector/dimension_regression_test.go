package vector

import (
	"context"
	"strings"
	"testing"
)

func TestMemoryStoreRejectsMismatchedEmbeddingDimension(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore(3)
	ctx := context.Background()

	err := store.Add(ctx, []Document{{ID: "bad", Embedding: []float32{1, 2}}})
	if err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("Add() error = %v, want dimension error", err)
	}
	count, err := store.Count(ctx)
	if err != nil {
		t.Fatalf("Count() error = %v", err)
	}
	if count != 0 {
		t.Fatalf("Count() = %d, want 0 after rejected document", count)
	}

	if err := store.Add(ctx, []Document{{ID: "good", Embedding: []float32{1, 2, 3}}}); err != nil {
		t.Fatalf("Add(valid) error = %v", err)
	}
	if _, err := store.Search(ctx, []float32{1, 2}, 1); err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("Search() error = %v, want dimension error", err)
	}
	results, err := store.Search(ctx, []float32{1, 2, 3}, 1)
	if err != nil {
		t.Fatalf("Search(valid) error = %v", err)
	}
	if len(results) != 1 || results[0].ID != "good" {
		t.Fatalf("Search(valid) = %+v, want good document", results)
	}
}
