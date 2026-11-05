package memory

import (
	"context"
	"errors"
	"testing"
	"time"
)

type failingVectorStore struct {
	addErr    error
	deleteErr error
	searchErr error
}

func (s *failingVectorStore) Add(context.Context, string, []float32, map[string]any) error {
	return s.addErr
}
func (s *failingVectorStore) Search(context.Context, []float32, int) ([]VectorResult, error) {
	return nil, s.searchErr
}
func (s *failingVectorStore) Delete(context.Context, string) error {
	return s.deleteErr
}
func (s *failingVectorStore) Clear(context.Context) error { return nil }
func (s *failingVectorStore) Count(context.Context) (int, error) {
	return 0, nil
}

func TestVectorMemorySaveDoesNotPublishEntryWhenStoreAddFails(t *testing.T) {
	t.Parallel()

	mem := NewVectorMemory(newMockEmbedder(8), WithVectorStore(&failingVectorStore{
		addErr: errors.New("store unavailable"),
	}))

	err := mem.Save(context.Background(), Entry{ID: "entry-1", Content: "hello"})
	if err == nil {
		t.Fatal("Save() error = nil, want store failure")
	}
	if _, getErr := mem.Get(context.Background(), "entry-1"); !errors.Is(getErr, ErrNotFound) {
		t.Fatalf("Get() error = %v, want ErrNotFound", getErr)
	}
	if stats := mem.Stats(); stats.EntryCount != 0 {
		t.Fatalf("Stats().EntryCount = %d, want 0", stats.EntryCount)
	}
}

func TestVectorMemoryCapacityEvictsOldestEntry(t *testing.T) {
	t.Parallel()

	mem := NewVectorMemory(
		newMockEmbedder(8),
		WithVectorConfig(VectorConfig{
			Dimension:   8,
			Capacity:    2,
			MinScore:    0,
			DefaultTopK: 10,
		}),
	)

	ctx := context.Background()
	for _, id := range []string{"one", "two", "three"} {
		if err := mem.Save(ctx, Entry{ID: id, Content: id}); err != nil {
			t.Fatalf("Save(%q) error = %v", id, err)
		}
	}

	if stats := mem.Stats(); stats.EntryCount != 2 {
		t.Fatalf("Stats().EntryCount = %d, want 2", stats.EntryCount)
	}
	if _, err := mem.Get(ctx, "one"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get(oldest) error = %v, want ErrNotFound", err)
	}
	if _, err := mem.Get(ctx, "three"); err != nil {
		t.Fatalf("Get(newest) error = %v", err)
	}
}

func TestVectorMemoryGetReturnsDefensiveCopy(t *testing.T) {
	t.Parallel()

	mem := NewVectorMemory(newMockEmbedder(8))
	ctx := context.Background()
	if err := mem.Save(ctx, Entry{
		ID:       "entry-1",
		Content:  "hello",
		Metadata: map[string]any{"key": "value"},
	}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	first, err := mem.Get(ctx, "entry-1")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	first.Metadata["key"] = "mutated"

	second, err := mem.Get(ctx, "entry-1")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if second.Metadata["key"] != "value" {
		t.Fatalf("Get() returned shared metadata: %v", second.Metadata)
	}
}

func TestVectorMemorySearchAppliesOffsetInVectorBranch(t *testing.T) {
	t.Parallel()

	mem := NewVectorMemory(newMockEmbedder(2), WithMinScore(-1))
	ctx := context.Background()
	entries := []Entry{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0.9, 0.1}},
		{ID: "c", Embedding: []float32{0, 1}},
	}
	for _, entry := range entries {
		if err := mem.Save(ctx, entry); err != nil {
			t.Fatalf("Save(%q) error = %v", entry.ID, err)
		}
	}

	results, err := mem.Search(ctx, SearchQuery{
		Embedding: []float32{1, 0},
		Limit:     1,
		Offset:    1,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 || results[0].ID != "b" {
		t.Fatalf("Search() = %v, want [b]", entriesIDs(results))
	}
}

func TestVectorMemorySearchHonorsCreatedAtOrder(t *testing.T) {
	t.Parallel()

	mem := NewVectorMemory(newMockEmbedder(2), WithMinScore(-1))
	ctx := context.Background()
	base := time.Now().Add(-time.Hour)
	entries := []Entry{
		{ID: "later", Embedding: []float32{1, 0}, CreatedAt: base.Add(2 * time.Minute)},
		{ID: "earlier", Embedding: []float32{1, 0}, CreatedAt: base},
		{ID: "middle", Embedding: []float32{1, 0}, CreatedAt: base.Add(time.Minute)},
	}
	for _, entry := range entries {
		if err := mem.Save(ctx, entry); err != nil {
			t.Fatalf("Save(%q) error = %v", entry.ID, err)
		}
	}

	results, err := mem.Search(ctx, SearchQuery{
		Embedding: []float32{1, 0},
		Limit:     3,
		OrderBy:   "created_at",
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if got := entriesIDs(results); len(got) != 3 || got[0] != "earlier" || got[1] != "middle" || got[2] != "later" {
		t.Fatalf("Search() order = %v, want [earlier middle later]", got)
	}
}

func entriesIDs(entries []Entry) []string {
	ids := make([]string, len(entries))
	for i, entry := range entries {
		ids[i] = entry.ID
	}
	return ids
}
