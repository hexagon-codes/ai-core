package memory

import (
	"context"
	"testing"
)

func TestBufferMemorySearchMetadataWithCompositeValues(t *testing.T) {
	t.Parallel()

	buffer := NewBuffer(4)
	ctx := context.Background()
	entry := Entry{
		ID:       "entry-1",
		Role:     "user",
		Content:  "hello",
		Metadata: map[string]any{"tags": []string{"a", "b"}},
	}
	if err := buffer.Save(ctx, entry); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	results, err := buffer.Search(ctx, SearchQuery{
		Metadata: map[string]any{"tags": []string{"a", "b"}},
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Search() returned %d entries, want 1", len(results))
	}
}

func TestMatchQueryMetadataDoesNotPanicOnUncomparableValue(t *testing.T) {
	t.Parallel()

	entry := Entry{Metadata: map[string]any{"tags": []string{"a"}}}
	if matchQuery(entry, SearchQuery{Metadata: map[string]any{"tags": []string{"a"}}}) != true {
		t.Fatal("matchQuery() should match equal composite metadata values")
	}
	if matchQuery(entry, SearchQuery{Metadata: map[string]any{"tags": []string{"b"}}}) {
		t.Fatal("matchQuery() should not match different composite metadata values")
	}
}
