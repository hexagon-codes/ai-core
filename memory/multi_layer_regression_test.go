package memory

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestMultiLayerMemorySearchReportsLayerFailure(t *testing.T) {
	t.Parallel()

	mem := NewMultiLayerMemory(
		WithEmbedder(newMockEmbedder(8)),
		WithLongTermStore(&failingVectorStore{searchErr: errors.New("vector store down")}),
	)
	_, err := mem.Search(context.Background(), SearchQuery{Query: "hello", Limit: 5})
	if err == nil || !strings.Contains(err.Error(), "vector store down") {
		t.Fatalf("Search() error = %v, want long-term layer failure", err)
	}
}
