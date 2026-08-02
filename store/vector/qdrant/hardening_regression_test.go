package qdrant

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/store/vector"
)

func TestPointIDsDoNotCollideForDistinctDocumentIDs(t *testing.T) {
	t.Parallel()

	store := &Store{}
	first := store.toPointID("Aa")
	second := store.toPointID("BB")
	if first == second {
		t.Fatalf("distinct document IDs mapped to the same Qdrant point ID %v", first)
	}
}

func TestLegacyPointIDStrategyIsExplicitForMigrationOnly(t *testing.T) {
	t.Parallel()

	store := &Store{config: Config{PointIDStrategy: PointIDLegacyHash31}}
	if got := store.toPointID("legacy-id"); got != legacyPointID("legacy-id") {
		t.Fatalf("legacy point ID = %v, want migration-compatible ID", got)
	}
}

func TestFilterSerializationOrderIsDeterministic(t *testing.T) {
	t.Parallel()

	store := &Store{}
	filter := map[string]any{"zeta": 1, "alpha": 2, "middle": 3}
	for i := 0; i < 100; i++ {
		built := store.buildFilter(filter)
		conditions, ok := built["must"].([]map[string]any)
		if !ok || len(conditions) != 3 {
			t.Fatalf("unexpected filter shape: %#v", built)
		}
		got := []string{conditions[0]["key"].(string), conditions[1]["key"].(string), conditions[2]["key"].(string)}
		want := []string{"alpha", "middle", "zeta"}
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("filter order = %v, want %v", got, want)
			}
		}
	}
}

func TestMutationsWaitForQdrantCommit(t *testing.T) {
	t.Parallel()

	var missingWait atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("wait") != "true" {
			missingWait.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"result":{"status":"completed"}}`))
	}))
	t.Cleanup(server.Close)

	store := newHTTPTestStore(server)
	doc := vector.Document{ID: "doc", Content: "content", Embedding: []float32{1, 2, 3}}
	if err := store.Add(context.Background(), []vector.Document{doc}); err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if err := store.Delete(context.Background(), []string{"doc"}); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if missingWait.Load() != 0 {
		t.Fatalf("mutations without wait=true = %d, want 0", missingWait.Load())
	}
}

func TestResponsesAreBoundedAndHTTPBodiesAreRedacted(t *testing.T) {
	t.Parallel()

	const secret = "credential-should-not-escape"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(secret))
	}))
	t.Cleanup(server.Close)

	store := newHTTPTestStore(server)
	store.config.MaxResponseBytes = 8
	_, err := store.doRequest(context.Background(), http.MethodGet, "/", nil)
	if !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("doRequest() error = %v, want ErrResponseTooLarge", err)
	}

	store.config.MaxResponseBytes = 1024
	_, err = store.doRequest(context.Background(), http.MethodGet, "/", nil)
	if err == nil {
		t.Fatal("doRequest() error = nil, want HTTP error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("HTTP error leaked response body: %v", err)
	}
}

func TestAddRejectsInvalidIdentityEmbeddingAndReservedMetadata(t *testing.T) {
	t.Parallel()

	store := &Store{config: Config{Dimension: 3}}
	tests := []struct {
		name string
		doc  vector.Document
	}{
		{name: "empty identity", doc: vector.Document{Embedding: []float32{1, 2, 3}}},
		{name: "wrong dimension", doc: vector.Document{ID: "doc", Embedding: []float32{1, 2}}},
		{name: "reserved original identity", doc: vector.Document{ID: "doc", Embedding: []float32{1, 2, 3}, Metadata: map[string]any{"_original_id": "victim"}}},
		{name: "reserved content", doc: vector.Document{ID: "doc", Embedding: []float32{1, 2, 3}, Metadata: map[string]any{"content": "shadow"}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := store.Add(context.Background(), []vector.Document{tt.doc})
			if !errors.Is(err, ErrInvalidDocument) {
				t.Fatalf("Add() error = %v, want ErrInvalidDocument", err)
			}
		})
	}
}

func TestSearchRejectsInvalidLimitAndVectorDimension(t *testing.T) {
	t.Parallel()

	store := &Store{config: Config{Dimension: 3}}
	if _, err := store.Search(context.Background(), []float32{1, 2, 3}, 0); !errors.Is(err, ErrInvalidSearch) {
		t.Fatalf("Search(k=0) error = %v, want ErrInvalidSearch", err)
	}
	if _, err := store.Search(context.Background(), []float32{1, 2}, 1); !errors.Is(err, ErrInvalidSearch) {
		t.Fatalf("Search(wrong dimension) error = %v, want ErrInvalidSearch", err)
	}
}

func TestBatchOperationsRejectUnsafeConfiguration(t *testing.T) {
	t.Parallel()

	store := &Store{}
	tests := []struct {
		name string
		opts []BatchOption
	}{
		{name: "zero batch size", opts: []BatchOption{WithBatchSize(0)}},
		{name: "zero concurrency", opts: []BatchOption{WithConcurrency(0)}},
		{name: "negative retries", opts: []BatchOption{WithRetry(-1, time.Second)}},
		{name: "negative retry delay", opts: []BatchOption{WithRetry(1, -time.Second)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := store.AddBatch(context.Background(), nil, tt.opts...)
			if !errors.Is(err, ErrInvalidBatchConfig) {
				t.Fatalf("AddBatch() error = %v, want ErrInvalidBatchConfig", err)
			}
		})
	}
}

func TestEnsureCollectionDoesNotTurnAuthorizationFailureIntoCreate(t *testing.T) {
	t.Parallel()

	var createCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			createCalls.Add(1)
		}
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	t.Cleanup(server.Close)

	store := newHTTPTestStore(server)
	err := store.ensureCollection(context.Background())
	if err == nil {
		t.Fatal("ensureCollection() error = nil, want authorization failure")
	}
	if createCalls.Load() != 0 {
		t.Fatalf("create calls = %d, want 0 after authorization failure", createCalls.Load())
	}
}

func TestEnsureCollectionCreatesOnlyAfterNotFound(t *testing.T) {
	t.Parallel()

	var createCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			http.Error(w, "missing", http.StatusNotFound)
		case http.MethodPut:
			createCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"result":true}`))
		default:
			http.Error(w, "unexpected method", http.StatusMethodNotAllowed)
		}
	}))
	t.Cleanup(server.Close)

	store := newHTTPTestStore(server)
	if err := store.ensureCollection(context.Background()); err != nil {
		t.Fatalf("ensureCollection() error = %v", err)
	}
	if createCalls.Load() != 1 {
		t.Fatalf("create calls = %d, want 1", createCalls.Load())
	}
}

func TestClearDoesNotHideDeleteFailure(t *testing.T) {
	t.Parallel()

	var followupCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			followupCalls.Add(1)
		}
		http.Error(w, "storage unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)

	store := newHTTPTestStore(server)
	err := store.Clear(context.Background())
	if err == nil {
		t.Fatal("Clear() error = nil, want delete failure")
	}
	if followupCalls.Load() != 0 {
		t.Fatalf("follow-up calls = %d, want 0 after delete failure", followupCalls.Load())
	}
}

func TestNewRejectsInvalidConfigurationBeforeNetworkAccess(t *testing.T) {
	t.Parallel()

	_, err := New(Config{Host: "127.0.0.1/collections", Port: 6333, Collection: "../escape", Dimension: -1})
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("New() error = %v, want ErrInvalidConfig", err)
	}
}

func newHTTPTestStore(server *httptest.Server) *Store {
	return &Store{
		config: Config{
			Collection: "documents",
			Dimension:  3,
			Distance:   DistanceCosine,
		},
		client:  server.Client(),
		baseURL: server.URL,
	}
}
