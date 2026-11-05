package memory

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestEntityMemoryMaxEntitiesIsEnforced(t *testing.T) {
	t.Parallel()

	extractor := &mockEntityExtractor{result: &ExtractionResult{
		Entities: []ExtractedEntity{
			{Name: "one", Type: EntityTypePerson},
			{Name: "two", Type: EntityTypePerson},
			{Name: "three", Type: EntityTypePerson},
		},
	}}
	mem := NewEntityMemory(extractor, WithEntityConfig(EntityConfig{
		BufferCapacity:  10,
		AsyncExtraction: false,
		BatchSize:       1,
		MaxEntities:     2,
	}))

	if err := mem.Save(context.Background(), NewUserEntry("extract entities")); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if stats := mem.EntityStats(); stats.TotalEntities > 2 {
		t.Fatalf("EntityStats().TotalEntities = %d, want <= 2", stats.TotalEntities)
	}
}

func TestEntityMemoryGetEntityReturnsDefensiveCopy(t *testing.T) {
	t.Parallel()

	mem := NewEntityMemory(&mockEntityExtractor{})
	mem.AddEntity(&Entity{
		Name:       "Alice",
		Type:       EntityTypePerson,
		Attributes: map[string]any{"role": "engineer"},
		Relations:  []EntityRelation{{Type: "knows", TargetName: "Bob"}},
		Sources:    []string{"source-1"},
	})
	entity, ok := mem.GetEntity("alice")
	if !ok {
		t.Fatal("GetEntity() did not find the entity")
	}
	entity.Attributes["role"] = "mutated"
	entity.Relations[0].TargetName = "Mallory"
	entity.Sources[0] = "mutated-source"

	again, ok := mem.GetEntity("alice")
	if !ok {
		t.Fatal("GetEntity() did not find the entity on second read")
	}
	if again.Attributes["role"] != "engineer" {
		t.Fatalf("GetEntity() returned shared attributes: %v", again.Attributes)
	}
	if again.Relations[0].TargetName != "Bob" {
		t.Fatalf("GetEntity() returned shared relations: %v", again.Relations)
	}
	if again.Sources[0] != "source-1" {
		t.Fatalf("GetEntity() returned shared sources: %v", again.Sources)
	}
}

func TestEntityMemoryGeneratedIDsAreUsedForSources(t *testing.T) {
	t.Parallel()

	extractor := &mockEntityExtractor{result: &ExtractionResult{
		Entities: []ExtractedEntity{{Name: "Alice", Type: EntityTypePerson}},
	}}
	mem := NewEntityMemory(extractor, WithAsyncExtraction(false), WithBatchSize(1))
	ctx := context.Background()

	if err := mem.Save(ctx, NewUserEntry("Alice is here")); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	entity, ok := mem.GetEntity("Alice")
	if !ok {
		t.Fatal("entity not extracted")
	}
	if len(entity.Sources) != 1 || entity.Sources[0] == "" {
		t.Fatalf("entity Sources = %v, want one generated entry ID", entity.Sources)
	}
	contextEntries, err := mem.GetEntityContext(ctx, "Alice")
	if err != nil {
		t.Fatalf("GetEntityContext() error = %v", err)
	}
	if len(contextEntries) != 1 || contextEntries[0].ID != entity.Sources[0] {
		t.Fatalf("GetEntityContext() = %+v, want source entry", contextEntries)
	}
}

type blockingEntityExtractor struct {
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

func (e *blockingEntityExtractor) Extract(context.Context, string) (*ExtractionResult, error) {
	e.once.Do(func() { close(e.started) })
	<-e.release
	return &ExtractionResult{Entities: []ExtractedEntity{{Name: "stale", Type: EntityTypePerson}}}, nil
}

func TestEntityMemoryClearDiscardsInFlightExtraction(t *testing.T) {
	t.Parallel()

	extractor := &blockingEntityExtractor{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	mem := NewEntityMemory(extractor,
		WithEntityConfig(EntityConfig{
			BufferCapacity:  10,
			AsyncExtraction: true,
			BatchSize:       1,
		}),
	)
	if err := mem.Save(context.Background(), NewUserEntry("extract me")); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	select {
	case <-extractor.started:
	case <-time.After(2 * time.Second):
		t.Fatal("extractor did not start")
	}

	if err := mem.Clear(context.Background()); err != nil {
		t.Fatalf("Clear() error = %v", err)
	}
	close(extractor.release)
	time.Sleep(50 * time.Millisecond)

	if stats := mem.EntityStats(); stats.TotalEntities != 0 {
		t.Fatalf("Clear() was repopulated by stale extraction: %+v", stats)
	}
}
