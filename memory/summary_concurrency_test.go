package memory

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type blockingSummarizer struct {
	once    sync.Once
	started chan struct{}
	release chan struct{}
}

func (s *blockingSummarizer) Summarize(context.Context, string) (string, error) {
	s.once.Do(func() { close(s.started) })
	<-s.release
	return "summary", nil
}

func TestSummaryMemoryKeepsEntrySavedDuringSummarize(t *testing.T) {
	t.Parallel()

	summarizer := &blockingSummarizer{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	mem := NewSummaryMemory(summarizer, WithMaxEntries(2), WithKeepRecent(1))
	ctx := context.Background()

	if err := mem.Save(ctx, Entry{ID: "one", Role: "user", Content: "one"}); err != nil {
		t.Fatalf("Save(one) error = %v", err)
	}
	if err := mem.Save(ctx, Entry{ID: "two", Role: "user", Content: "two"}); err != nil {
		t.Fatalf("Save(two) error = %v", err)
	}

	saveDone := make(chan error, 1)
	go func() {
		saveDone <- mem.Save(ctx, Entry{ID: "three", Role: "user", Content: "three"})
	}()

	select {
	case <-summarizer.started:
	case <-time.After(2 * time.Second):
		t.Fatal("summarizer did not start")
	}

	fourDone := make(chan error, 1)
	go func() {
		fourDone <- mem.Save(ctx, Entry{ID: "four", Role: "user", Content: "four"})
	}()
	// 给并发 Save 一个进入 buffer 的机会；无论它落在摘要前还是摘要后，
	// 最终都不应丢失。
	time.Sleep(20 * time.Millisecond)

	close(summarizer.release)
	if err := <-saveDone; err != nil {
		t.Fatalf("Save(three) error = %v", err)
	}
	if err := <-fourDone; err != nil {
		t.Fatalf("Save(four) error = %v", err)
	}

	if _, err := mem.Get(ctx, "four"); err != nil {
		t.Fatalf("entry saved during summarize was lost: %v", err)
	}
}

func TestSummaryMemoryDeleteDuringSummarizeIsNotResurrected(t *testing.T) {
	t.Parallel()

	summarizer := &blockingSummarizer{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	mem := NewSummaryMemory(summarizer, WithMaxEntries(2), WithKeepRecent(1))
	ctx := context.Background()
	for _, id := range []string{"one", "two"} {
		if err := mem.Save(ctx, Entry{ID: id, Role: "user", Content: id}); err != nil {
			t.Fatalf("Save(%s) error = %v", id, err)
		}
	}

	saveDone := make(chan error, 1)
	go func() {
		saveDone <- mem.Save(ctx, Entry{ID: "three", Role: "user", Content: "three"})
	}()
	select {
	case <-summarizer.started:
	case <-time.After(2 * time.Second):
		t.Fatal("summarizer did not start")
	}
	if err := mem.Delete(ctx, "one"); err != nil {
		t.Fatalf("Delete(one) error = %v", err)
	}
	close(summarizer.release)
	if err := <-saveDone; err != nil {
		t.Fatalf("Save(three) error = %v", err)
	}
	if _, err := mem.Get(ctx, "one"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted entry resurrected: %v", err)
	}
}

func TestSummaryMemorySameIDUpdateDuringSummarizeIsPreserved(t *testing.T) {
	t.Parallel()

	summarizer := &blockingSummarizer{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	mem := NewSummaryMemory(summarizer, WithMaxEntries(2), WithKeepRecent(1))
	ctx := context.Background()
	for _, id := range []string{"one", "two"} {
		if err := mem.Save(ctx, Entry{ID: id, Role: "user", Content: id}); err != nil {
			t.Fatalf("Save(%s) error = %v", id, err)
		}
	}

	saveDone := make(chan error, 1)
	go func() {
		saveDone <- mem.Save(ctx, Entry{ID: "three", Role: "user", Content: "three"})
	}()
	select {
	case <-summarizer.started:
	case <-time.After(2 * time.Second):
		t.Fatal("summarizer did not start")
	}
	// 摘要快照包含 two；摘要期间用同 ID 写入新版本，提交阶段不能把它删掉。
	if err := mem.Save(ctx, Entry{ID: "two", Role: "user", Content: "updated"}); err != nil {
		t.Fatalf("Save(two updated) error = %v", err)
	}
	close(summarizer.release)
	if err := <-saveDone; err != nil {
		t.Fatalf("Save(three) error = %v", err)
	}

	updated, err := mem.Get(ctx, "two")
	if err != nil {
		t.Fatalf("updated entry was deleted by summarize: %v", err)
	}
	if updated.Content != "updated" {
		t.Fatalf("entry content = %q, want updated", updated.Content)
	}
	if _, err := mem.Get(ctx, "one"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unchanged summarized entry should be removed, got %v", err)
	}
	if mem.GetSummary() == "" {
		t.Fatal("summary was not committed")
	}
}

type reentrantSummarizer struct {
	memory *SummaryMemory
	err    error
}

func (s *reentrantSummarizer) Summarize(ctx context.Context, _ string) (string, error) {
	s.err = s.memory.ForceSummarize(ctx)
	return "summary", nil
}

func TestSummaryMemorySummarizerReentrancyReturnsInProgress(t *testing.T) {
	t.Parallel()

	summarizer := &reentrantSummarizer{}
	mem := NewSummaryMemory(summarizer, WithMaxEntries(1), WithKeepRecent(1))
	summarizer.memory = mem
	ctx := context.Background()

	if err := mem.Save(ctx, Entry{ID: "one", Role: "user", Content: "one"}); err != nil {
		t.Fatalf("Save(one) error = %v", err)
	}
	if err := mem.Save(ctx, Entry{ID: "two", Role: "user", Content: "two"}); err != nil {
		t.Fatalf("Save(two) error = %v", err)
	}
	if !errors.Is(summarizer.err, ErrSummaryInProgress) {
		t.Fatalf("reentrant ForceSummarize error = %v, want ErrSummaryInProgress", summarizer.err)
	}
}
