package streamx

import (
	"context"
	"errors"
	"testing"
)

func TestBatchPropagatesSourceErrorAfterPartialBatch(t *testing.T) {
	t.Parallel()

	reader, writer := Pipe[int](2)
	if err := writer.Send(1); err != nil {
		t.Fatalf("Send() error = %v", err)
	}
	_ = writer.CloseWithError(errors.New("source failed"))

	items, err := Batch(reader, 2).Collect(context.Background())
	if err == nil || err.Error() != "source failed" {
		t.Fatalf("Collect() error = %v, want source failed", err)
	}
	if len(items) != 1 || len(items[0]) != 1 || items[0][0] != 1 {
		t.Fatalf("Collect() items = %v, want [[1]] before error", items)
	}
}

func TestWindowPropagatesSourceErrorAfterWindow(t *testing.T) {
	t.Parallel()

	reader, writer := Pipe[int](2)
	_ = writer.Send(1)
	_ = writer.Send(2)
	_ = writer.CloseWithError(errors.New("source failed"))

	items, err := Window(reader, 2).Collect(context.Background())
	if err == nil || err.Error() != "source failed" {
		t.Fatalf("Collect() error = %v, want source failed", err)
	}
	if len(items) != 1 || len(items[0]) != 2 {
		t.Fatalf("Collect() items = %v, want one [1 2] window", items)
	}
}
