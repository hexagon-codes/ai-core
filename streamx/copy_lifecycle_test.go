package streamx

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestCopyClosesOnlyAfterLastBranch(t *testing.T) {
	var closed atomic.Int32
	source, writer := PipeWithClose[string](1, func() error { closed.Add(1); return nil })
	branches := source.Copy(2)
	if _, err := source.Recv(); !errors.Is(err, ErrStreamClosed) {
		t.Fatalf("original reader still consumes source: %v", err)
	}
	if err := branches[0].Close(); err != nil {
		t.Fatal(err)
	}
	if closed.Load() != 0 {
		t.Fatal("one branch closed the shared source")
	}
	if err := writer.Send("remaining"); err != nil {
		t.Fatal(err)
	}
	if value, err := branches[1].Recv(); err != nil || value != "remaining" {
		t.Fatalf("value=%q error=%v", value, err)
	}
	_ = branches[1].Close()
	_ = branches[1].Close()
	_ = source.Close()
	if closed.Load() != 1 {
		t.Fatalf("source closed %d times", closed.Load())
	}
	if err := writer.Send("late"); !errors.Is(err, ErrStreamClosed) {
		t.Fatalf("producer not released: %v", err)
	}
}

func TestCopyClosingBlockedBranchDoesNotCloseOtherBranch(t *testing.T) {
	source, writer := Pipe[string](0)
	branches := source.Copy(2)
	done := make(chan error, 1)
	go func() { _, err := branches[0].Recv(); done <- err }()
	_ = branches[0].Close()
	select {
	case err := <-done:
		if !errors.Is(err, ErrStreamClosed) {
			t.Fatalf("error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("closed branch remained blocked")
	}
	go func() { _ = writer.Send("live"); _ = writer.Close() }()
	if value, err := branches[1].Recv(); err != nil || value != "live" {
		t.Fatalf("remaining value=%q error=%v", value, err)
	}
	_ = branches[1].Close()
}

func TestCopyPropagatesSourceErrorToEveryBranch(t *testing.T) {
	failure := errors.New("source failed")
	source, writer := Pipe[string](0)
	branches := source.Copy(2)
	_ = writer.CloseWithError(failure)
	for _, reader := range branches {
		if _, err := reader.Recv(); !errors.Is(err, failure) {
			t.Fatalf("error=%v, want source failure", err)
		}
	}
}

func TestCollectCancellationUnblocksRecv(t *testing.T) {
	source, _ := Pipe[string](0)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { _, err := source.Collect(ctx); done <- err }()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("collect did not release blocked read")
	}
}
