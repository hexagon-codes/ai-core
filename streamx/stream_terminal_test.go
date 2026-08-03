package streamx

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestStreamOnTerminalRunsExactlyOnceAfterEOF(t *testing.T) {
	stream := NewStream(strings.NewReader("data: [DONE]\n"), OpenAIFormat)
	var calls atomic.Int32
	stream.OnTerminal(func() { calls.Add(1) })

	for range stream.Chunks() {
	}
	select {
	case <-stream.Done():
	case <-time.After(time.Second):
		t.Fatal("stream did not reach terminal state")
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("terminal callbacks=%d, want exactly 1", got)
	}
}

func TestStreamCloseBeforeStartIsTerminal(t *testing.T) {
	stream := NewStream(strings.NewReader("data: ignored\n"), OpenAIFormat)
	var calls atomic.Int32
	stream.OnTerminal(func() { calls.Add(1) })

	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-stream.Done():
	default:
		t.Fatal("Close before Start must close Done")
	}
	select {
	case _, ok := <-stream.Chunks():
		if ok {
			t.Fatal("Close before Start must close Chunks")
		}
	default:
		t.Fatal("Close before Start must close Chunks without starting a reader")
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("terminal callbacks=%d, want exactly 1", got)
	}

	var lateCalls atomic.Int32
	stream.OnTerminal(func() { lateCalls.Add(1) })
	if got := lateCalls.Load(); got != 1 {
		t.Fatalf("late terminal callbacks=%d, want immediate invocation", got)
	}
}

func TestStreamOnFirstChunkComposesAndRunsOnce(t *testing.T) {
	stream := NewStream(strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n"+
			"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n"+
			"data: [DONE]\n",
	), OpenAIFormat)
	var first, second atomic.Int32
	stream.OnFirstChunk(func() { first.Add(1) })
	stream.OnFirstChunk(func() { second.Add(1) })
	for range stream.Chunks() {
	}
	<-stream.Done()
	if first.Load() != 1 || second.Load() != 1 {
		t.Fatalf("first-chunk callbacks=(%d,%d), want (1,1)", first.Load(), second.Load())
	}

	var late atomic.Int32
	stream.OnFirstChunk(func() { late.Add(1) })
	if got := late.Load(); got != 1 {
		t.Fatalf("late first-chunk callback=%d, want immediate invocation", got)
	}
}

func TestStreamTerminalObserverPanicCannotBlockLaterObserversOrDone(t *testing.T) {
	stream := NewStream(strings.NewReader("data: [DONE]\n"), OpenAIFormat)
	var later atomic.Int32
	stream.OnTerminal(func() { panic("observer canary") })
	stream.OnTerminal(func() { later.Add(1) })
	for range stream.Chunks() {
	}
	select {
	case <-stream.Done():
	case <-time.After(time.Second):
		t.Fatal("panicking terminal observer prevented Done publication")
	}
	if got := later.Load(); got != 1 {
		t.Fatalf("later terminal observer calls=%d, want 1", got)
	}
}

func TestStreamFirstChunkObserverPanicCannotBlockLaterObservers(t *testing.T) {
	stream := NewStream(strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n"+
			"data: [DONE]\n",
	), OpenAIFormat)
	var later atomic.Int32
	stream.OnFirstChunk(func() { panic("first observer canary") })
	stream.OnFirstChunk(func() { later.Add(1) })
	for range stream.Chunks() {
	}
	<-stream.Done()
	if got := later.Load(); got != 1 {
		t.Fatalf("later first-chunk observer calls=%d, want 1", got)
	}
}

func TestStreamErrReportsParseFailureWithoutConsumingErrorsChannel(t *testing.T) {
	stream := NewStream(strings.NewReader("data: definitely-not-json\n"), OpenAIFormat)
	for range stream.Chunks() {
	}
	<-stream.Done()
	if stream.Err() == nil {
		t.Fatal("Err()=nil after parse failure")
	}
	select {
	case err := <-stream.Errors():
		if err == nil {
			t.Fatal("Errors() yielded nil after parse failure")
		}
	default:
		t.Fatal("Err() consumed the public Errors channel")
	}
}

func TestStreamErrReportsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stream := NewStreamWithContext(ctx, strings.NewReader("data: [DONE]\n"), OpenAIFormat)
	stream.Start()
	<-stream.Done()
	if !errors.Is(stream.Err(), context.Canceled) {
		t.Fatalf("Err()=%v, want context.Canceled", stream.Err())
	}
}

func TestStreamLateLifecycleObserverPanicIsIsolated(t *testing.T) {
	stream := NewStream(strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n"+
			"data: [DONE]\n",
	), OpenAIFormat)
	for range stream.Chunks() {
	}
	<-stream.Done()

	assertDoesNotPanic := func(name string, register func()) {
		t.Helper()
		defer func() {
			if recovered := recover(); recovered != nil {
				t.Fatalf("%s observer panic escaped registration: %v", name, recovered)
			}
		}()
		register()
	}
	assertDoesNotPanic("terminal", func() {
		stream.OnTerminal(func() { panic("late terminal canary") })
	})
	assertDoesNotPanic("first chunk", func() {
		stream.OnFirstChunk(func() { panic("late first chunk canary") })
	})
}

func TestStreamCloseWaitsForTerminalObservers(t *testing.T) {
	stream := NewStream(strings.NewReader("data: [DONE]\n"), OpenAIFormat)
	observerStarted := make(chan struct{})
	releaseObserver := make(chan struct{})
	stream.OnTerminal(func() {
		close(observerStarted)
		<-releaseObserver
	})
	stream.Start()

	select {
	case <-observerStarted:
	case <-time.After(time.Second):
		t.Fatal("terminal observer did not start")
	}
	closeReturned := make(chan struct{})
	go func() {
		_ = stream.Close()
		close(closeReturned)
	}()
	select {
	case <-closeReturned:
		t.Fatal("Close returned before terminal observer completed")
	case <-time.After(20 * time.Millisecond):
	}
	close(releaseObserver)
	select {
	case <-closeReturned:
	case <-time.After(time.Second):
		t.Fatal("Close did not return after terminal observer completed")
	}
}

func TestStreamCollectWaitsForTerminalObservers(t *testing.T) {
	stream := NewStream(strings.NewReader("data: [DONE]\n"), OpenAIFormat)
	observerStarted := make(chan struct{})
	releaseObserver := make(chan struct{})
	stream.OnTerminal(func() {
		close(observerStarted)
		<-releaseObserver
	})

	collectReturned := make(chan struct{})
	go func() {
		_, _ = stream.Collect()
		close(collectReturned)
	}()
	select {
	case <-observerStarted:
	case <-time.After(time.Second):
		t.Fatal("terminal observer did not start")
	}
	select {
	case <-collectReturned:
		t.Fatal("Collect returned before terminal observer completed")
	case <-time.After(20 * time.Millisecond):
	}
	close(releaseObserver)
	select {
	case <-collectReturned:
	case <-time.After(time.Second):
		t.Fatal("Collect did not return after terminal observer completed")
	}
}

func TestStreamCollectDoesNotMisclassifyNormalTerminalCleanupCancellation(t *testing.T) {
	for iteration := 0; iteration < 100; iteration++ {
		parent, cancel := context.WithCancel(context.Background())
		stream := NewStreamWithContext(
			parent,
			strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n"+
				"data: [DONE]\n"),
			OpenAIFormat,
		)
		stream.OnTerminal(cancel)
		result, err := stream.Collect()
		if err != nil {
			t.Fatalf("iteration %d: normal EOF cleanup returned error: %v", iteration, err)
		}
		if result == nil || result.Content != "ok" {
			t.Fatalf("iteration %d: result=%#v, want content ok", iteration, result)
		}
	}
}
