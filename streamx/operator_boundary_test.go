package streamx

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

func TestStreamAdaptersKeepMergedTailAndError(t *testing.T) {
	adapters := map[string]func(*StreamReader[string]) *StreamReader[string]{"backpressure": func(r *StreamReader[string]) *StreamReader[string] { return Backpressure(r, nil) }, "debounce": func(r *StreamReader[string]) *StreamReader[string] { return Debounce(r, time.Millisecond) }}
	for name, adapt := range adapters {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			r := adapt(Merge(FromSlice([]string{}), FromValue("tail")))
			defer r.Close()
			values, err := r.Collect(ctx)
			if err != nil || !reflect.DeepEqual(values, []string{"tail"}) {
				t.Fatalf("values=%v error=%v", values, err)
			}
			failure := errors.New("upstream failed")
			input, w := Pipe[string](1)
			if err := w.Send("first"); err != nil {
				t.Fatal(err)
			}
			w.CloseWithError(failure)
			out := adapt(input)
			defer out.Close()
			values, err = out.Collect(ctx)
			if !errors.Is(err, failure) || !reflect.DeepEqual(values, []string{"first"}) {
				t.Fatalf("partial=%v error=%v", values, err)
			}
		})
	}
}

func TestDebounceFlushesLastBurstOnSourceCompletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	out := Debounce(FromSlice([]string{"first", "second", "last"}), time.Hour)
	defer out.Close()
	values, err := out.Collect(ctx)
	if err != nil || !reflect.DeepEqual(values, []string{"last"}) {
		t.Fatalf("values=%v error=%v", values, err)
	}
}

func TestPipeClosePanicReturnsFailureAndReleasesWaiters(t *testing.T) {
	failure := errors.New("close failed")
	calls := 0
	r, w := PipeWithClose[string](0, func() error { calls++; panic(failure) })
	e := r.Close()
	var p *ClosePanicError
	if !errors.As(e, &p) || !errors.Is(e, failure) || len(p.Stack) == 0 || !p.UnsafeToReplay() {
		t.Fatalf("close error=%v", e)
	}
	if r.Close() != e || calls != 1 {
		t.Fatal("close callback repeated or error lost")
	}
	if w.Send("late") != ErrStreamClosed {
		t.Fatal("writer not released")
	}
}

func TestCollectKeepsValuesAndCloseFailure(t *testing.T) {
	failure := errors.New("cleanup failed")
	r, w := PipeWithClose[string](1, func() error { return failure })
	w.Send("value")
	w.Close()
	values, e := r.Collect(t.Context())
	var closeError *collectCloseError
	if !errors.Is(e, failure) || !errors.As(e, &closeError) || !reflect.DeepEqual(values, []string{"value"}) {
		t.Fatalf("values=%v error=%v", values, e)
	}
}

func TestPipeFirstTerminalErrorIsStable(t *testing.T) {
	r, w := Pipe[string](1)
	defer r.Close()
	first := errors.New("first failure")
	w.Send("value")
	w.CloseWithError(first)
	w.CloseWithError(&ClosePanicError{Value: "late"})
	w.CloseWithError(nil)
	value, e := r.Recv()
	if value != "value" || e != nil {
		t.Fatalf("buffered value=%q error=%v", value, e)
	}
	_, e = r.Recv()
	if !errors.Is(e, first) {
		t.Fatalf("first terminal overwritten: %v", e)
	}
}
func TestPipeCloseReleasesBlockedSenders(t *testing.T) {
	r, w := Pipe[int](1)
	defer r.Close()
	if e := w.Send(1); e != nil {
		t.Fatal(e)
	}
	ready := make(chan struct{})
	done := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func() { ready <- struct{}{}; done <- w.Send(2) }()
	}
	for i := 0; i < 8; i++ {
		<-ready
	}
	if e := w.Close(); e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 8; i++ {
		select {
		case e := <-done:
			if e != ErrStreamClosed {
				t.Fatalf("send=%v", e)
			}
		case <-time.After(time.Second):
			t.Fatal("blocked sender not released")
		}
	}
	value, e := r.Recv()
	if value != 1 || e != nil {
		t.Fatalf("buffered value lost: %d %v", value, e)
	}
}

func TestZipCloseReleasesBothInputs(t *testing.T) {
	left, right := 0, 0
	a, wa := PipeWithClose[string](0, func() error { left++; return nil })
	b, wb := PipeWithClose[int](0, func() error { right++; return nil })
	defer wa.Close()
	defer wb.Close()
	defer a.Close()
	defer b.Close()
	out := Zip(a, b, func(v string, n int) string { return v })
	if e := out.Close(); e != nil {
		t.Fatal(e)
	}
	if left != 1 || right != 1 {
		t.Fatalf("left=%d right=%d", left, right)
	}
}
func TestMergedCollectPreservesAllCloseFailures(t *testing.T) {
	one, two := errors.New("first cleanup"), errors.New("second cleanup")
	a, wa := PipeWithClose[string](1, func() error { return one })
	b, wb := PipeWithClose[string](1, func() error { return two })
	wa.Send("one")
	wb.Send("two")
	wa.Close()
	wb.Close()
	values, e := Merge(a, b).Collect(t.Context())
	if len(values) != 2 || !errors.Is(e, one) || !errors.Is(e, two) {
		t.Fatalf("values=%v error=%v", values, e)
	}
}

func TestZipKeepsPendingLeftAcrossRightSourceEOF(t *testing.T) {
	for _, mode := range []string{"left", "right", "both"} {
		t.Run(mode, func(t *testing.T) {
			left, right := FromValue("left"), FromValue("right")
			if mode != "right" {
				left = Merge(FromSlice([]string{}), left)
			}
			if mode != "left" {
				right = Merge(FromSlice([]string{}), right)
			}
			out := Zip(left, right, func(a, b string) string { return a + ":" + b })
			values, e := out.Collect(t.Context())
			if e != nil || !reflect.DeepEqual(values, []string{"left:right"}) {
				t.Fatalf("values=%v error=%v", values, e)
			}
		})
	}
}
func TestZipCollectPreservesBothCloseFailures(t *testing.T) {
	one, two := errors.New("first zip cleanup"), errors.New("second zip cleanup")
	a, wa := PipeWithClose[string](1, func() error { return one })
	b, wb := PipeWithClose[string](1, func() error { return two })
	wa.Send("left")
	wb.Send("right")
	wa.Close()
	wb.Close()
	values, e := Zip(a, b, func(a, b string) string { return a + b }).Collect(t.Context())
	if len(values) != 1 || !errors.Is(e, one) || !errors.Is(e, two) {
		t.Fatalf("values=%v error=%v", values, e)
	}
}

func TestBackgroundReadersContainSourcePanics(t *testing.T) {
	adapters := map[string]func(*StreamReader[string]) *StreamReader[string]{"buffer": func(r *StreamReader[string]) *StreamReader[string] { return Buffer(r, 1) }, "timeout": func(r *StreamReader[string]) *StreamReader[string] { return Timeout(r, time.Second) }, "backpressure": func(r *StreamReader[string]) *StreamReader[string] { return Backpressure(r, nil) }, "debounce": func(r *StreamReader[string]) *StreamReader[string] { return Debounce(r, time.Millisecond) }, "copy": func(r *StreamReader[string]) *StreamReader[string] {
		branches := r.Copy(2)
		t.Cleanup(func() { branches[1].Close() })
		return branches[0]
	}}
	for name, adapt := range adapters {
		t.Run(name, func(t *testing.T) {
			failure := errors.New("source mapper failed")
			source := Map(FromValue("value"), func(string) string { panic(failure) })
			ctx, c := context.WithTimeout(t.Context(), time.Second)
			defer c()
			_, e := adapt(source).Collect(ctx)
			var p *SourcePanicError
			if !errors.Is(e, failure) || !errors.As(e, &p) || len(p.Stack) == 0 || !p.UnsafeToReplay() {
				t.Fatalf("source panic=%v", e)
			}
		})
	}
}
func TestBackpressureObserverPanicIsAStreamFailure(t *testing.T) {
	out := Backpressure(FromValue("value"), &BackpressureConfig{BufferSize: 1, HighWaterMark: 0, OnHighWater: func() { panic("observer failed") }})
	_, e := out.Collect(t.Context())
	var p *SourcePanicError
	if !errors.As(e, &p) {
		t.Fatalf("observer error=%v", e)
	}
}
