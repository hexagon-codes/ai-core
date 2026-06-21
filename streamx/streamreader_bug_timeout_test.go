package streamx

import (
	"runtime"
	"testing"
	"time"
)

// TestTimeoutReader_Bug_RedeliverAfterTimeout verifies that an item whose
// arrival raced (and lost to) the timeout is discarded rather than being
// re-delivered on the next recv() (off-by-one desync).
func TestTimeoutReader_Bug_RedeliverAfterTimeout(t *testing.T) {
	src, w := Pipe[string](0)

	// Producer: "a" arrives after the timeout window (it will be abandoned),
	// then "b" arrives shortly after.
	go func() {
		time.Sleep(40 * time.Millisecond)
		_ = w.Send("a")
		_ = w.Send("b")
		_ = w.Close()
	}()

	tr := Timeout(src, 15*time.Millisecond)

	// First recv must time out (nothing ready within 15ms).
	if _, err := tr.Recv(); err != ErrStreamTimeout {
		t.Fatalf("recv#1: want ErrStreamTimeout, got %v", err)
	}

	// "a" raced the timeout and is abandoned. Drain the rest, tolerating
	// further intermediate timeouts. The abandoned "a" must not be
	// re-delivered; "b" should eventually arrive.
	var got []string
	sawB := false
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		v, err := tr.Recv()
		if err == ErrStreamTimeout {
			continue
		}
		if err != nil {
			break
		}
		got = append(got, v)
		if v == "b" {
			sawB = true
		}
	}

	for _, v := range got {
		if v == "a" {
			t.Fatalf("timed-out item \"a\" was re-delivered (desync): got %v", got)
		}
	}
	if !sawB {
		t.Fatalf("expected to receive \"b\" after timeout, got %v", got)
	}
}

// TestTimeoutReader_Bug_GoroutineLeak verifies that abandoning a timed-out
// stream WITHOUT calling Close() does not leak the background goroutine.
func TestTimeoutReader_Bug_GoroutineLeak(t *testing.T) {
	runtime.GC()
	before := runtime.NumGoroutine()

	const n = 50
	for i := 0; i < n; i++ {
		src, _ := Pipe[int](0) // never written to -> recv blocks forever
		tr := Timeout(src, 5*time.Millisecond)
		if _, err := tr.Recv(); err != ErrStreamTimeout {
			t.Fatalf("iter %d: want ErrStreamTimeout, got %v", i, err)
		}
		// Intentionally NOT calling tr.Close(): abandoning the reader must not
		// leak the background goroutine.
	}

	// Give abandoned goroutines a chance to exit.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if runtime.NumGoroutine()-before < n/2 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	leaked := runtime.NumGoroutine() - before
	if leaked >= n/2 {
		t.Fatalf("goroutine leak: started with %d, now leaked ~%d after abandoning %d timed-out streams",
			before, leaked, n)
	}
}
