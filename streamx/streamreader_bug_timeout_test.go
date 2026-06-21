package streamx

import (
	"runtime"
	"testing"
	"time"
)

// TestTimeoutReader_LosslessAfterTimeout verifies the lossless semantic: items
// that become ready only after a recv() timed out are NOT dropped — they are
// delivered, in order, on subsequent recv() calls. A timeout means "not ready
// yet", never "data destroyed". (Replaces an earlier test that asserted the
// drop semantic; see F2 — Timeout is now lossless by design.)
func TestTimeoutReader_LosslessAfterTimeout(t *testing.T) {
	src, w := Pipe[string](0)

	// Producer: both elements arrive only after the 15ms timeout window.
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

	// Drain the rest, tolerating intermediate timeouts. Lossless: both "a" and
	// "b" must arrive, in order; nothing dropped.
	var got []string
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
	}

	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("lossless violated: want [a b] after timeout, got %v", got)
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
