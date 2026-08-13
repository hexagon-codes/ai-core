package media

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestTaskState_Terminal(t *testing.T) {
	cases := map[TaskState]bool{
		TaskQueued:         false,
		TaskRunning:        false,
		TaskSucceeded:      true,
		TaskFailed:         true,
		TaskState("other"): false,
	}
	for s, want := range cases {
		if got := s.Terminal(); got != want {
			t.Errorf("TaskState(%q).Terminal() = %v, want %v", s, got, want)
		}
	}
}

func TestWaitFor_ImmediateDone(t *testing.T) {
	calls := 0
	err := WaitFor(context.Background(), time.Millisecond, func(context.Context) (bool, error) {
		calls++
		return true, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 1 {
		t.Errorf("expected exactly 1 poll, got %d", calls)
	}
}

func TestWaitFor_DoneAfterSeveralPolls(t *testing.T) {
	calls := 0
	err := WaitFor(context.Background(), time.Millisecond, func(context.Context) (bool, error) {
		calls++
		return calls >= 3, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 3 {
		t.Errorf("expected 3 polls, got %d", calls)
	}
}

func TestWaitFor_PropagatesPollError(t *testing.T) {
	sentinel := errors.New("boom")
	err := WaitFor(context.Background(), time.Millisecond, func(context.Context) (bool, error) {
		return false, sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Errorf("expected sentinel error, got %v", err)
	}
}

func TestWaitFor_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	go func() {
		// 取消前先让首轮轮询返回 not-done。
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	err := WaitFor(ctx, 5*time.Millisecond, func(context.Context) (bool, error) {
		calls++
		return false, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if calls == 0 {
		t.Error("expected at least one poll before cancel")
	}
}

func TestWaitFor_DefaultInterval(t *testing.T) {
	// interval<=0 应回退到 DefaultPollInterval 而非死循环；首轮即 done 可立即返回。
	done := make(chan struct{})
	go func() {
		_ = WaitFor(context.Background(), 0, func(context.Context) (bool, error) {
			return true, nil
		})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("WaitFor with default interval did not return promptly on immediate done")
	}
}
