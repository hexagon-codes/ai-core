package streamx

import (
	"context"
	"errors"
	"testing"
	"time"
)

// Copy 的每个分支都必须保留最后释放源时的关闭错误，不能把清理失败报告成正常 EOF。
func TestCopyBranchesPreserveSourceCloseFailure(t *testing.T) {
	failure := errors.New("source cleanup failed")
	source, writer := PipeWithClose[int](1, func() error { return failure })
	if err := writer.Send(1); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	branches := source.Copy(2)
	if len(branches) != 2 {
		t.Fatalf("branches=%d", len(branches))
	}
	for index, branch := range branches {
		values, err := branch.Collect(context.Background())
		if len(values) != 1 || values[0] != 1 {
			t.Fatalf("branch %d values=%v", index, values)
		}
		if !errors.Is(err, failure) {
			t.Fatalf("branch %d lost close failure: %v", index, err)
		}
	}
}

// 取消必须唤醒静默源上的 ForEach/Reduce，并返回取消原因而不是空成功。
func TestForEachAndReduceCancelBlockedSource(t *testing.T) {
	for _, mode := range []string{"foreach", "reduce"} {
		t.Run(mode, func(t *testing.T) {
			source, writer := Pipe[int](0)
			defer writer.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			done := make(chan error, 1)
			go func() {
				if mode == "foreach" {
					done <- source.ForEach(ctx, func(int) error { return nil })
					return
				}
				_, err := Reduce(ctx, source, 0, func(a, b int) int { return a + b })
				done <- err
			}()
			select {
			case err := <-done:
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("cancel returned %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("cancel did not wake blocked source")
			}
		})
	}
}
