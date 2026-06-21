package streamx

// 回归锁定（F2）：streamx.Timeout 曾在一次超时后丢弃"下一条到达"的元素，导致
// 生产者发出的元素永不可达消费者（静默数据丢失）。现已改为无损语义：超时只返回
// ErrStreamTimeout 信号，在途元素留在缓冲并在后续 recv 按序投递。
//
// 修复前：生产者发 [only-element] → 消费者收到 []；修复后：收到 [only-element]。

import (
	"testing"
	"time"
)

// 单元素场景：唯一一条元素在超时窗口之后才产出（从未与超时赛跑），不得被丢弃。
func TestF2_TimeoutSilentlyDropsNeverRacedElement(t *testing.T) {
	src, w := Pipe[string](0)

	go func() {
		time.Sleep(40 * time.Millisecond)
		_ = w.Send("only-element")
		_ = w.Close()
	}()

	tr := Timeout(src, 15*time.Millisecond)

	var got []string
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		v, err := tr.Recv()
		if err == ErrStreamTimeout {
			continue
		}
		if err != nil {
			break // EOF
		}
		got = append(got, v)
	}

	if len(got) != 1 || got[0] != "only-element" {
		t.Fatalf("数据丢失回归：生产者发出 [only-element]，消费者收到 %v", got)
	}
}

// 不变量（property）：无论超时与到达如何交错，生产者发出的全部元素都应被消费者
// 按序、无重复、无丢失地收到。timeout 远小于元素间隔 → 大量超时穿插，但 0 丢失。
func TestF2_TimeoutLosslessInvariant(t *testing.T) {
	src, w := Pipe[int](0)
	const n = 30

	go func() {
		for i := 0; i < n; i++ {
			time.Sleep(2 * time.Millisecond)
			_ = w.Send(i)
		}
		_ = w.Close()
	}()

	tr := Timeout(src, 1*time.Millisecond) // 1ms << 2ms 间隔 → 频繁超时

	var got []int
	deadline := time.Now().Add(5 * time.Second)
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

	if len(got) != n {
		t.Fatalf("lossless 不变量被破坏：发出 %d 条，收到 %d 条: %v", n, len(got), got)
	}
	for i := 0; i < n; i++ {
		if got[i] != i {
			t.Fatalf("顺序/内容被破坏 at %d: %v", i, got)
		}
	}
}
