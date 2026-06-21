package streamx

// 回归锁定（F1）：Distinct 在"可比较元素 → 不可比较元素触发回退"后，曾因迁移块
// 未真正把 seenSet 迁移到 seen（只重置为空表）而丢失已见状态，导致之前出现过的
// 可比较元素被当作新元素再次发射，违反去重契约。
//
// 修复前输出：[1 2 [9] 1]（元素 1 发射两次）；修复后：[1 2 [9]]。
// 根因与修复见 streamreader.go distinctReader.recv 的回退迁移块。

import (
	"context"
	"reflect"
	"testing"
)

func TestDistinct_FallbackPreservesSeenState(t *testing.T) {
	// any 流：1, 2 先入 seenSet；[]int{9} 不可比较→触发回退；末尾 1 是 1 的重复。
	in := []any{1, 2, []int{9}, 1}
	src := FromSlice(in)
	d := Distinct(src, func(a, b any) bool { return reflect.DeepEqual(a, b) })

	got, err := d.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	ones := 0
	for _, v := range got {
		if iv, ok := v.(int); ok && iv == 1 {
			ones++
		}
	}
	if ones != 1 {
		t.Fatalf("regression: element 1 emitted %d times after fallback (dedup state lost); "+
			"want exactly 1. got=%v", ones, got)
	}
	// 完整断言：三个去重后的元素，首次出现顺序保持。
	if len(got) != 3 {
		t.Fatalf("want 3 distinct elements, got %d: %v", len(got), got)
	}
}
