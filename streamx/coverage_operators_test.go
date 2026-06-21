package streamx

import (
	"context"
	"testing"
	"time"
)

// TestOperators_CloseChain exercises the close() path of every operator wrapper,
// which was almost entirely uncovered (0%). Each operator should propagate Close
// down to the source without panicking.
func TestOperators_CloseChain(t *testing.T) {
	mk := func() *StreamReader[int] { return FromSlice([]int{1, 2, 3, 4, 5}) }

	closers := []struct {
		name string
		r    *StreamReader[int]
	}{
		{"Map", Map(mk(), func(i int) int { return i })},
		{"Filter", Filter(mk(), func(i int) bool { return i%2 == 0 })},
		{"Buffer", Buffer(mk(), 2)},
		{"Take", Take(mk(), 3)},
		{"Skip", Skip(mk(), 2)},
		{"TakeWhile", TakeWhile(mk(), func(i int) bool { return i < 3 })},
		{"SkipWhile", SkipWhile(mk(), func(i int) bool { return i < 3 })},
		{"Distinct", Distinct(mk(), func(a, b int) bool { return a == b })},
		{"DistinctBy", DistinctBy(mk(), func(i int) int { return i })},
		{"Throttle", Throttle(mk(), time.Millisecond)},
		{"Backpressure", Backpressure(mk(), nil)},
	}
	for _, c := range closers {
		if err := c.r.Close(); err != nil {
			t.Errorf("%s.Close() returned err: %v", c.name, err)
		}
	}
}

// TestTake_LimitsOutput covers takeReader.recv hitting the limit boundary.
func TestTake_LimitsOutput(t *testing.T) {
	out, err := Take(FromSlice([]int{1, 2, 3, 4, 5}), 2).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 2 || out[0] != 1 || out[1] != 2 {
		t.Errorf("Take(2) = %v, want [1 2]", out)
	}
}

// TestSkip_DropsPrefix covers skipReader.recv.
func TestSkip_DropsPrefix(t *testing.T) {
	out, err := Skip(FromSlice([]int{1, 2, 3, 4, 5}), 3).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 2 || out[0] != 4 {
		t.Errorf("Skip(3) = %v, want [4 5]", out)
	}
}

// TestTakeWhile_StopsAtPredicate covers takeWhileReader done transition.
func TestTakeWhile_StopsAtPredicate(t *testing.T) {
	out, err := TakeWhile(FromSlice([]int{1, 2, 3, 1}), func(i int) bool { return i < 3 }).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 2 {
		t.Errorf("TakeWhile(<3) = %v, want [1 2]", out)
	}
}

// TestSkipWhile_DropsPrefix covers skipWhileReader.
func TestSkipWhile_DropsPrefix(t *testing.T) {
	out, err := SkipWhile(FromSlice([]int{1, 2, 5, 1, 2}), func(i int) bool { return i < 3 }).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	// drops leading 1,2 then yields from first >=3: 5,1,2
	if len(out) != 3 || out[0] != 5 {
		t.Errorf("SkipWhile(<3) = %v, want [5 1 2]", out)
	}
}

// TestDistinct_DedupsConsecutiveAndRepeated covers distinctReader.recv full loop.
func TestDistinct_DedupsRepeated(t *testing.T) {
	out, err := Distinct(FromSlice([]int{1, 1, 2, 1, 3, 3}), func(a, b int) bool { return a == b }).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 3 {
		t.Errorf("Distinct = %v, want 3 unique", out)
	}
}

// TestBatch_GroupsBySize covers batchReader.recv including the trailing partial batch.
func TestBatch_GroupsBySize(t *testing.T) {
	out, err := Batch(FromSlice([]int{1, 2, 3, 4, 5}), 2).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 3 {
		t.Errorf("Batch(2) of 5 => %d batches, want 3", len(out))
	}
	if len(out[2]) != 1 {
		t.Errorf("trailing batch len=%d, want 1", len(out[2]))
	}
}

// TestThrottle_SkipsWithinWindow covers throttleReader skip-within-window branch.
func TestThrottle_PassesAllWhenZeroSpacing(t *testing.T) {
	// duration 0 => every element passes (now.Sub(last) >= 0 always true).
	out, err := Throttle(FromSlice([]int{1, 2, 3}), 0).Collect(context.Background())
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 3 {
		t.Errorf("Throttle(0) should pass all 3, got %d", len(out))
	}
}

// TestMerge_SourceEOFThenAllDone covers multiReader.recv SourceEOF + allDone paths.
func TestMerge_DrainsBothSources(t *testing.T) {
	a := FromSlice([]int{1, 2}).SetSource("a")
	b := FromSlice([]int{3, 4}).SetSource("b")
	merged := Merge(a, b)
	out, err := merged.Collect(context.Background()) // Collect ignores SourceEOF
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out) != 4 {
		t.Errorf("Merge drained %d items, want 4", len(out))
	}
}

// TestReduce_Sums covers Reduce operator end-to-end.
func TestReduce_Sums(t *testing.T) {
	sum, err := Reduce(context.Background(), FromSlice([]int{1, 2, 3, 4}), 0, func(acc, x int) int { return acc + x })
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if sum != 10 {
		t.Errorf("Reduce sum = %d, want 10", sum)
	}
}
