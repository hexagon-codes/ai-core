package streamx

import (
	"context"
	"testing"
)

// TestDistinct_Bug_Correctness ensures the optimized Distinct still de-dups
// correctly for comparable types and preserves first-seen order.
func TestDistinct_Bug_Correctness(t *testing.T) {
	in := []int{1, 1, 2, 3, 2, 4, 4, 4, 5, 1}
	src := FromSlice(in)
	d := Distinct(src, func(a, b int) bool { return a == b })

	got, err := d.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	want := []int{1, 2, 3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

// TestDistinct_Bug_NonComparable ensures non-comparable element types (which
// cannot be map keys) still work via the equals fallback path.
func TestDistinct_Bug_NonComparable(t *testing.T) {
	in := [][]int{{1, 2}, {1, 2}, {3}, {3}, {4, 5}}
	src := FromSlice(in)
	eq := func(a, b []int) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	d := Distinct(src, eq)
	got, err := d.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 distinct slices, got %d: %v", len(got), got)
	}
}

// BenchmarkDistinct_AllUnique exercises the worst case for the linear scan:
// every element unique, so each new element is compared against all prior
// elements (O(n^2)). With a set-backed implementation this is O(n).
func BenchmarkDistinct_AllUnique(b *testing.B) {
	const n = 5000
	data := make([]int, n)
	for i := range data {
		data[i] = i
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		src := FromSlice(data)
		d := Distinct(src, func(a, c int) bool { return a == c })
		for {
			if _, err := d.Recv(); err != nil {
				break
			}
		}
	}
}
