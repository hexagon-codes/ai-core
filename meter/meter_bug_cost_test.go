package meter

import (
	"math"
	"testing"
)

// TestMeter_Bug_SubMicroCostTruncation verifies that many low-token calls whose
// per-record cost is below 1 micro-USD still accumulate into the global cost,
// and that Stats().EstimatedCost agrees with StatsByModel().EstimatedCost.
func TestMeter_Bug_SubMicroCostTruncation(t *testing.T) {
	m := New()
	const model = "gpt-4o-mini" // InputPrice 0.15 /M => 1 input token = 0.15 micro-USD

	// 10000 single-input-token calls. Each per-record cost is 0.15 micro-USD,
	// which truncates to 0 under int64(...) per record, but should sum to a
	// non-zero global cost.
	const n = 10000
	for i := 0; i < n; i++ {
		m.Record(model, 1, 0)
	}

	global := m.Stats().EstimatedCost
	byModel := m.StatsByModel(model).EstimatedCost

	// Expected cost: n input tokens * 0.15 USD / 1M tokens.
	want := float64(n) / 1_000_000 * 0.15

	if global <= 0 {
		t.Fatalf("global EstimatedCost truncated to %v, want ~%v", global, want)
	}
	if math.Abs(global-byModel) > 1e-9 {
		t.Errorf("global (%v) and by-model (%v) cost disagree (diff %v)",
			global, byModel, math.Abs(global-byModel))
	}
	if math.Abs(global-want) > 1e-9 {
		t.Errorf("global EstimatedCost = %v, want ~%v", global, want)
	}
}
