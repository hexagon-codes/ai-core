package meter

import (
	"math"
	"testing"
)

func TestStatsByModelCostMatchesRetainedRecordsAfterPruning(t *testing.T) {
	t.Parallel()

	m := NewWithOptions(map[string]Pricing{
		"model-x": {InputPrice: 10, OutputPrice: 20},
	}, 10)
	for i := 0; i < 20; i++ {
		m.Record("model-x", 1_000_000, 1_000_000)
	}

	byModel := m.StatsByModel("model-x")
	want := float64(byModel.InputTokens)/1_000_000*10 + float64(byModel.OutputTokens)/1_000_000*20
	if math.Abs(byModel.EstimatedCost-want) > 1e-9 {
		t.Fatalf("StatsByModel cost = %v, want %v from retained tokens", byModel.EstimatedCost, want)
	}
	if byModel.InputTokens != 10_000_000 {
		t.Fatalf("StatsByModel input tokens = %d, want 10 retained records", byModel.InputTokens)
	}

	global := m.Stats().EstimatedCost
	if global <= byModel.EstimatedCost {
		t.Fatalf("global cumulative cost %v should exceed retained cost %v", global, byModel.EstimatedCost)
	}
}

func TestMeterClearResetsModelCosts(t *testing.T) {
	t.Parallel()

	m := NewWithOptions(map[string]Pricing{
		"model-x": {InputPrice: 10, OutputPrice: 20},
	}, 10)
	m.Record("model-x", 1_000_000, 0)
	m.Clear()
	if cost := m.StatsByModel("model-x").EstimatedCost; cost != 0 {
		t.Fatalf("StatsByModel cost after Clear = %v, want 0", cost)
	}
}
