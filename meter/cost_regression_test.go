package meter

import "testing"

func TestStatsByModelCostMatchesGlobalAfterPruning(t *testing.T) {
	t.Parallel()

	m := NewWithOptions(map[string]Pricing{
		"model-x": {InputPrice: 10, OutputPrice: 20},
	}, 10)
	for i := 0; i < 20; i++ {
		m.Record("model-x", 1_000_000, 1_000_000)
	}

	globalCost := m.Stats().EstimatedCost
	modelCost := m.StatsByModel("model-x").EstimatedCost
	if modelCost != globalCost {
		t.Fatalf("StatsByModel cost = %v, global = %v", modelCost, globalCost)
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
