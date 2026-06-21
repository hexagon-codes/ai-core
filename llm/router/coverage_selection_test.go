package router

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

// mockProv is a minimal llm.Provider for exercising router selection strategies.
type mockProv struct {
	name     string
	models   []llm.ModelInfo
	completeErr error
	resp     *llm.CompletionResponse
}

func (m *mockProv) Name() string { return m.name }
func (m *mockProv) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if m.completeErr != nil {
		return nil, m.completeErr
	}
	if m.resp != nil {
		return m.resp, nil
	}
	return &llm.CompletionResponse{Model: req.Model, Content: m.name}, nil
}
func (m *mockProv) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if m.completeErr != nil {
		return nil, m.completeErr
	}
	return streamx.NewStream(nil, streamx.OpenAIFormat), nil
}
func (m *mockProv) Models() []llm.ModelInfo { return m.models }
func (m *mockProv) CountTokens(messages []llm.Message) (int, error) {
	return len(messages), nil
}

// TestRouter_LeastCostSelect covers leastCostSelect (was 0% in selection helpers).
func TestRouter_LeastCostSelect(t *testing.T) {
	cheap := &mockProv{name: "cheap", models: []llm.ModelInfo{{ID: "m", InputCost: 1, OutputCost: 1}}}
	pricey := &mockProv{name: "pricey", models: []llm.ModelInfo{{ID: "m", InputCost: 50, OutputCost: 50}}}
	r := New(WithStrategy(StrategyLeastCost))
	r.Register("pricey", pricey)
	r.Register("cheap", cheap)

	// Empty model => leastCostSelect compares all providers' models (no ID filter),
	// so it can actually pick the cheapest. (With a concrete-but-absent model ID,
	// leastCostSelect finds no match and degrades to available[0].)
	resp, err := r.Complete(context.Background(), llm.CompletionRequest{Model: ""})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if resp.Content != "cheap" {
		t.Errorf("leastCost should pick cheap, got %q", resp.Content)
	}
}

// TestRouter_LeastLatencySelect covers leastLatencySelect + recordLatency.
func TestRouter_LeastLatencySelect(t *testing.T) {
	fast := &mockProv{name: "fast", models: []llm.ModelInfo{{ID: "f"}}}
	slow := &mockProv{name: "slow", models: []llm.ModelInfo{{ID: "s"}}}
	r := New(WithStrategy(StrategyLeastLatency))
	r.Register("fast", fast)
	r.Register("slow", slow)

	// Seed latencies directly via recordLatency through GetStats path.
	r.recordLatency("fast", 5*time.Millisecond)
	r.recordLatency("slow", 500*time.Millisecond)

	resp, err := r.Complete(context.Background(), llm.CompletionRequest{Model: "absent"})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if resp.Content != "fast" {
		t.Errorf("leastLatency should pick fast, got %q", resp.Content)
	}
}

// TestRouter_WeightedSelect covers weightedSelect across many draws.
func TestRouter_WeightedSelect(t *testing.T) {
	a := &mockProv{name: "a", models: []llm.ModelInfo{{ID: "x"}}}
	b := &mockProv{name: "b", models: []llm.ModelInfo{{ID: "y"}}}
	r := New(WithStrategy(StrategyWeighted), WithWeights(map[string]int{"a": 9, "b": 1}))
	r.Register("a", a)
	r.Register("b", b)

	counts := map[string]int{}
	for i := 0; i < 200; i++ {
		resp, err := r.Complete(context.Background(), llm.CompletionRequest{Model: "absent"})
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		counts[resp.Content]++
	}
	if counts["a"] <= counts["b"] {
		t.Errorf("weighted(9:1) should favor a; got a=%d b=%d", counts["a"], counts["b"])
	}
}

// TestRouter_FallbackSelect covers fallbackSelect + SetHealthy.
func TestRouter_FallbackSelect(t *testing.T) {
	a := &mockProv{name: "a", models: []llm.ModelInfo{{ID: "x"}}}
	b := &mockProv{name: "b", models: []llm.ModelInfo{{ID: "y"}}}
	r := New(WithStrategy(StrategyFallback), WithHealthCheck(true))
	r.Register("a", a)
	r.Register("b", b)
	r.SetHealthy("a", false) // a is down

	resp, err := r.Complete(context.Background(), llm.CompletionRequest{Model: "absent"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if resp.Content != "b" {
		t.Errorf("fallbackSelect should skip unhealthy a and pick b, got %q", resp.Content)
	}
}

// TestRouter_ModelsAndCountTokens covers Models, CountTokens, Name (were 0%).
func TestRouter_ModelsAndCountTokens(t *testing.T) {
	a := &mockProv{name: "a", models: []llm.ModelInfo{{ID: "x"}, {ID: "shared"}}}
	b := &mockProv{name: "b", models: []llm.ModelInfo{{ID: "y"}, {ID: "shared"}}}
	r := New()
	r.Register("a", a)
	r.Register("b", b)

	if r.Name() != "router" {
		t.Errorf("Name()=%q", r.Name())
	}
	models := r.Models()
	// shared dedup => 3 unique
	if len(models) != 3 {
		t.Errorf("expected 3 deduped models, got %d", len(models))
	}
	n, err := r.CountTokens([]llm.Message{{Content: "a"}, {Content: "b"}})
	if err != nil || n != 2 {
		t.Errorf("CountTokens => %d, %v", n, err)
	}
}

// TestRouter_StreamFallback covers Stream + fallback-on-error path.
func TestRouter_StreamFallback(t *testing.T) {
	primary := &mockProv{name: "primary", models: []llm.ModelInfo{{ID: "m"}}, completeErr: errors.New("503 service unavailable")}
	fb := &mockProv{name: "fb", models: []llm.ModelInfo{{ID: "n"}}}
	r := New(WithFallback("fb"))
	r.Register("primary", primary)
	r.Register("fb", fb)
	r.modelMap["m"] = "primary"

	// primary.Stream errors → fallback used → non-nil stream returned
	s, err := r.Stream(context.Background(), llm.CompletionRequest{Model: "m"})
	if err != nil {
		t.Fatalf("expected fallback to succeed, got err %v", err)
	}
	if s == nil {
		t.Errorf("expected non-nil stream from fallback")
	}
}

// TestRouter_RouteByCapability covers RouteByCapability (was 0%).
func TestRouter_RouteByCapability(t *testing.T) {
	vision := &mockProv{name: "vision", models: []llm.ModelInfo{{ID: "v", Features: []string{"vision"}}}}
	text := &mockProv{name: "text", models: []llm.ModelInfo{{ID: "t"}}}
	r := New()
	r.Register("text", text)
	r.Register("vision", vision)

	p := RouteByCapability(r, "vision")
	if p == nil || p.Name() != "vision" {
		t.Errorf("RouteByCapability(vision) should return vision provider, got %v", p)
	}
	if RouteByCapability(r, "nonexistent") != nil {
		t.Errorf("RouteByCapability(nonexistent) should return nil")
	}
}

// TestRouter_UnregisterCleansModelMap covers Unregister fully.
func TestRouter_UnregisterCleansModelMap(t *testing.T) {
	a := &mockProv{name: "a", models: []llm.ModelInfo{{ID: "x"}}}
	r := New()
	r.Register("a", a)
	if _, ok := r.modelMap["x"]; !ok {
		t.Fatal("model x should be mapped after Register")
	}
	r.Unregister("a")
	if _, ok := r.providers["a"]; ok {
		t.Error("provider a should be gone")
	}
	if _, ok := r.modelMap["x"]; ok {
		t.Error("modelMap entry for x should be cleaned on Unregister")
	}
}
