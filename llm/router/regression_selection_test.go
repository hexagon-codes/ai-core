package router

import (
	"context"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestLeastCostSelectChoosesProviderSupportingModel(t *testing.T) {
	t.Parallel()

	r := &Router{providers: map[string]llm.Provider{
		"first": &mockRouterProvider{
			name:   "first",
			models: []llm.ModelInfo{{ID: "other", InputCost: 1, OutputCost: 1}},
		},
		"second": &mockRouterProvider{
			name:   "second",
			models: []llm.ModelInfo{{ID: "target", InputCost: 10, OutputCost: 10}},
		},
	}}

	selected, err := r.leastCostSelect([]string{"first", "second"}, "target")
	if err != nil {
		t.Fatalf("leastCostSelect() error = %v", err)
	}
	if selected == nil || selected.Name() != "second" {
		t.Fatalf("leastCostSelect() = %v, want second", selected)
	}
}

func TestSmartRouterHardCostConstraintFiltersCandidate(t *testing.T) {
	t.Parallel()

	r := &SmartRouter{Router: New()}
	candidate := candidate{
		providerName: "expensive",
		modelInfo: llm.ModelInfo{
			ID:         "expensive-model",
			InputCost:  100,
			OutputCost: 100,
		},
	}
	routingContext := NewRoutingContext(TaskTypeChat, ComplexitySimple).WithMaxCost(0.000001)

	if r.meetsHardConstraints(candidate, routingContext, llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	}) {
		t.Fatal("meetsHardConstraints() allowed a candidate above MaxCostPerRequest")
	}
}

type fixedClassifierProvider struct {
	content string
}

func (p *fixedClassifierProvider) Name() string { return "fixed-classifier" }
func (p *fixedClassifierProvider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{{ID: "classifier-model"}}
}
func (p *fixedClassifierProvider) CountTokens([]llm.Message) (int, error) { return 0, nil }
func (p *fixedClassifierProvider) Complete(context.Context, llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return &llm.CompletionResponse{Content: p.content}, nil
}
func (p *fixedClassifierProvider) Stream(context.Context, llm.CompletionRequest) (*llm.Stream, error) {
	return nil, nil
}

func TestCompositeClassifierThresholdControlsLLMFallback(t *testing.T) {
	t.Parallel()

	llmClassifier := NewLLMClassifier(&fixedClassifierProvider{
		content: `{"task_type":"reasoning","complexity":"complex"}`,
	})
	request := llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "请修复这段 Go 代码"}},
	}

	lowThreshold := NewCompositeClassifier(&fixedClassifierProvider{content: "unused"})
	lowThreshold.llmClassifier = llmClassifier
	lowThreshold.confidenceThreshold = 0.7
	if taskType, _ := lowThreshold.Classify(context.Background(), request); taskType != TaskTypeCoding {
		t.Fatalf("low-threshold Classify() = %s, want coding", taskType)
	}

	highThreshold := NewCompositeClassifier(&fixedClassifierProvider{content: "unused"})
	highThreshold.llmClassifier = llmClassifier
	highThreshold.confidenceThreshold = 0.95
	if taskType, _ := highThreshold.Classify(context.Background(), request); taskType != TaskTypeReasoning {
		t.Fatalf("high-threshold Classify() = %s, want reasoning", taskType)
	}
}

func TestRouterUnregisterOnEmptyListDoesNotPanic(t *testing.T) {
	t.Parallel()

	r := New()
	r.Unregister("missing")
	r.Unregister("missing")
}

func TestLLMClassifierParsesSimpleComplexity(t *testing.T) {
	t.Parallel()

	classifier := NewLLMClassifier(&fixedClassifierProvider{})
	taskType, complexity := classifier.parseResponse(`{"task_type":"coding","complexity":"simple"}`)
	if taskType != TaskTypeCoding || complexity != ComplexitySimple {
		t.Fatalf("parseResponse() = %s/%s, want coding/simple", taskType, complexity)
	}
}
