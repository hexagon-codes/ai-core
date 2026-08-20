package llm

import (
	"errors"
	"reflect"
	"testing"
)

func TestReasoningContract_StableValues(t *testing.T) {
	supports := map[ReasoningSupport]string{
		ReasoningSupported:   "supported",
		ReasoningUnsupported: "unsupported",
		ReasoningUnknown:     "unknown",
	}
	for support, want := range supports {
		if got := string(support); got != want {
			t.Errorf("ReasoningSupport %q = %q, want %q", support, got, want)
		}
	}

	dialects := map[ReasoningDialect]string{
		ReasoningDialectEffort:         "reasoning_effort",
		ReasoningDialectEnableThinking: "enable_thinking",
		ReasoningDialectThink:          "think",
		ReasoningDialectThinking:       "thinking",
	}
	for dialect, want := range dialects {
		if got := string(dialect); got != want {
			t.Errorf("ReasoningDialect %q = %q, want %q", dialect, got, want)
		}
	}
}

func TestReasoningPlan_SupportedOnOffUsesExactWire(t *testing.T) {
	tests := []struct {
		name       string
		enabled    bool
		capability ReasoningCapability
		wantWire   map[string]any
	}{
		{
			name:    "effort_on",
			enabled: true,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectEffort,
				OnValue:  "high",
				OffValue: "none",
			},
			wantWire: map[string]any{"reasoning_effort": "high"},
		},
		{
			name:    "effort_off",
			enabled: false,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectEffort,
				OnValue:  "high",
				OffValue: "none",
			},
			wantWire: map[string]any{"reasoning_effort": "none"},
		},
		{
			name:    "enable_thinking_on",
			enabled: true,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectEnableThinking,
				OnValue:  true,
				OffValue: false,
			},
			wantWire: map[string]any{"enable_thinking": true},
		},
		{
			name:    "enable_thinking_off",
			enabled: false,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectEnableThinking,
				OnValue:  true,
				OffValue: false,
			},
			wantWire: map[string]any{"enable_thinking": false},
		},
		{
			name:    "ollama_think_on",
			enabled: true,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectThink,
				OnValue:  "medium",
				OffValue: false,
			},
			wantWire: map[string]any{"think": "medium"},
		},
		{
			name:    "ollama_think_off",
			enabled: false,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectThink,
				OnValue:  "medium",
				OffValue: false,
			},
			wantWire: map[string]any{"think": false},
		},
		{
			name:    "anthropic_thinking_on",
			enabled: true,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectThinking,
				OnValue:  map[string]any{"type": "enabled", "budget_tokens": 1024},
				OffValue: map[string]any{"type": "disabled"},
			},
			wantWire: map[string]any{
				"thinking": map[string]any{"type": "enabled", "budget_tokens": 1024},
			},
		},
		{
			name:    "anthropic_thinking_off",
			enabled: false,
			capability: ReasoningCapability{
				Support:  ReasoningSupported,
				Dialect:  ReasoningDialectThinking,
				OnValue:  map[string]any{"type": "enabled", "budget_tokens": 1024},
				OffValue: map[string]any{"type": "disabled"},
			},
			wantWire: map[string]any{
				"thinking": map[string]any{"type": "disabled"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := planReasoning(ReasoningPlanRequest{
				Enabled:    tt.enabled,
				Model:      "configured-model",
				BaseURL:    "https://configured.example/v1",
				Capability: tt.capability,
			})
			if err != nil {
				t.Fatalf("planReasoning() error = %v", err)
			}
			if !reflect.DeepEqual(plan.Wire, tt.wantWire) {
				t.Fatalf("wire = %#v, want exact %#v", plan.Wire, tt.wantWire)
			}
			if plan.Receipt.Enabled != tt.enabled {
				t.Errorf("receipt enabled = %v, want %v", plan.Receipt.Enabled, tt.enabled)
			}
			if plan.Receipt.Support != ReasoningSupported {
				t.Errorf("receipt support = %q, want %q", plan.Receipt.Support, ReasoningSupported)
			}
			if plan.Receipt.Dialect != tt.capability.Dialect {
				t.Errorf("receipt dialect = %q, want %q", plan.Receipt.Dialect, tt.capability.Dialect)
			}
			if !plan.Receipt.Sent {
				t.Error("receipt sent = false, want true")
			}
			if plan.Receipt.Accepted || plan.Receipt.Observed || plan.Receipt.Applied {
				t.Fatalf("new receipt = %#v, sent alone must not imply accepted, observed, or applied", plan.Receipt)
			}
		})
	}
}

func TestReasoningPlan_UnsupportedOnReturnsTypedReject(t *testing.T) {
	plan, err := planReasoning(ReasoningPlanRequest{
		Enabled: true,
		Model:   "configured-model",
		BaseURL: "https://configured.example/v1",
		Capability: ReasoningCapability{
			Support: ReasoningUnsupported,
		},
	})
	if err == nil {
		t.Fatal("planReasoning() error = nil, want typed unsupported rejection")
	}

	var unsupported *ReasoningUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error type = %T, want *ReasoningUnsupportedError", err)
	}
	if len(plan.Wire) != 0 {
		t.Fatalf("wire = %#v, want empty wire for rejected request", plan.Wire)
	}
	if plan.Receipt.Support != ReasoningUnsupported {
		t.Errorf("receipt support = %q, want %q", plan.Receipt.Support, ReasoningUnsupported)
	}
	if plan.Receipt.Sent || plan.Receipt.Accepted || plan.Receipt.Observed || plan.Receipt.Applied {
		t.Fatalf("rejected receipt = %#v, want no upstream side effect", plan.Receipt)
	}
}

func TestReasoningPlan_NonSupportedOnOffMatrix(t *testing.T) {
	tests := []struct {
		name            string
		support         ReasoningSupport
		enabled         bool
		wantRejected    bool
		wantApplication ReasoningApplication
	}{
		{name: "unsupported_on", support: ReasoningUnsupported, enabled: true, wantRejected: true, wantApplication: ReasoningApplicationRejected},
		{name: "unsupported_off", support: ReasoningUnsupported, enabled: false, wantApplication: ReasoningApplicationUnknown},
		{name: "unknown_on", support: ReasoningUnknown, enabled: true, wantApplication: ReasoningApplicationUnknown},
		{name: "unknown_off", support: ReasoningUnknown, enabled: false, wantApplication: ReasoningApplicationUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := planReasoning(ReasoningPlanRequest{
				Enabled: tt.enabled,
				Model:   "configured-model",
				Capability: ReasoningCapability{
					Support:  tt.support,
					Dialect:  ReasoningDialectEffort,
					OnValue:  "high",
					OffValue: "none",
				},
			})
			var unsupported *ReasoningUnsupportedError
			if got := errors.As(err, &unsupported); got != tt.wantRejected {
				t.Fatalf("unsupported error = %v, want %v; error = %v", got, tt.wantRejected, err)
			}
			if len(plan.Wire) != 0 || plan.Receipt.Sent {
				t.Fatalf("plan = %#v, non-supported capability must not send a control field", plan)
			}
			if plan.Receipt.Enabled != tt.enabled || plan.Receipt.Support != tt.support {
				t.Fatalf("receipt = %#v, request/support snapshot drifted", plan.Receipt)
			}
			if plan.Receipt.Application != tt.wantApplication {
				t.Fatalf("application = %q, want %q", plan.Receipt.Application, tt.wantApplication)
			}
		})
	}
}

func TestReasoningPlan_UnknownDoesNotSendOrInferFromModelOrURL(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		baseURL string
	}{
		{
			name:    "openai_model_name",
			model:   "gpt-5.6-luna",
			baseURL: "https://api.openai.com/v1",
		},
		{
			name:    "qwen_model_and_dashscope_url",
			model:   "qwen3.5-plus",
			baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
		},
		{
			name:    "reasoning_model_and_siliconflow_url",
			model:   "deepseek-r1",
			baseURL: "https://api.siliconflow.cn/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := planReasoning(ReasoningPlanRequest{
				Enabled: true,
				Model:   tt.model,
				BaseURL: tt.baseURL,
				Capability: ReasoningCapability{
					Support:  ReasoningUnknown,
					Dialect:  ReasoningDialectEffort,
					OnValue:  "high",
					OffValue: "none",
				},
			})
			if err != nil {
				t.Fatalf("planReasoning() error = %v", err)
			}
			if len(plan.Wire) != 0 {
				t.Fatalf("wire = %#v, unknown capability must not send inferred fields", plan.Wire)
			}
			if plan.Receipt.Support != ReasoningUnknown {
				t.Errorf("receipt support = %q, want %q", plan.Receipt.Support, ReasoningUnknown)
			}
			if plan.Receipt.Sent || plan.Receipt.Accepted || plan.Receipt.Observed || plan.Receipt.Applied {
				t.Fatalf("unknown receipt = %#v, want no inferred execution facts", plan.Receipt)
			}
		})
	}
}

func TestReasoningReceipt_SentAndAcceptedDoNotMeanApplied(t *testing.T) {
	plan, err := planReasoning(ReasoningPlanRequest{
		Enabled: true,
		Capability: ReasoningCapability{
			Support:  ReasoningSupported,
			Dialect:  ReasoningDialectEffort,
			OnValue:  "high",
			OffValue: "none",
		},
	})
	if err != nil {
		t.Fatalf("planReasoning() error = %v", err)
	}
	if !plan.Receipt.Sent || plan.Receipt.Applied {
		t.Fatalf("planned receipt = %#v, sent must not imply applied", plan.Receipt)
	}

	accepted := plan.Receipt.WithEvidence(true, false, 0)
	if !accepted.Accepted {
		t.Fatal("accepted receipt = false, want true")
	}
	if accepted.Observed || accepted.Applied {
		t.Fatalf("accepted receipt = %#v, acceptance without observation must not imply applied", accepted)
	}
}

func TestReasoningReceipt_ObservedDeltaOrTokensMeansApplied(t *testing.T) {
	base := ReasoningReceipt{
		Enabled: true,
		Support: ReasoningSupported,
		Dialect: ReasoningDialectEffort,
		Sent:    true,
	}

	tests := []struct {
		name            string
		observedDelta   bool
		reasoningTokens int
	}{
		{name: "reasoning_delta", observedDelta: true},
		{name: "reasoning_tokens", reasoningTokens: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := base.WithEvidence(true, tt.observedDelta, tt.reasoningTokens)
			if !got.Accepted || !got.Observed || !got.Applied {
				t.Fatalf("receipt = %#v, observed reasoning evidence must imply applied", got)
			}
		})
	}
}

func TestReasoningReceipt_OnOffEvidenceMatrix(t *testing.T) {
	tests := []struct {
		name            string
		enabled         bool
		observed        bool
		wantApplication ReasoningApplication
		wantApplied     bool
	}{
		{name: "on_without_evidence", enabled: true, wantApplication: ReasoningApplicationUnknown},
		{name: "on_with_evidence", enabled: true, observed: true, wantApplication: ReasoningApplicationApplied, wantApplied: true},
		{name: "off_without_evidence", enabled: false, wantApplication: ReasoningApplicationUnknown},
		{name: "off_with_evidence", enabled: false, observed: true, wantApplication: ReasoningApplicationIgnored},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := (ReasoningReceipt{
				Enabled:     tt.enabled,
				Support:     ReasoningSupported,
				Dialect:     ReasoningDialectEffort,
				Sent:        true,
				Application: ReasoningApplicationUnknown,
			}).WithEvidence(true, tt.observed, 0)
			if got.Application != tt.wantApplication || got.Applied != tt.wantApplied {
				t.Fatalf("receipt = %#v, want application=%q applied=%v", got, tt.wantApplication, tt.wantApplied)
			}
		})
	}
}
