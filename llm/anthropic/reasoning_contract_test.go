package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

func TestReasoningContract_AnthropicExplicitModes(t *testing.T) {
	tests := []struct {
		name       string
		metadata   map[string]any
		wantType   string
		wantBudget float64
	}{
		{
			name: "enabled",
			metadata: map[string]any{
				"reasoning_v1": map[string]any{"mode": "enabled", "budget_tokens": 1024},
			},
			wantType:   "enabled",
			wantBudget: 1024,
		},
		{
			name: "adaptive",
			metadata: map[string]any{
				"reasoning_v1": map[string]any{"mode": "adaptive"},
			},
			wantType: "adaptive",
		},
		{
			name: "disabled",
			metadata: map[string]any{
				"reasoning_v1": map[string]any{"mode": "disabled"},
			},
			wantType: "disabled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, _, err := New("test-key").buildRequestBody(llm.CompletionRequest{
				Model:     "claude-contract-model",
				Messages:  []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
				MaxTokens: 4096,
				Metadata:  tt.metadata,
			}, true)
			if err != nil {
				t.Fatalf("buildRequestBody() error = %v", err)
			}

			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal request body: %v", err)
			}
			thinking, ok := payload["thinking"].(map[string]any)
			if !ok {
				t.Fatalf("payload thinking type = %T, want object; payload=%v", payload["thinking"], payload)
			}
			if got := thinking["type"]; got != tt.wantType {
				t.Errorf("thinking.type = %#v, want %q", got, tt.wantType)
			}
			if tt.wantBudget > 0 {
				if got := thinking["budget_tokens"]; got != tt.wantBudget {
					t.Errorf("thinking.budget_tokens = %#v, want %.0f", got, tt.wantBudget)
				}
			} else if _, exists := thinking["budget_tokens"]; exists {
				t.Errorf("thinking.budget_tokens must be absent for mode %q", tt.wantType)
			}
		})
	}
}

func TestReasoningContract_AnthropicThinkingDeltaIsReasoning(t *testing.T) {
	chunk, err := (&streamx.ClaudeParser{}).Parse([]byte(
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"checked the constraints"}}`,
	))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if chunk.Reasoning != "checked the constraints" {
		t.Errorf("Reasoning = %q, want thinking delta", chunk.Reasoning)
	}
	if chunk.Content != "" {
		t.Errorf("Content = %q, thinking delta must not leak into answer content", chunk.Content)
	}
}

func TestReasoningContract_AnthropicSSEErrorIsTerminal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"event: error\n" +
				`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` + "\n\n",
		))
	}))
	defer server.Close()

	stream, err := New("test-key", WithBaseURL(server.URL)).Stream(context.Background(), llm.CompletionRequest{
		Model:    "claude-contract-model",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Stream() setup error = %v", err)
	}
	result, err := stream.Collect()
	if err == nil {
		t.Fatalf("Collect() error = nil, want terminal overloaded_error; result=%+v", result)
	}
	if !strings.Contains(err.Error(), "overloaded_error") {
		t.Errorf("Collect() error = %q, want overloaded_error", err)
	}
}

func TestReasoningContract_AnthropicNoEvidenceStaysUnknown(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			`data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","model":"claude-contract-model"}}` + "\n\n" +
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}` + "\n\n" +
				`data: {"type":"message_stop"}` + "\n\n",
		))
	}))
	defer server.Close()

	stream, err := New("test-key", WithBaseURL(server.URL)).Stream(context.Background(), llm.CompletionRequest{
		Model:    "claude-contract-model",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
		Metadata: map[string]any{
			"reasoning_v1": map[string]any{"mode": "enabled", "budget_tokens": 1024},
		},
	})
	if err != nil {
		t.Fatalf("Stream() setup error = %v", err)
	}
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if result.Content != "answer" {
		t.Fatalf("Content = %q, want answer", result.Content)
	}

	receipt, ok := anthropicReasoningReceipt(result)
	if !ok {
		t.Fatal("reasoning request without upstream evidence must emit reasoning_evidence")
	}
	if got := receipt["application"]; got != "unknown" {
		t.Errorf("reasoning_evidence.application = %#v, want unknown", got)
	}
	if observed, _ := receipt["observed"].(bool); observed {
		t.Errorf("reasoning_evidence.observed = true, want false; evidence=%v", receipt)
	}
}

// anthropicReasoningReceipt 只读取对外 JSON 合同，避免测试耦合实现结构。
func anthropicReasoningReceipt(value any) (map[string]any, bool) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var envelope struct {
		Receipt map[string]any `json:"reasoning_evidence"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil || envelope.Receipt == nil {
		return nil, false
	}
	return envelope.Receipt, true
}
