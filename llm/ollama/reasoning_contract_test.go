package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
)

func TestReasoningContract_OllamaCapabilityTriStateFromTagsAndShow(t *testing.T) {
	tests := []struct {
		name          string
		tagsResponse  string
		showResponse  string
		wantSupport   string
		wantShowCalls int
	}{
		{
			name:          "tags reports thinking",
			tagsResponse:  `{"models":[{"name":"reasoner:latest","capabilities":["completion","thinking"],"details":{"context_length":8192}}]}`,
			wantSupport:   "supported",
			wantShowCalls: 0,
		},
		{
			name:          "tags explicitly omits thinking",
			tagsResponse:  `{"models":[{"name":"plain:latest","capabilities":["completion"],"details":{"context_length":8192}}]}`,
			wantSupport:   "unsupported",
			wantShowCalls: 0,
		},
		{
			name:          "show reports thinking when tags is incomplete",
			tagsResponse:  `{"models":[{"name":"show-reasoner:latest"}]}`,
			showResponse:  `{"capabilities":["completion","thinking"]}`,
			wantSupport:   "supported",
			wantShowCalls: 1,
		},
		{
			name:          "show explicitly omits thinking",
			tagsResponse:  `{"models":[{"name":"show-plain:latest"}]}`,
			showResponse:  `{"capabilities":["completion"]}`,
			wantSupport:   "unsupported",
			wantShowCalls: 1,
		},
		{
			name:          "tags and show omit capability evidence",
			tagsResponse:  `{"models":[{"name":"unknown:latest"}]}`,
			showResponse:  `{}`,
			wantSupport:   "unknown",
			wantShowCalls: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			showCalls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/api/tags":
					_, _ = w.Write([]byte(tt.tagsResponse))
				case "/api/show":
					showCalls++
					_, _ = w.Write([]byte(tt.showResponse))
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			models := New(WithBaseURL(server.URL)).Models()
			if len(models) != 1 {
				t.Fatalf("Models() count = %d, want 1", len(models))
			}

			gotSupport, ok := modelReasoningSupport(models[0])
			if !ok {
				t.Errorf("model %q has no reasoning_support, want %q", models[0].ID, tt.wantSupport)
			} else if gotSupport != tt.wantSupport {
				t.Errorf("reasoning_support = %q, want %q", gotSupport, tt.wantSupport)
			}
			if showCalls != tt.wantShowCalls {
				t.Errorf("/api/show calls = %d, want %d", showCalls, tt.wantShowCalls)
			}
		})
	}
}

func TestReasoningContract_OllamaThinkBooleanWire(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{name: "true", want: true},
		{name: "false", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := New().buildRequestBody(llm.CompletionRequest{
				Model:    "reasoner:latest",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
				Metadata: map[string]any{"thinking": tt.want},
			}, true)
			if err != nil {
				t.Fatalf("buildRequestBody() error = %v", err)
			}

			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal request body: %v", err)
			}
			got, ok := payload["think"].(bool)
			if !ok {
				t.Fatalf("payload think type = %T, want bool; payload=%v", payload["think"], payload)
			}
			if got != tt.want {
				t.Errorf("payload think = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReasoningContract_OllamaThinkDoesNotConsumeThinkingEffort(t *testing.T) {
	body, err := New().buildRequestBody(llm.CompletionRequest{
		Model:    "reasoner:latest",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
		Metadata: map[string]any{
			"thinking":        "on",
			"thinking_effort": "max",
			llm.ReasoningCapabilityMetadataKey: llm.ReasoningCapability{
				Support:  llm.ReasoningSupported,
				Dialect:  llm.ReasoningDialectThink,
				OnValue:  true,
				OffValue: false,
			},
		},
	}, true)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if think, ok := payload["think"].(bool); !ok || !think {
		t.Fatalf("payload think=%#v, want bool(true); payload=%v", payload["think"], payload)
	}
	if _, exists := payload["thinking_effort"]; exists {
		t.Fatalf("Ollama payload consumed thinking_effort: %v", payload)
	}
	if _, exists := payload["reasoning_effort"]; exists {
		t.Fatalf("Ollama payload emitted reasoning_effort: %v", payload)
	}
}

func TestReasoningContract_OllamaUnknownCapabilityFallsBackToNativeBooleanThink(t *testing.T) {
	body, err := New().buildRequestBody(llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
		Metadata: map[string]any{
			"thinking": "on",
			llm.ReasoningCapabilityMetadataKey: llm.ReasoningCapability{
				Support: llm.ReasoningUnknown,
			},
		},
	}, true)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if think, ok := payload["think"].(bool); !ok || !think {
		t.Fatalf("unknown capability short-circuited native think=true: payload=%v", payload)
	}
}

func TestReasoningContract_OllamaNativeStreamUsesHostInjectedRouteEvidence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"untrusted-body-model","message":{"role":"assistant","thinking":"public summary"},"done":true}` + "\n"))
	}))
	defer server.Close()

	req := llm.CompletionRequest{
		Model:    "frozen-route-model",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	}
	InjectTrustedReasoningDisclosureEvidence(&req, "frozen-provider", "frozen-route-model")
	stream, err := New(WithBaseURL(server.URL)).Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var disclosure *streamx.ReasoningDisclosure
	for chunk := range stream.Chunks() {
		if chunk.ReasoningDisclosure != nil {
			copied := *chunk.ReasoningDisclosure
			disclosure = &copied
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	want := streamx.ReasoningDisclosure{
		Visibility: streamx.ReasoningVisible,
		Source:     "ollama",
		Dialect:    "message.thinking",
		Provider:   "frozen-provider",
		Model:      "frozen-route-model",
	}
	if disclosure == nil || *disclosure != want {
		t.Fatalf("native stream disclosure=%#v, want %#v", disclosure, want)
	}
}

func TestReasoningContract_OllamaNativeStreamDoesNotInferRouteFromWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"untrusted-body-model","message":{"role":"assistant","thinking":"private trace"},"done":true}` + "\n"))
	}))
	defer server.Close()

	stream, err := New(WithBaseURL(server.URL)).Stream(context.Background(), llm.CompletionRequest{
		Model:    "frozen-route-model",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var disclosure *streamx.ReasoningDisclosure
	for chunk := range stream.Chunks() {
		if chunk.ReasoningDisclosure != nil {
			copied := *chunk.ReasoningDisclosure
			disclosure = &copied
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	if disclosure == nil || disclosure.Visibility != streamx.ReasoningNotExposed {
		t.Fatalf("wire response made disclosure visible: %#v", disclosure)
	}
	if disclosure.Provider != "" || disclosure.Model != "" {
		t.Fatalf("wire response supplied route identity: %#v", disclosure)
	}
}

func TestReasoningContract_OllamaObservedThinkingProducesReceipt(t *testing.T) {
	chunk, err := (&ollamaStreamParser{}).Parse([]byte(
		`{"model":"reasoner:latest","message":{"role":"assistant","thinking":"checked the constraints"}}`,
	))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if chunk.Reasoning != "checked the constraints" {
		t.Fatalf("Reasoning = %q, want observed reasoning delta", chunk.Reasoning)
	}

	receipt, ok := reasoningReceipt(chunk)
	if !ok {
		t.Fatal("observed message.thinking must emit reasoning_evidence")
	}
	observed, ok := receipt["observed"].(bool)
	if !ok || !observed {
		t.Errorf("reasoning_evidence.observed = %#v, want true", receipt["observed"])
	}
}

// modelReasoningSupport 只读取对外 JSON 合同，避免测试依赖包内实现细节。
func modelReasoningSupport(model llm.ModelInfo) (string, bool) {
	data, err := json.Marshal(model)
	if err != nil {
		return "", false
	}
	var envelope struct {
		ReasoningSupport string `json:"reasoning_support"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil || envelope.ReasoningSupport == "" {
		return "", false
	}
	return envelope.ReasoningSupport, true
}

// reasoningReceipt 通过公开序列化结果验证执行证据，不读取 Provider 私有状态。
func reasoningReceipt(value any) (map[string]any, bool) {
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
