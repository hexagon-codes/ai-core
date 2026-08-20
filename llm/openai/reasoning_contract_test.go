package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

const reasoningCapabilityMetadataKey = "reasoning_capability_v1"

func TestReasoningContract_ExplicitCapabilityControlsWireAcrossBaseURLs(t *testing.T) {
	tests := []struct {
		name           string
		intent         string
		dialect        string
		onValue        any
		offValue       any
		wantField      string
		wantValue      any
		forbiddenField string
	}{
		{
			name:           "reasoning effort on",
			intent:         "on",
			dialect:        "reasoning_effort",
			onValue:        "high",
			offValue:       "none",
			wantField:      "reasoning_effort",
			wantValue:      "high",
			forbiddenField: "enable_thinking",
		},
		{
			name:           "reasoning effort off",
			intent:         "off",
			dialect:        "reasoning_effort",
			onValue:        "high",
			offValue:       "none",
			wantField:      "reasoning_effort",
			wantValue:      "none",
			forbiddenField: "enable_thinking",
		},
		{
			name:           "enable thinking on",
			intent:         "on",
			dialect:        "enable_thinking",
			onValue:        true,
			offValue:       false,
			wantField:      "enable_thinking",
			wantValue:      true,
			forbiddenField: "reasoning_effort",
		},
		{
			name:           "enable thinking off",
			intent:         "off",
			dialect:        "enable_thinking",
			onValue:        true,
			offValue:       false,
			wantField:      "enable_thinking",
			wantValue:      false,
			forbiddenField: "reasoning_effort",
		},
	}

	for _, endpoint := range []struct {
		name       string
		customHost string
	}{
		{name: "localhost"},
		{name: "custom URL", customHost: "custom-reasoning-gateway.invalid"},
	} {
		for _, tt := range tests {
			t.Run(endpoint.name+"/"+tt.name, func(t *testing.T) {
				payload, _ := completeReasoningContractRequest(t, endpoint.customHost, llm.CompletionRequest{
					Model:    "opaque-contract-model",
					Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
					Metadata: reasoningCapabilityMetadata("supported", tt.dialect, tt.intent, tt.onValue, tt.offValue),
				}, plainReasoningContractResponse)

				if got := payload[tt.wantField]; got != tt.wantValue {
					t.Fatalf("%s = %#v, want %#v; payload=%v", tt.wantField, got, tt.wantValue, payload)
				}
				if _, exists := payload[tt.forbiddenField]; exists {
					t.Fatalf("explicit %s capability must not emit %s; payload=%v", tt.dialect, tt.forbiddenField, payload)
				}
			})
		}
	}
}

func TestReasoningContract_UnknownCapabilityDoesNotInjectWireFields(t *testing.T) {
	tests := []struct {
		name       string
		customHost string
		model      string
		dialect    string
		scope      llm.ReasoningPolicyScope
	}{
		{
			name:    "model name must not infer reasoning effort",
			model:   "gpt-5.6-contract",
			dialect: "reasoning_effort",
			scope:   llm.ReasoningPolicyScopeStructuredVisionRecognition,
		},
		{
			name:       "base URL must not infer enable thinking",
			customHost: "siliconflow-gateway.invalid",
			model:      "opaque-contract-model",
			dialect:    "enable_thinking",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload, _ := completeReasoningContractRequest(t, tt.customHost, llm.CompletionRequest{
				Model:                tt.model,
				Messages:             []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
				Metadata:             reasoningCapabilityMetadata("unknown", tt.dialect, "on", "high", "none"),
				ReasoningPolicyScope: tt.scope,
			}, plainReasoningContractResponse)

			for _, field := range []string{"reasoning_effort", "enable_thinking"} {
				if value, exists := payload[field]; exists {
					t.Fatalf("unknown capability must not emit %s=%#v; payload=%v", field, value, payload)
				}
			}
		})
	}
}

func TestReasoningContract_ExecutionReceiptRequiresObservedEvidence(t *testing.T) {
	tests := []struct {
		name            string
		responseBody    string
		wantObserved    bool
		wantApplication string
	}{
		{
			name:            "HTTP 2xx only accepts the request",
			responseBody:    plainReasoningContractResponse,
			wantObserved:    false,
			wantApplication: "unknown",
		},
		{
			name: "reasoning content proves application",
			responseBody: `{
				"id":"resp_reasoning_content",
				"model":"opaque-contract-model",
				"choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning_content":"reason"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
			}`,
			wantObserved:    true,
			wantApplication: "applied",
		},
		{
			name: "reasoning tokens prove application",
			responseBody: `{
				"id":"resp_reasoning_tokens",
				"model":"opaque-contract-model",
				"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"completion_tokens_details":{"reasoning_tokens":12}}
			}`,
			wantObserved:    true,
			wantApplication: "applied",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, response := completeReasoningContractRequest(t, "", llm.CompletionRequest{
				Model:    "opaque-contract-model",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
				Metadata: reasoningCapabilityMetadata("supported", "reasoning_effort", "on", "high", "none"),
			}, tt.responseBody)

			receipt := reasoningReceiptFromResponse(t, response)
			assertReasoningReceiptField(t, receipt, "accepted", true)
			assertReasoningReceiptField(t, receipt, "observed", tt.wantObserved)
			assertReasoningReceiptField(t, receipt, "application", tt.wantApplication)
		})
	}
}

const plainReasoningContractResponse = `{
	"id":"resp_plain",
	"model":"opaque-contract-model",
	"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
}`

func reasoningCapabilityMetadata(support, dialect, intent string, onValue, offValue any) map[string]any {
	return map[string]any{
		"thinking": intent,
		reasoningCapabilityMetadataKey: map[string]any{
			"support": support,
			"dialect": dialect,
			"on":      onValue,
			"off":     offValue,
		},
	}
}

func completeReasoningContractRequest(
	t *testing.T,
	customHost string,
	req llm.CompletionRequest,
	responseBody string,
) (map[string]any, *llm.CompletionResponse) {
	t.Helper()

	type requestCapture struct {
		payload map[string]any
		err     error
	}
	captured := make(chan requestCapture, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		err := json.NewDecoder(r.Body).Decode(&payload)
		captured <- requestCapture{payload: payload, err: err}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(responseBody))
	}))
	t.Cleanup(server.Close)

	baseURL := server.URL + "/v1"
	client := server.Client()
	if customHost != "" {
		transport := &http.Transport{
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
			},
		}
		t.Cleanup(transport.CloseIdleConnections)
		client = &http.Client{Transport: transport}
		baseURL = fmt.Sprintf("http://%s/v1", customHost)
	}

	response, err := New(
		"test-key",
		WithBaseURL(baseURL),
		WithHTTPClient(client),
	).Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	select {
	case capture := <-captured:
		if capture.err != nil {
			t.Fatalf("decode request body: %v", capture.err)
		}
		return capture.payload, response
	default:
		t.Fatal("httptest server did not capture a request")
		return nil, nil
	}
}

func reasoningReceiptFromResponse(t *testing.T, response *llm.CompletionResponse) map[string]any {
	t.Helper()
	raw, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal completion response: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("decode completion response: %v", err)
	}
	receipt, ok := payload["reasoning_evidence"].(map[string]any)
	if !ok {
		t.Fatalf("CompletionResponse must expose internal reasoning_evidence; response=%s", raw)
	}
	return receipt
}

func assertReasoningReceiptField(t *testing.T, receipt map[string]any, field string, want any) {
	t.Helper()
	if got := receipt[field]; got != want {
		t.Fatalf("reasoning_evidence.%s = %#v, want %#v; evidence=%v", field, got, want, receipt)
	}
}
