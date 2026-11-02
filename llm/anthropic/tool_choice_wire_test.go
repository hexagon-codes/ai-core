package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func anthropicChoiceSchema(t *testing.T) *llm.Schema {
	t.Helper()
	var result llm.Schema
	if err := json.Unmarshal([]byte(`{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}`), &result); err != nil {
		t.Fatal(err)
	}
	return &result
}

func TestAnthropicToolChoiceActualHTTP(t *testing.T) {
	for _, test := range []struct {
		name               string
		choice             any
		wantType, wantName string
	}{
		{name: "auto", choice: "auto", wantType: "auto"},
		{name: "none", choice: "none", wantType: "none"},
		{name: "required", choice: "required", wantType: "any"},
		{name: "named", choice: map[string]any{"type": "function", "function": map[string]any{"name": "emit_schema"}}, wantType: "tool", wantName: "emit_schema"},
		{name: "named string", choice: "emit_schema", wantType: "tool", wantName: "emit_schema"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var wire struct {
					ToolChoice map[string]any `json:"tool_choice"`
					Tools      []struct {
						Name        string         `json:"name"`
						InputSchema map[string]any `json:"input_schema"`
					} `json:"tools"`
					MaxTokens   int               `json:"max_tokens"`
					Temperature float64           `json:"temperature"`
					TopP        float64           `json:"top_p"`
					Stop        []string          `json:"stop_sequences"`
					Metadata    map[string]string `json:"metadata"`
				}
				if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				if wire.ToolChoice["type"] != test.wantType || wire.ToolChoice["name"] != func() any {
					if test.wantName != "" {
						return test.wantName
					}
					return nil
				}() {
					t.Errorf("tool choice wire=%v", wire.ToolChoice)
				}
				if len(wire.Tools) != 1 || wire.Tools[0].Name != "emit_schema" || wire.Tools[0].InputSchema["type"] != "object" {
					t.Errorf("tool schema wire=%+v", wire.Tools)
				}
				if wire.MaxTokens != 256 || wire.Temperature != 0.2 || wire.TopP != 0.8 || len(wire.Stop) != 1 || wire.Stop[0] != "END" || wire.Metadata["user_id"] != "synthetic-user" {
					t.Errorf("generation parameters lost: %+v", wire)
				}
				w.Header().Set("Content-Type", "application/json")
				if test.wantType == "any" || test.wantType == "tool" {
					fmt.Fprint(w, `{"id":"synthetic","model":"actual","role":"assistant","content":[{"type":"tool_use","id":"call-schema","name":"emit_schema","input":{"title":"synthetic"}}],"stop_reason":"tool_use","usage":{"input_tokens":2,"output_tokens":1}}`)
				} else {
					fmt.Fprint(w, `{"id":"synthetic","model":"actual","role":"assistant","content":[{"type":"text","text":"{\"title\":\"synthetic\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
				}
			}))
			defer server.Close()
			temperature, topP := 0.2, 0.8
			response, err := New("synthetic", WithBaseURL(server.URL)).Complete(t.Context(), llm.CompletionRequest{Model: "claude-synthetic", Messages: []llm.Message{{Role: llm.RoleUser, Content: "return schema result"}}, Tools: []llm.ToolDefinition{llm.NewToolDefinition("emit_schema", "emit schema", anthropicChoiceSchema(t))}, ToolChoice: test.choice, MaxTokens: 256, Temperature: &temperature, TopP: &topP, Stop: []string{"END"}, User: "synthetic-user"})
			if err != nil {
				t.Fatal(err)
			}
			data := response.Content
			if test.wantType == "any" || test.wantType == "tool" {
				if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "emit_schema" {
					t.Fatalf("tool result missing: %+v", response)
				}
				data = response.ToolCalls[0].Arguments
			}
			var result struct {
				Title string `json:"title"`
			}
			if json.Unmarshal([]byte(data), &result) != nil || result.Title != "synthetic" {
				t.Fatalf("schema result=%q", data)
			}
		})
	}
}

func TestAnthropicNativeSchemaAndUnsupportedObjectFormat(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ordinal := calls.Add(1)
		var wire struct {
			OutputConfig struct {
				Format struct {
					Type   string         `json:"type"`
					Schema map[string]any `json:"schema"`
				} `json:"format"`
			} `json:"output_config"`
		}
		if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
			t.Error(err)
		}
		if ordinal == 1 && (wire.OutputConfig.Format.Type != "json_schema" || wire.OutputConfig.Format.Schema["type"] != "object") {
			t.Errorf("native output format=%+v", wire)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"schema","model":"actual","content":[{"type":"text","text":"{\"title\":\"synthetic\"}"}],"stop_reason":"end_turn"}`)
	}))
	defer server.Close()
	provider := New("synthetic", WithBaseURL(server.URL))
	request := llm.CompletionRequest{Model: "claude-synthetic", Messages: []llm.Message{{Role: llm.RoleUser, Content: "schema"}}, ResponseFormat: &llm.ResponseFormat{Type: "json_schema", JSONSchema: &llm.ResponseFormatJSONSchema{Name: "schema", Schema: anthropicChoiceSchema(t)}}}
	response, err := provider.Complete(t.Context(), request)
	if err != nil || response.Content != `{"title":"synthetic"}` {
		t.Fatalf("response=%+v error=%v", response, err)
	}
	request.ResponseFormat = &llm.ResponseFormat{Type: "json_object"}
	if _, err := provider.Complete(t.Context(), request); err == nil || !strings.Contains(err.Error(), "json_object") {
		t.Fatalf("unsupported format was silently ignored: %v", err)
	}
	request.ResponseFormat = nil
	if _, err := provider.Complete(t.Context(), request); err != nil {
		t.Fatalf("ordinary prompt request was blocked: %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("unsupported format made an HTTP request: %d", calls.Load())
	}
}

func TestAnthropicToolChoiceStreamActualHTTP(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var wire struct {
			ToolChoice map[string]any `json:"tool_choice"`
			Stream     bool           `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
			t.Error(err)
		}
		if !wire.Stream || wire.ToolChoice["type"] != "tool" || wire.ToolChoice["name"] != "emit_schema" {
			t.Errorf("stream tool choice=%+v", wire)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"id\":\"stream\",\"model\":\"actual\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":2}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call-schema\",\"name\":\"emit_schema\",\"input\":{}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"title\\\":\\\"synthetic\\\"}\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":1}}\n\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()
	stream, err := New("synthetic", WithBaseURL(server.URL)).Stream(t.Context(), llm.CompletionRequest{Model: "claude-synthetic", Tools: []llm.ToolDefinition{llm.NewToolDefinition("emit_schema", "schema", anthropicChoiceSchema(t))}, ToolChoice: map[string]any{"type": "function", "function": map[string]any{"name": "emit_schema"}}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := stream.Collect()
	_ = stream.Close()
	if err != nil || len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "emit_schema" || result.ToolCalls[0].Arguments != `{"title":"synthetic"}` {
		t.Fatalf("stream schema result=%+v error=%v", result, err)
	}
}

func TestAnthropicForcedChoiceKeepsProviderUnsupportedError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var wire map[string]any
		if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
			t.Error(err)
		}
		if wire["tool_choice"] == nil || wire["thinking"] == nil {
			t.Error("requested choice or thinking was silently dropped")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"manual thinking does not support forced tool use"}}`)
	}))
	defer server.Close()
	_, err := New("synthetic", WithBaseURL(server.URL)).Complete(t.Context(), llm.CompletionRequest{Model: "claude-synthetic", Tools: []llm.ToolDefinition{llm.NewToolDefinition("emit_schema", "schema", nil)}, ToolChoice: "required", Metadata: map[string]any{"reasoning_v1": map[string]any{"mode": "enabled", "budget_tokens": 1024}}})
	var providerError *llm.ProviderError
	if !errors.As(err, &providerError) || providerError.StatusCode != 400 {
		t.Fatalf("provider unsupported error lost: %v", err)
	}
}
