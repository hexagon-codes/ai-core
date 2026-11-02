package gemini

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func geminiChoiceSchema(t *testing.T) *llm.Schema {
	t.Helper()
	var result llm.Schema
	if err := json.Unmarshal([]byte(`{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}`), &result); err != nil {
		t.Fatal(err)
	}
	return &result
}

func TestGeminiToolChoiceTwoRoundExecutionActualHTTP(t *testing.T) {
	for _, useStream := range []bool{false, true} {
		t.Run(fmt.Sprint(useStream), func(t *testing.T) {
			var httpCalls, toolCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var wire struct {
					Contents []struct {
						Role  string `json:"role"`
						Parts []struct {
							FunctionCall *struct {
								ID   string         `json:"id"`
								Name string         `json:"name"`
								Args map[string]any `json:"args"`
							} `json:"functionCall"`
							FunctionResponse *struct {
								ID       string         `json:"id"`
								Name     string         `json:"name"`
								Response map[string]any `json:"response"`
							} `json:"functionResponse"`
						} `json:"parts"`
					} `json:"contents"`
				}
				if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				if httpCalls.Add(1) == 1 {
					response := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"provider-call-42","name":"add","args":{"a":2,"b":3}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}`
					if useStream {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, "data: "+response+"\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, response)
					}
					return
				}
				if len(wire.Contents) != 3 || len(wire.Contents[1].Parts) != 1 || len(wire.Contents[2].Parts) != 1 {
					t.Errorf("round-trip contents=%+v", wire.Contents)
					w.WriteHeader(400)
					return
				}
				call, result := wire.Contents[1].Parts[0].FunctionCall, wire.Contents[2].Parts[0].FunctionResponse
				if wire.Contents[1].Role != "model" || call == nil || call.ID != "provider-call-42" || call.Name != "add" || call.Args["a"] != float64(2) {
					t.Errorf("assistant function call lost: %+v", call)
				}
				if wire.Contents[2].Role != "user" || result == nil || result.ID != "provider-call-42" || result.Name != "add" || result.Response["content"] != `{"value":5}` {
					t.Errorf("function response lost identity/result: %+v", result)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"{\"total\":5}"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":1,"totalTokenCount":5}}`)
			}))
			defer server.Close()
			provider := New("synthetic", WithBaseURL(server.URL))
			var arithmeticSchema llm.Schema
			if err := json.Unmarshal([]byte(`{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}`), &arithmeticSchema); err != nil {
				t.Fatal(err)
			}
			request := llm.CompletionRequest{Model: "gemini-synthetic", Messages: []llm.Message{{Role: llm.RoleUser, Content: "add two and three"}}, Tools: []llm.ToolDefinition{llm.NewToolDefinition("add", "add numbers", &arithmeticSchema)}, ToolChoice: map[string]any{"type": "function", "function": map[string]any{"name": "add"}}}
			var calls []llm.ToolCall
			if useStream {
				stream, err := provider.Stream(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				result, err := stream.Collect()
				_ = stream.Close()
				if err != nil {
					t.Fatal(err)
				}
				calls = result.ToolCalls
			} else {
				result, err := provider.Complete(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				calls = result.ToolCalls
			}
			if len(calls) != 1 || calls[0].ID != "provider-call-42" || calls[0].Name != "add" {
				t.Fatalf("first round has no executable call: %+v", calls)
			}
			var args struct{ A, B int }
			if err := json.Unmarshal([]byte(calls[0].Arguments), &args); err != nil {
				t.Fatal(err)
			}
			toolCalls.Add(1)
			value := args.A + args.B
			request.Messages = append(request.Messages, llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCallRef{{ID: calls[0].ID, Name: calls[0].Name, Arguments: calls[0].Arguments}}}, llm.Message{Role: llm.RoleTool, ToolCallID: calls[0].ID, Name: "stale-name", Content: fmt.Sprintf(`{"value":%d}`, value)})
			request.ToolChoice = "none"
			final, err := provider.Complete(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			var output struct {
				Total int `json:"total"`
			}
			if json.Unmarshal([]byte(final.Content), &output) != nil || output.Total != 5 || httpCalls.Load() != 2 || toolCalls.Load() != 1 {
				t.Fatalf("final=%+v HTTP=%d tool=%d", final, httpCalls.Load(), toolCalls.Load())
			}
		})
	}
}

func TestGeminiLegacyDirectNameFallbackRequiresKnownTool(t *testing.T) {
	request := llm.CompletionRequest{Tools: []llm.ToolDefinition{llm.NewToolDefinition("known_function", "legacy", nil)}, Messages: []llm.Message{{Role: llm.RoleTool, ToolCallID: "known_function", Content: "result"}}}
	body, err := New("synthetic").buildRequestBody(request)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Contents []struct {
			Parts []struct {
				Response struct {
					Name string `json:"name"`
					ID   string `json:"id"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if json.Unmarshal(body, &wire) != nil || wire.Contents[0].Parts[0].Response.Name != "known_function" || wire.Contents[0].Parts[0].Response.ID != "" {
		t.Fatalf("legacy direct name mapping=%s", body)
	}
	request.Messages[0].ToolCallID = "random-call-identifier"
	if _, err := New("synthetic").buildRequestBody(request); err == nil {
		t.Fatal("unresolved random call ID was used as a function name")
	}
}

func TestGeminiExplicitToolErrorProjection(t *testing.T) {
	for _, failed := range []bool{false, true} {
		t.Run(fmt.Sprint(failed), func(t *testing.T) {
			request := llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCallRef{{ID: "provider-id", Name: "actual-name", Arguments: "{}"}}}, {Role: llm.RoleTool, ToolCallID: "provider-id", Name: "stale-name", Content: "failure-like text", ToolResultError: failed}}}
			body, err := New("synthetic").buildRequestBody(request)
			if err != nil {
				t.Fatal(err)
			}
			var wire struct {
				Contents []struct {
					Parts []struct {
						Response struct {
							Name  string         `json:"name"`
							ID    string         `json:"id"`
							Value map[string]any `json:"response"`
						} `json:"functionResponse"`
					} `json:"parts"`
				} `json:"contents"`
			}
			if err := json.Unmarshal(body, &wire); err != nil {
				t.Fatal(err)
			}
			response := wire.Contents[1].Parts[0].Response
			if response.Name != "actual-name" || response.ID != "provider-id" || response.Value["content"] != "failure-like text" || (response.Value["error"] != nil) != failed {
				t.Fatalf("explicit error projection=%+v", response)
			}
		})
	}
}

func TestGeminiToolChoiceActualHTTP(t *testing.T) {
	for _, test := range []struct {
		name               string
		choice             any
		wantMode, wantName string
	}{
		{name: "auto", choice: "auto", wantMode: "AUTO"},
		{name: "none", choice: "none", wantMode: "NONE"},
		{name: "required", choice: "required", wantMode: "ANY"},
		{name: "named", choice: map[string]any{"type": "function", "function": map[string]any{"name": "emit_schema"}}, wantMode: "ANY", wantName: "emit_schema"},
		{name: "named string", choice: "emit_schema", wantMode: "ANY", wantName: "emit_schema"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var wire struct {
					ToolConfig struct {
						FunctionCallingConfig struct {
							Mode    string   `json:"mode"`
							Allowed []string `json:"allowedFunctionNames"`
						} `json:"functionCallingConfig"`
					} `json:"toolConfig"`
					Tools []struct {
						Functions []struct {
							Name       string         `json:"name"`
							Parameters map[string]any `json:"parameters"`
						} `json:"functionDeclarations"`
					} `json:"tools"`
					Generation struct {
						Max         int      `json:"maxOutputTokens"`
						Temperature float64  `json:"temperature"`
						TopP        float64  `json:"topP"`
						Stop        []string `json:"stopSequences"`
					} `json:"generationConfig"`
					User   any `json:"user"`
					Labels any `json:"labels"`
				}
				if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				choice := wire.ToolConfig.FunctionCallingConfig
				if choice.Mode != test.wantMode || (test.wantName == "" && len(choice.Allowed) != 0) || (test.wantName != "" && (len(choice.Allowed) != 1 || choice.Allowed[0] != test.wantName)) {
					t.Errorf("tool choice wire=%+v", choice)
				}
				if len(wire.Tools) != 1 || len(wire.Tools[0].Functions) != 1 || wire.Tools[0].Functions[0].Name != "emit_schema" || wire.Tools[0].Functions[0].Parameters["type"] != "object" {
					t.Errorf("tool schema wire=%+v", wire.Tools)
				}
				if wire.Generation.Max != 256 || wire.Generation.Temperature != 0.2 || wire.Generation.TopP != 0.8 || len(wire.Generation.Stop) != 1 || wire.Generation.Stop[0] != "END" {
					t.Errorf("generation parameters lost: %+v", wire.Generation)
				}
				if wire.User != nil || wire.Labels != nil {
					t.Errorf("host user tracking was disguised as native API parameter: user=%v labels=%v", wire.User, wire.Labels)
				}
				w.Header().Set("Content-Type", "application/json")
				if test.wantMode == "ANY" {
					fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"emit_schema","args":{"title":"synthetic"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}`)
				} else {
					fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"{\"title\":\"synthetic\"}"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}`)
				}
			}))
			defer server.Close()
			temperature, topP := 0.2, 0.8
			request := llm.CompletionRequest{Model: "gemini-synthetic", Messages: []llm.Message{{Role: llm.RoleUser, Content: "return schema result"}}, Tools: []llm.ToolDefinition{llm.NewToolDefinition("emit_schema", "emit schema", geminiChoiceSchema(t))}, ToolChoice: test.choice, MaxTokens: 256, Temperature: &temperature, TopP: &topP, Stop: []string{"END"}, User: "synthetic-user"}
			response, err := New("synthetic", WithBaseURL(server.URL)).Complete(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			data := response.Content
			if test.wantMode == "ANY" {
				if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "emit_schema" {
					t.Fatalf("tool result missing: %+v", response)
				}
				data = response.ToolCalls[0].Arguments
			}
			var result struct {
				Title string `json:"title"`
			}
			if json.Unmarshal([]byte(data), &result) != nil || result.Title != "synthetic" || request.User != "synthetic-user" {
				t.Fatalf("schema result=%q user=%q", data, request.User)
			}
		})
	}
}

func TestGeminiToolChoiceStreamActualHTTP(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var wire struct {
			ToolConfig struct {
				FunctionCallingConfig struct {
					Mode    string   `json:"mode"`
					Allowed []string `json:"allowedFunctionNames"`
				} `json:"functionCallingConfig"`
			} `json:"toolConfig"`
		}
		if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
			t.Error(err)
		}
		choice := wire.ToolConfig.FunctionCallingConfig
		if choice.Mode != "ANY" || len(choice.Allowed) != 1 || choice.Allowed[0] != "emit_schema" {
			t.Errorf("stream tool choice=%+v", choice)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"emit_schema\",\"args\":{\"title\":\"synthetic\"}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":1,\"totalTokenCount\":3}}\n\n")
	}))
	defer server.Close()
	stream, err := New("synthetic", WithBaseURL(server.URL)).Stream(t.Context(), llm.CompletionRequest{Model: "gemini-synthetic", Tools: []llm.ToolDefinition{llm.NewToolDefinition("emit_schema", "schema", geminiChoiceSchema(t))}, ToolChoice: map[string]any{"type": "function", "function": map[string]any{"name": "emit_schema"}}})
	if err != nil {
		t.Fatal(err)
	}
	result, err := stream.Collect()
	_ = stream.Close()
	if err != nil || len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "emit_schema" || result.ToolCalls[0].Arguments != `{"title":"synthetic"}` || result.UsageReceipt == nil || result.UsageReceipt.TotalTokens != 3 {
		t.Fatalf("stream schema result=%+v error=%v", result, err)
	}
}
