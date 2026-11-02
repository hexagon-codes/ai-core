package anthropic

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestAnthropicTwoRoundToolExecutionWithFailureMediaActualHTTP(t *testing.T) {
	for _, useStream := range []bool{false, true} {
		t.Run(fmt.Sprint(useStream), func(t *testing.T) {
			var imageBytes bytes.Buffer
			if err := png.Encode(&imageBytes, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
				t.Fatal(err)
			}
			imageData := base64.StdEncoding.EncodeToString(imageBytes.Bytes())
			var httpCalls, toolCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var wire struct {
					Messages []struct {
						Role    string           `json:"role"`
						Content []map[string]any `json:"content"`
					} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				if httpCalls.Add(1) == 1 {
					if useStream {
						w.Header().Set("Content-Type", "text/event-stream")
						write := func(value any) { data, _ := json.Marshal(value); fmt.Fprintf(w, "data: %s\n\n", data) }
						write(map[string]any{"type": "message_start", "message": map[string]any{"id": "message-native", "model": "actual", "role": "assistant", "usage": map[string]int{"input_tokens": 2}}})
						for index, call := range []struct{ id, name, args string }{{"tool-add", "add", `{"a":2,"b":3}`}, {"tool-failed", "unavailable", `{}`}} {
							write(map[string]any{"type": "content_block_start", "index": index, "content_block": map[string]any{"type": "tool_use", "id": call.id, "name": call.name, "input": map[string]any{}}})
							write(map[string]any{"type": "content_block_delta", "index": index, "delta": map[string]any{"type": "input_json_delta", "partial_json": call.args}})
							write(map[string]any{"type": "content_block_stop", "index": index})
						}
						write(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use"}, "usage": map[string]int{"output_tokens": 2}})
						write(map[string]any{"type": "message_stop"})
					} else {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, `{"id":"message-native","model":"actual","content":[{"type":"text","text":"working"},{"type":"tool_use","id":"tool-add","name":"add","input":{"a":2,"b":3}},{"type":"tool_use","id":"tool-failed","name":"unavailable","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":2,"output_tokens":2}}`)
					}
					return
				}
				if len(wire.Messages) != 3 || wire.Messages[1].Role != "assistant" || wire.Messages[2].Role != "user" {
					t.Errorf("native tool turn was not grouped: %+v", wire.Messages)
					w.WriteHeader(400)
					return
				}
				seen := map[string]string{}
				for _, block := range wire.Messages[1].Content {
					if block["type"] == "tool_use" {
						seen[block["id"].(string)] = block["name"].(string)
					}
				}
				if seen["tool-add"] != "add" || seen["tool-failed"] != "unavailable" {
					t.Errorf("assistant tool_use lost: %+v", wire.Messages[1])
				}
				results := wire.Messages[2].Content
				if len(results) != 3 || results[0]["type"] != "tool_result" || results[0]["tool_use_id"] != "tool-add" || results[1]["type"] != "tool_result" || results[1]["tool_use_id"] != "tool-failed" || results[2]["type"] != "text" {
					t.Errorf("native tool_result order/IDs lost: %+v", results)
					w.WriteHeader(400)
					return
				}
				failedJSON, _ := json.Marshal(results[1]["content"])
				if results[1]["is_error"] != true {
					t.Errorf("explicit tool failure flag lost: %+v", results[1])
				}
				if !strings.Contains(string(failedJSON), "synthetic execution failure") || !strings.Contains(string(failedJSON), imageData) || !strings.Contains(string(failedJSON), "image/png") {
					t.Errorf("failed tool text or media lost: %s", failedJSON)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"final-native","model":"actual","content":[{"type":"text","text":"{\"total\":5,\"failed\":true}"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":1}}`)
			}))
			defer server.Close()
			var arithmetic llm.Schema
			if err := json.Unmarshal([]byte(`{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}`), &arithmetic); err != nil {
				t.Fatal(err)
			}
			provider := New("synthetic", WithBaseURL(server.URL))
			request := llm.CompletionRequest{Model: "claude-synthetic", Messages: []llm.Message{{Role: llm.RoleUser, Content: "run both tools"}}, Tools: []llm.ToolDefinition{llm.NewToolDefinition("add", "add numbers", &arithmetic), llm.NewToolDefinition("unavailable", "synthetic failure", nil)}, ToolChoice: "required"}
			var calls []llm.ToolCall
			var text string
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
				calls, text = result.ToolCalls, result.Content
			} else {
				result, err := provider.Complete(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				calls, text = result.ToolCalls, result.Content
			}
			if len(calls) != 2 {
				t.Fatalf("first round executable calls=%+v", calls)
			}
			assistant := llm.Message{Role: llm.RoleAssistant, Content: text}
			var toolResults []llm.Message
			for _, call := range calls {
				assistant.ToolCalls = append(assistant.ToolCalls, llm.ToolCallRef{ID: call.ID, Name: call.Name, Arguments: call.Arguments})
				toolCalls.Add(1)
				switch call.Name {
				case "add":
					var args struct{ A, B int }
					if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
						t.Fatal(err)
					}
					toolResults = append(toolResults, llm.Message{Role: llm.RoleTool, ToolCallID: call.ID, Name: call.Name, Content: fmt.Sprintf(`{"value":%d}`, args.A+args.B)})
				case "unavailable":
					failure := errors.New("synthetic execution failure")
					toolResults = append(toolResults, llm.Message{Role: llm.RoleTool, ToolCallID: call.ID, Name: call.Name, Content: failure.Error(), ToolResultError: true, MultiContent: []llm.ContentPart{{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64," + imageData}}}})
				default:
					t.Fatalf("unexpected tool %q", call.Name)
				}
			}
			request.Messages = append(request.Messages, assistant)
			request.Messages = append(request.Messages, toolResults...)
			request.Messages = append(request.Messages, llm.Message{Role: llm.RoleUser, Content: "return the structured result"})
			request.ToolChoice = "none"
			var outputSchema llm.Schema
			if err := json.Unmarshal([]byte(`{"type":"object","properties":{"total":{"type":"integer"},"failed":{"type":"boolean"}},"required":["total","failed"],"additionalProperties":false}`), &outputSchema); err != nil {
				t.Fatal(err)
			}
			request.ResponseFormat = &llm.ResponseFormat{Type: "json_schema", JSONSchema: &llm.ResponseFormatJSONSchema{Name: "result", Schema: &outputSchema}}
			final, err := provider.Complete(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			var result struct {
				Total  int  `json:"total"`
				Failed bool `json:"failed"`
			}
			if json.Unmarshal([]byte(final.Content), &result) != nil || result.Total != 5 || !result.Failed || httpCalls.Load() != 2 || toolCalls.Load() != 2 {
				t.Fatalf("final=%+v HTTP=%d tool=%d", final, httpCalls.Load(), toolCalls.Load())
			}
		})
	}
}

func TestAnthropicEmptySuccessfulToolResultHasValidNativeBlock(t *testing.T) {
	body, _, err := New("synthetic").buildRequestBody(llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCallRef{{ID: "empty-call", Name: "no_output", Arguments: "{}"}}}, {Role: llm.RoleTool, ToolCallID: "empty-call"}}}, false)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	block := wire.Messages[1].Content[0]
	if block["type"] != "tool_result" || block["tool_use_id"] != "empty-call" || block["content"] != nil || block["is_error"] != nil {
		t.Fatalf("empty successful tool result was made invalid or failed: %+v", block)
	}
}
