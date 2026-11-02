package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestAnthropicThinkingContinuationTwoRoundActualHTTP(t *testing.T) {
	for _, useStream := range []bool{false, true} {
		t.Run(fmt.Sprint(useStream), func(t *testing.T) {
			var httpCalls, toolCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if httpCalls.Add(1) == 1 {
					if !useStream {
						w.Header().Set("Content-Type", "application/json")
						fmt.Fprint(w, `{"id":"native-message","model":"actual","content":[{"type":"thinking","thinking":"private chain","signature":"signature-native"},{"type":"text","text":"working"},{"type":"tool_use","id":"native-call","name":"echo","input":{"title":"synthetic"}}],"stop_reason":"tool_use","usage":{"input_tokens":2,"output_tokens":2}}`)
						return
					}
					w.Header().Set("Content-Type", "text/event-stream")
					write := func(value any) { data, _ := json.Marshal(value); fmt.Fprintf(w, "data: %s\n\n", data) }
					write(map[string]any{"type": "message_start", "message": map[string]any{"id": "native-message", "model": "actual", "role": "assistant", "usage": map[string]int{"input_tokens": 2}}})
					write(map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""}})
					for _, value := range []string{"private ", "chain"} {
						write(map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "thinking_delta", "thinking": value}})
					}
					for _, value := range []string{"signature-", "native"} {
						write(map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "signature_delta", "signature": value}})
					}
					write(map[string]any{"type": "content_block_stop", "index": 0})
					write(map[string]any{"type": "content_block_start", "index": 1, "content_block": map[string]any{"type": "text", "text": ""}})
					write(map[string]any{"type": "content_block_delta", "index": 1, "delta": map[string]any{"type": "text_delta", "text": "working"}})
					write(map[string]any{"type": "content_block_stop", "index": 1})
					write(map[string]any{"type": "content_block_start", "index": 2, "content_block": map[string]any{"type": "tool_use", "id": "native-call", "name": "echo", "input": map[string]any{}}})
					for _, value := range []string{`{"title":`, `"synthetic"}`} {
						write(map[string]any{"type": "content_block_delta", "index": 2, "delta": map[string]any{"type": "input_json_delta", "partial_json": value}})
					}
					write(map[string]any{"type": "content_block_stop", "index": 2})
					write(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use"}, "usage": map[string]int{"output_tokens": 2}})
					write(map[string]any{"type": "message_stop"})
					return
				}
				var wire struct {
					Messages []struct {
						Content []map[string]any `json:"content"`
					} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				if len(wire.Messages) != 3 || len(wire.Messages[1].Content) != 3 {
					t.Errorf("source native blocks missing: %+v", wire)
					w.WriteHeader(400)
					return
				}
				thinking := wire.Messages[1].Content[0]
				if thinking["type"] != "thinking" || thinking["thinking"] != "private chain" || thinking["signature"] != "signature-native" {
					t.Errorf("native signature/thinking was rewritten: %+v", thinking)
				}
				result := wire.Messages[2].Content[0]
				if result["type"] != "tool_result" || result["tool_use_id"] != "native-call" {
					t.Errorf("native result association=%+v", result)
				}
				content, _ := json.Marshal(result["content"])
				if !strings.Contains(string(content), "synthetic") {
					t.Errorf("actual tool result missing: %s", content)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"final","model":"actual","content":[{"type":"text","text":"{\"title\":\"synthetic\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}`)
			}))
			defer server.Close()
			provider := New("synthetic", WithBaseURL(server.URL))
			request := llm.CompletionRequest{Model: "claude-synthetic", Messages: []llm.Message{{Role: llm.RoleUser, Content: "echo schema"}}, Tools: []llm.ToolDefinition{llm.NewToolDefinition("echo", "echo title", anthropicChoiceSchema(t))}, ToolChoice: "auto", Metadata: map[string]any{"reasoning_v1": map[string]any{"mode": "enabled", "budget_tokens": 1024}}}
			var calls []llm.ToolCall
			var text string
			var continuation *llm.ProviderContinuation
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
				calls, text, continuation = result.ToolCalls, result.Content, result.ProviderContinuation
			} else {
				result, err := provider.Complete(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				calls, text, continuation = result.ToolCalls, result.Content, result.ProviderContinuation
				public, _ := json.Marshal(result)
				if strings.Contains(string(public), "private chain") || strings.Contains(string(public), "signature-native") {
					t.Fatal("private continuation was exposed in public completion JSON")
				}
			}
			if len(calls) != 1 || text != "working" || continuation == nil || continuation.Protocol != ContinuationProtocol {
				t.Fatalf("first round calls=%+v text=%q continuation=%+v", calls, text, continuation)
			}
			var args struct {
				Title string `json:"title"`
			}
			if err := json.Unmarshal([]byte(calls[0].Arguments), &args); err != nil {
				t.Fatal(err)
			}
			toolCalls.Add(1)
			assistant := llm.Message{Role: llm.RoleAssistant, Content: text, ToolCalls: []llm.ToolCallRef{{ID: calls[0].ID, Name: calls[0].Name, Arguments: calls[0].Arguments}}, ProviderContinuation: continuation}
			request.Messages = append(request.Messages, assistant, llm.Message{Role: llm.RoleTool, ToolCallID: calls[0].ID, Name: calls[0].Name, Content: args.Title})
			for _, variant := range []string{"source", "protocol", "model", "endpoint", "credential"} {
				copy := request
				copy.Messages = append([]llm.Message(nil), request.Messages...)
				copy.Messages[1].ProviderContinuation = continuation.Clone()
				target := provider
				switch variant {
				case "source":
					copy.Messages[1].ProviderContinuation.Source = "foreign"
				case "protocol":
					copy.Messages[1].ProviderContinuation.Protocol = "another.protocol"
				case "model":
					copy.Model = "another-model"
				case "endpoint":
					target = New("synthetic", WithBaseURL(server.URL+"/another"))
				case "credential":
					target = New("another-key", WithBaseURL(server.URL))
				}
				body, _, err := target.buildRequestBody(copy, false)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(body), "private chain") || strings.Contains(string(body), "signature-native") || !strings.Contains(string(body), "tool_use") {
					t.Fatalf("%s source boundary did not preserve generic facts: %s", variant, body)
				}
			}
			request.ToolChoice = "none"
			request.ResponseFormat = &llm.ResponseFormat{Type: "json_schema", JSONSchema: &llm.ResponseFormatJSONSchema{Name: "result", Schema: anthropicChoiceSchema(t)}}
			final, err := provider.Complete(t.Context(), request)
			if err != nil || final.Content != `{"title":"synthetic"}` || httpCalls.Load() != 2 || toolCalls.Load() != 1 {
				t.Fatalf("final=%+v error=%v HTTP=%d tools=%d", final, err, httpCalls.Load(), toolCalls.Load())
			}
		})
	}
}
