package streamx

import "testing"

func TestOpenAIParserReturnsStreamError(t *testing.T) {
	t.Parallel()

	parser := &OpenAIParser{}
	if _, err := parser.Parse([]byte(`{"error":{"message":"quota exceeded"}}`)); err == nil {
		t.Fatal("Parse() error = nil, want stream error")
	}
}

func TestClaudeParserReturnsStreamError(t *testing.T) {
	t.Parallel()

	parser := &ClaudeParser{}
	if _, err := parser.Parse([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}`)); err == nil {
		t.Fatal("Parse() error = nil, want stream error")
	}
}

func TestClaudeParserEmitsUsage(t *testing.T) {
	t.Parallel()

	parser := &ClaudeParser{}
	start, err := parser.Parse([]byte(`{"type":"message_start","message":{"id":"m1","role":"assistant","model":"claude","usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":3}}}`))
	if err != nil {
		t.Fatalf("Parse(message_start) error = %v", err)
	}
	if start.Usage == nil || start.Usage.PromptTokens != 10 || start.Usage.CacheReadTokens != 3 {
		t.Fatalf("message_start usage = %+v", start.Usage)
	}
	delta, err := parser.Parse([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`))
	if err != nil {
		t.Fatalf("Parse(message_delta) error = %v", err)
	}
	if delta.Usage == nil || delta.Usage.PromptTokens != 10 || delta.Usage.CompletionTokens != 7 || delta.Usage.TotalTokens != 17 {
		t.Fatalf("message_delta usage = %+v", delta.Usage)
	}
}
