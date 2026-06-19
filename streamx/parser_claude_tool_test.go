package streamx

import "testing"

// TestClaudeParser_ToolUseStart 验证 content_block_start{tool_use} 被解析为
// 携带 id/name/index 的 ToolCall（此前被完全忽略，流式工具调用不可见）。
func TestClaudeParser_ToolUseStart(t *testing.T) {
	p := &ClaudeParser{}
	data := []byte(`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"get_weather","input":{}}}`)

	chunk, err := p.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunk.ToolCalls) != 1 {
		t.Fatalf("期望 1 个 ToolCall, got %d", len(chunk.ToolCalls))
	}
	tc := chunk.ToolCalls[0]
	if tc.ID != "toolu_abc" || tc.Name != "get_weather" || tc.Index != 1 || tc.Type != "function" {
		t.Errorf("ToolCall 解析错误: %+v", tc)
	}
}

// TestClaudeParser_InputJSONDelta 验证 content_block_delta{input_json_delta}
// 增量参数被解析为 ToolCall.Arguments（以 index 为合并键）。
func TestClaudeParser_InputJSONDelta(t *testing.T) {
	p := &ClaudeParser{}
	data := []byte(`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`)

	chunk, err := p.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunk.ToolCalls) != 1 {
		t.Fatalf("期望 1 个 ToolCall, got %d", len(chunk.ToolCalls))
	}
	tc := chunk.ToolCalls[0]
	if tc.Index != 1 || tc.Arguments != `{"city":` {
		t.Errorf("input_json_delta 解析错误: %+v", tc)
	}
	if chunk.Content != "" {
		t.Errorf("工具参数增量不应填入 Content, got %q", chunk.Content)
	}
}

// TestClaudeParser_TextDeltaUnchanged 验证文本增量路径保持不变（回归）。
func TestClaudeParser_TextDeltaUnchanged(t *testing.T) {
	p := &ClaudeParser{}
	data := []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}`)

	chunk, err := p.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if chunk.Content != "你好" {
		t.Errorf("文本增量应填入 Content, got %q", chunk.Content)
	}
	if len(chunk.ToolCalls) != 0 {
		t.Errorf("文本增量不应产生 ToolCalls, got %v", chunk.ToolCalls)
	}
}
