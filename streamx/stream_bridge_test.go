package streamx

import (
	"io"
	"strings"
	"testing"
)

// claudeSSEWithToolUse 是一段含「文本 → tool_use → 参数增量 → 结束」的 Claude SSE。
const claudeSSEWithToolUse = `data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","model":"claude-x"}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check"}}
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}}
data: {"type":"message_stop"}
`

// TestStreamReaderBridge_YieldsToolCall 验证 Stream.Reader() 桥接后，
// 经 StreamReader.Recv 能拿到文本与 tool_use chunk（端到端：SSE→ClaudeParser→桥）。
func TestStreamReaderBridge_YieldsToolCall(t *testing.T) {
	sr := NewStream(strings.NewReader(claudeSSEWithToolUse), ClaudeFormat).Reader()

	var sawTool bool
	var text string
	for {
		c, err := sr.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		text += c.Content
		if len(c.ToolCalls) > 0 && c.ToolCalls[0].Name == "get_weather" {
			sawTool = true
		}
	}
	if !sawTool {
		t.Error("桥接后应能看到 tool_use chunk")
	}
	if text != "Let me check" {
		t.Errorf("文本聚合错误: %q", text)
	}
}

// TestStreamReaderBridge_Copy 验证桥接所得 reader 可经 Copy 扇出，每路都看到工具调用。
func TestStreamReaderBridge_Copy(t *testing.T) {
	sr := NewStream(strings.NewReader(claudeSSEWithToolUse), ClaudeFormat).Reader()
	copies := sr.Copy(2)
	if len(copies) != 2 {
		t.Fatalf("Copy(2) 应返回 2 路, got %d", len(copies))
	}
	for i, r := range copies {
		has := false
		for {
			c, err := r.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(c.ToolCalls) > 0 {
				has = true
			}
		}
		if !has {
			t.Errorf("Copy 第 %d 路应看到工具调用", i)
		}
	}
}
