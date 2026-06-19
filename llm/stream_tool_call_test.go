package llm

import (
	"context"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/streamx"
)

func textChunk(s string) *streamx.Chunk { return &streamx.Chunk{Content: s} }

func toolChunk() *streamx.Chunk {
	return &streamx.Chunk{ToolCalls: []streamx.ToolCall{{Index: 0, ID: "t1", Name: "f", Type: "function"}}}
}

// TestChecker_NoToolCall 纯文本流 → 无工具调用。
func TestChecker_NoToolCall(t *testing.T) {
	sr := streamx.FromSlice([]*streamx.Chunk{textChunk("a"), textChunk("b")})
	has, err := DefaultStreamToolCallChecker(context.Background(), sr)
	if err != nil || has {
		t.Errorf("期望 (false,nil), got (%v,%v)", has, err)
	}
}

// TestChecker_ToolCallFirst 首个 chunk 即工具调用 → true。
func TestChecker_ToolCallFirst(t *testing.T) {
	sr := streamx.FromSlice([]*streamx.Chunk{toolChunk(), textChunk("x")})
	has, err := DefaultStreamToolCallChecker(context.Background(), sr)
	if err != nil || !has {
		t.Errorf("期望 (true,nil), got (%v,%v)", has, err)
	}
}

// TestChecker_TextThenToolCall 是 Claude-aware 的关键用例：
// 模型先输出文本再进入工具调用，Checker 不应被首个文本 chunk 误判为「无工具调用」。
func TestChecker_TextThenToolCall(t *testing.T) {
	sr := streamx.FromSlice([]*streamx.Chunk{textChunk("让我想想"), textChunk("…"), toolChunk()})
	has, err := DefaultStreamToolCallChecker(context.Background(), sr)
	if err != nil || !has {
		t.Errorf("text-then-toolcall 应判定为 true, got (%v,%v)", has, err)
	}
}

// TestChecker_CtxCancelled 已取消的 context 应返回错误而非阻塞。
func TestChecker_CtxCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	sr := streamx.FromSlice([]*streamx.Chunk{textChunk("a")})
	if _, err := DefaultStreamToolCallChecker(ctx, sr); err == nil {
		t.Error("已取消的 ctx 应返回错误")
	}
}

// TestChecker_E2E_ClaudeStreamWithToolUse 端到端：真实 Claude SSE（文本在前、
// tool_use 在后）经 Stream.Reader() 桥接后被 Checker 正确检出工具调用。
func TestChecker_E2E_ClaudeStreamWithToolUse(t *testing.T) {
	sse := `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me check"}}
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}
data: {"type":"message_stop"}
`
	sr := streamx.NewStream(strings.NewReader(sse), streamx.ClaudeFormat).Reader()
	has, err := DefaultStreamToolCallChecker(context.Background(), sr)
	if err != nil || !has {
		t.Errorf("e2e: Claude 流式 tool_use 应被检出, got (%v,%v)", has, err)
	}
}
