package streamx

import (
	"strings"
	"testing"
)

// 本文件固化 processLoop 内联 SSE 行解析与各 Provider parser.IsDone 协作的
// 边界语义。这些语义与 toolkit/net/sse 的 Reader.Read / IsOpenAIDone 存在实质差异
// （事件缓冲粒度、空白处理、event:/id: 行处理、解析-后-判结束的顺序、各厂商
// 专属 IsDone 语义），因此 streamx 保留本地实现而非下沉复用。若未来有人尝试用
// sse.Reader 替换本路径，下列断言会失败，提示其先确认边界等价。

// TestSSEBoundary_GeminiParseBeforeDone 固化 Gemini 末块语义：
// 末个 chunk 同时带 content 与 finishReason（IsDone 为真）。processLoop 必须
// 先解析并下发该 chunk，再判结束，否则末段内容会被丢弃。
// toolkit 的 IsOpenAIDone 只识别 "[DONE]" 文本，无法表达 finishReason 结束语义。
func TestSSEBoundary_GeminiParseBeforeDone(t *testing.T) {
	input := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n" +
		"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" World\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n"

	result, err := NewStream(strings.NewReader(input), GeminiFormat).Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}
	// 末块（含 finishReason）的内容不能被丢弃
	if result.Content != "Hello World" {
		t.Errorf("expected 'Hello World', got %q", result.Content)
	}
	if result.FinishReason != "STOP" {
		t.Errorf("expected finish_reason 'STOP', got %q", result.FinishReason)
	}
}

// TestSSEBoundary_ClaudeEventLinesIgnored 固化 Claude 语义：SSE 的 event: 行被
// 忽略，结束判定依据 data: JSON 内的 type 字段（message_stop）。
// toolkit 的 Reader.Read 会把 event: 行累积进 Event 并可能产出空 Data 事件，
// 与此处「仅解析 data: JSON」的行为不同。
func TestSSEBoundary_ClaudeEventLinesIgnored(t *testing.T) {
	input := "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"model\":\"claude-3\"}}\n\n" +
		"event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n" +
		"event: message_stop\n" +
		"data: {\"type\":\"message_stop\"}\n\n"

	result, err := NewStream(strings.NewReader(input), ClaudeFormat).Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}
	if result.ID != "msg_1" {
		t.Errorf("expected ID 'msg_1', got %q", result.ID)
	}
	if result.Content != "Hi" {
		t.Errorf("expected content 'Hi', got %q", result.Content)
	}
	if result.FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got %q", result.FinishReason)
	}
}

// TestSSEBoundary_DoneWhitespaceTolerant 固化 OpenAI [DONE] 的空白容忍：
// processLoop 对整行与 data 载荷各做一次 TrimSpace，"data:   [DONE]  " 仍判结束。
func TestSSEBoundary_DoneWhitespaceTolerant(t *testing.T) {
	input := "data: {\"choices\":[{\"delta\":{\"content\":\"X\"}}]}\n\n" +
		"data:   [DONE]  \n\n"

	result, err := NewStream(strings.NewReader(input), OpenAIFormat).Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}
	if result.Content != "X" {
		t.Errorf("expected content 'X', got %q", result.Content)
	}
}
