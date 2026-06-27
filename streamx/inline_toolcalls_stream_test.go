package streamx

import (
	"encoding/json"
	"strings"
	"testing"
)

// 流式路径：当工具调用以 ｜DSML｜ 标记泄漏进 content 的增量里、tool_calls delta 始终为空时，
// 流收尾（finalize）必须把内嵌标记还原成结构化 ToolCalls 并剥净正文——否则桌面流式聊天里
// 用户"会话建任务"同样失败（与同步路径同源 bug）。标记跨多个 content delta 累积。
func TestStream_RecoversInlineToolCalls_OpenAI(t *testing.T) {
	prose := "马上安排 🦀\n"
	markup := "<｜DSML｜tool_calls>\n" +
		"<｜DSML｜invoke name=\"cron_task\">\n" +
		"<｜DSML｜parameter name=\"action\" string=\"true\">create</｜DSML｜parameter>\n" +
		"<｜DSML｜parameter name=\"schedule\" string=\"true\">0 9 * * *</｜DSML｜parameter>\n" +
		"</｜DSML｜invoke>\n" +
		"</｜DSML｜tool_calls>"

	mkLine := func(content string) string {
		b, _ := json.Marshal(map[string]any{
			"choices": []map[string]any{{"delta": map[string]any{"content": content}}},
		})
		return "data: " + string(b)
	}
	// 把泄漏内容拆成两个 content delta，验证跨块累积后再还原。
	input := mkLine(prose) + "\n" + mkLine(markup) + "\ndata: [DONE]\n"

	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("collect: %v", err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 recovered tool call from streamed content, got %d (streaming path dropped it)", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "cron_task" {
		t.Errorf("recovered tool name = %q, want cron_task", result.ToolCalls[0].Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(result.ToolCalls[0].Arguments), &args); err != nil {
		t.Fatalf("recovered args not valid JSON: %v", err)
	}
	if args["action"] != "create" || args["schedule"] != "0 9 * * *" {
		t.Errorf("recovered args wrong: %v", args)
	}
	if result.FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", result.FinishReason)
	}
	if strings.Contains(result.Content, "DSML") || strings.Contains(result.Content, "tool_calls") {
		t.Errorf("streamed content still leaks markup: %q", result.Content)
	}
	if result.Content != "马上安排 🦀" {
		t.Errorf("cleaned streamed content = %q, want %q", result.Content, "马上安排 🦀")
	}
}

// 流里本就带结构化 tool_calls 时，finalize 不得触发兜底（结构化优先）。
func TestStream_StructuredToolCalls_NotRecovered(t *testing.T) {
	input := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_real\",\"type\":\"function\",\"function\":{\"name\":\"real_tool\",\"arguments\":\"{}\"}}]}}]}\n" +
		"data: [DONE]\n"
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("collect: %v", err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Name != "real_tool" || result.ToolCalls[0].ID != "call_real" {
		t.Fatalf("structured streamed tool call must be preserved, got %+v", result.ToolCalls)
	}
}
