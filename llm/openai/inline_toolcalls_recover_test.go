package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

// 端到端（真实 HTTP）：mock OpenAI 兼容网关返回一条 tool_calls 缺失、工具调用泄漏进 content
// 的响应，Complete 走完整条链（HTTP → json 解析 → parseResponse 兜底还原）后必须产出结构化
// 工具调用。覆盖 antml(｜DSML｜) 与 Hermes(<tool_call>) 两种真机方言。
func TestComplete_E2E_RecoversLeakedToolCalls(t *testing.T) {
	cases := []struct {
		name    string
		content string
	}{
		{
			name: "antml_DSML",
			content: "钳住了 🦀\n<｜DSML｜invoke name=\"cron_task\">" +
				"<｜DSML｜parameter name=\"action\" string=\"true\">create</｜DSML｜parameter>" +
				"<｜DSML｜parameter name=\"schedule\" string=\"true\">0 9 * * *</｜DSML｜parameter>" +
				"</｜DSML｜invoke>",
		},
		{
			name:    "hermes_tool_call",
			content: "好的 🦀\n<tool_call>{\"tool\":\"cron_task\",\"arguments\":{\"action\":\"create\",\"schedule\":\"0 9 * * *\"}}</tool_call>",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body, _ := json.Marshal(map[string]any{
				"id":    "chatcmpl-x",
				"model": "deepseek",
				"choices": []map[string]any{{
					"index":         0,
					"message":       map[string]any{"role": "assistant", "content": tc.content},
					"finish_reason": "stop",
				}},
			})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(body)
			}))
			defer srv.Close()

			p := New("k", WithBaseURL(srv.URL), WithModel("deepseek"))
			resp, err := p.Complete(context.Background(), llm.CompletionRequest{
				Model:    "deepseek",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "每天9点采集百度热搜写入知识库，直接建"}},
			})
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "cron_task" {
				t.Fatalf("expected recovered cron_task call, got %+v", resp.ToolCalls)
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(resp.ToolCalls[0].Arguments), &args); err != nil {
				t.Fatalf("args not valid JSON: %v", err)
			}
			if args["action"] != "create" || args["schedule"] != "0 9 * * *" {
				t.Errorf("recovered args wrong: %v", args)
			}
			if resp.FinishReason != "tool_calls" {
				t.Errorf("finish_reason = %q, want tool_calls", resp.FinishReason)
			}
		})
	}
}

// 同步 Complete 路径：DeepSeek 经 OpenAI 兼容网关把工具调用以 ｜DSML｜ 标记内嵌进 content、
// tool_calls 字段为空时，parseResponse 必须兜底还原结构化工具调用并剥净正文，否则会话→建
// 任务在引擎里被当普通文本丢弃（2026-06-27 真机 bug）。｜ 为全宽竖线 U+FF5C。
func TestParseResponse_RecoversInlineDeepSeekToolCalls(t *testing.T) {
	p := New("test")

	content := "钳住了，马上给你安排 🦀\n" +
		"<｜DSML｜tool_calls>\n" +
		"<｜DSML｜invoke name=\"cron_task\">\n" +
		"<｜DSML｜parameter name=\"action\" string=\"true\">create</｜DSML｜parameter>\n" +
		"<｜DSML｜parameter name=\"name\" string=\"true\">百度热搜每日采集</｜DSML｜parameter>\n" +
		"<｜DSML｜parameter name=\"schedule\" string=\"true\">0 9 * * *</｜DSML｜parameter>\n" +
		"</｜DSML｜invoke>\n" +
		"</｜DSML｜tool_calls>"

	// 用 JSON 还原一条真实的 OpenAI 响应（content 带泄漏标记、tool_calls 缺失）。
	raw, _ := json.Marshal(map[string]any{
		"id":    "resp_1",
		"model": "deepseek-ai/DeepSeek-V4-Pro",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
	})
	var resp openAIResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	got := p.parseResponse(&resp)

	if len(got.ToolCalls) != 1 {
		t.Fatalf("expected 1 recovered tool call, got %d (leaked markup was dropped)", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Name != "cron_task" {
		t.Errorf("recovered tool name = %q, want cron_task", got.ToolCalls[0].Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(got.ToolCalls[0].Arguments), &args); err != nil {
		t.Fatalf("recovered arguments not valid JSON: %v", err)
	}
	if args["action"] != "create" || args["schedule"] != "0 9 * * *" {
		t.Errorf("recovered args wrong: %v", args)
	}
	// finish_reason 从 stop 升级为 tool_calls（语义对齐：模型其实发起了工具调用）。
	if got.FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", got.FinishReason)
	}
	// 正文被剥净，泄漏标记不再外泄给用户。
	if strings.Contains(got.Content, "DSML") || strings.Contains(got.Content, "tool_calls") {
		t.Errorf("content still leaks markup: %q", got.Content)
	}
	if got.Content != "钳住了，马上给你安排 🦀" {
		t.Errorf("cleaned content = %q, want %q", got.Content, "钳住了，马上给你安排 🦀")
	}
}

// 结构化 tool_calls 已存在时，绝不触发兜底（避免误改/重复），即使 content 里恰好提到标记。
func TestParseResponse_StructuredToolCallsNotClobbered(t *testing.T) {
	p := New("test")
	raw, _ := json.Marshal(map[string]any{
		"id": "resp_2",
		"choices": []map[string]any{{
			"index": 0,
			"message": map[string]any{
				"role":    "assistant",
				"content": "顺带提一句 invoke name=\"x\" 只是说明",
				"tool_calls": []map[string]any{{
					"id": "call_real", "type": "function",
					"function": map[string]any{"name": "real_tool", "arguments": "{}"},
				}},
			},
			"finish_reason": "tool_calls",
		}},
	})
	var resp openAIResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got := p.parseResponse(&resp)
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].Name != "real_tool" || got.ToolCalls[0].ID != "call_real" {
		t.Fatalf("structured tool call must be preserved untouched, got %+v", got.ToolCalls)
	}
	if !strings.Contains(got.Content, "invoke name") {
		t.Errorf("content must be left intact when structured calls exist: %q", got.Content)
	}
}
