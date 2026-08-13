// Package openai 回归测试：BUG-20260520
//
// 症状：用户多轮对话触发"runtime stream 失败: llm complete: openai api error: 400 Bad Request,
//
//	body: ... Invalid JSON: Input should be a valid string or an array of content blocks ..."
//
// 根因：convertMessages 序列化历史消息时漏写两个字段——
//
//  1. assistant 消息上的 msg.ToolCalls   → 上一轮工具调用历史丢失
//
//  2. role=tool  消息上的 msg.ToolCallID → 工具结果失去与调用的关联
//
//     下游 OpenAI 兼容网关（new-api / one-api 等）把这种残缺的请求翻译成
//     Anthropic / Claude 上游格式时，content 字段被翻成 null → 400。
//
// 修复点：llm/openai/openai.go::convertMessages
//
// 本文件永久保留作回归屏障——任何后续重构若回退到"漏写 tool_calls/tool_call_id"
// 的状态，本测试立即 FAIL。
package openai

import (
	"encoding/json"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

// TestBUG20260520_AssistantMessageWithToolCallsMustSerializeToolCalls
// 复现 RED：assistant 消息含 ToolCalls 时，序列化结果必须包含 tool_calls 字段。
func TestBUG20260520_AssistantMessageWithToolCallsMustSerializeToolCalls(t *testing.T) {
	msgs := []llm.Message{
		{
			Role:    llm.RoleAssistant,
			Content: "",
			ToolCalls: []llm.ToolCallRef{
				{ID: "call_abc123", Name: "write_file", Arguments: `{"path":"out.md","content":"hello"}`},
			},
		},
	}

	out := convertMessages(msgs)
	if len(out) != 1 {
		t.Fatalf("期望 1 条消息，得到 %d", len(out))
	}

	got := out[0]

	// 必须含 tool_calls 字段
	rawTC, ok := got["tool_calls"]
	if !ok {
		t.Fatalf("BUG-20260520: 含 ToolCalls 的 assistant 消息必须序列化出 tool_calls 字段，实际 keys=%v", keysOf(got))
	}

	tcs, ok := rawTC.([]map[string]any)
	if !ok {
		t.Fatalf("tool_calls 类型应为 []map[string]any，实际 %T", rawTC)
	}
	if len(tcs) != 1 {
		t.Fatalf("期望 1 个 tool_call，得到 %d", len(tcs))
	}

	tc := tcs[0]
	if tc["id"] != "call_abc123" {
		t.Errorf("tool_calls[0].id 期望 call_abc123，实际 %v", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_calls[0].type 期望 function，实际 %v", tc["type"])
	}
	fn, ok := tc["function"].(map[string]any)
	if !ok {
		t.Fatalf("tool_calls[0].function 类型异常，实际 %T", tc["function"])
	}
	if fn["name"] != "write_file" {
		t.Errorf("tool_calls[0].function.name 期望 write_file，实际 %v", fn["name"])
	}
	if fn["arguments"] != `{"path":"out.md","content":"hello"}` {
		t.Errorf("tool_calls[0].function.arguments 错误，实际 %v", fn["arguments"])
	}

	// content 为空时应序列化为 JSON null（OpenAI 规范：assistant + tool_calls 时 content 可省略或为 null）
	// 这是触发上游 Claude 网关 400 的关键——空字符串和 null 在网关翻译时行为不同。
	body, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("JSON 序列化失败: %v", err)
	}
	bodyStr := string(body)
	if !contains(bodyStr, `"tool_calls":[`) {
		t.Errorf("序列化 JSON 必须含 tool_calls 数组，实际: %s", bodyStr)
	}
	if !contains(bodyStr, `"content":null`) {
		t.Errorf("BUG-20260520: assistant + tool_calls 且 Content 为空时 content 应为 null，实际: %s", bodyStr)
	}
}

// TestBUG20260520_ToolMessageMustSerializeToolCallID
// 复现 RED：role=tool 消息必须序列化 tool_call_id，否则上游网关无法关联回原调用。
// （BUG-20260523 加固后必须配前置 assistant 形成合法序列，单条裸 tool 会被
// SanitizeToolCallSequence 正确剥离。本测试关注的是"字段必出"契约，
// 序列契约由 bug_20260523_orphan_tool_message_test.go 守护。）
func TestBUG20260520_ToolMessageMustSerializeToolCallID(t *testing.T) {
	msgs := []llm.Message{
		{
			Role:    llm.RoleAssistant,
			Content: "",
			ToolCalls: []llm.ToolCallRef{
				{ID: "call_abc123", Name: "write_file", Arguments: `{}`},
			},
		},
		{
			Role:       llm.RoleTool,
			Content:    "file written successfully",
			ToolCallID: "call_abc123",
		},
	}

	out := convertMessages(msgs)
	if len(out) != 2 {
		t.Fatalf("期望 2 条消息（assistant+tool 合法配对），得到 %d", len(out))
	}
	got := out[1] // tool message 是第 2 条

	if got["role"] != "tool" {
		t.Errorf("role 期望 tool，实际 %v", got["role"])
	}
	if got["tool_call_id"] != "call_abc123" {
		t.Fatalf("BUG-20260520: role=tool 消息必须含 tool_call_id 字段，实际 keys=%v", keysOf(got))
	}
	if got["content"] != "file written successfully" {
		t.Errorf("content 错误，实际 %v", got["content"])
	}
}

// TestBUG20260520_AssistantWithBothTextAndToolCalls
// 边界：assistant 同时含 text + tool_calls（reasoning model 常见格式）。
//
// BUG-20260523-v4 更新契约：当 ToolCalls 非空时，content **强制为 nil**
// （不再保留 text）——因为部分 OpenAI 兼容网关翻 Anthropic 时
// 遇到 text+tool_calls 共存会丢 tool_use 块，触发 400。
// 详见 bug_20260523_v4_content_null_test.go。
//
// 这里仅断言 tool_calls 字段不漏；content 行为由 v4 测试守护。
func TestBUG20260520_AssistantWithBothTextAndToolCalls(t *testing.T) {
	msgs := []llm.Message{
		{
			Role:    llm.RoleAssistant,
			Content: "我来帮你写文件",
			ToolCalls: []llm.ToolCallRef{
				{ID: "call_xyz", Name: "write_file", Arguments: `{}`},
			},
		},
	}
	out := convertMessages(msgs)
	if _, ok := out[0]["tool_calls"]; !ok {
		t.Errorf("BUG-20260520: assistant 同时含 text + tool_calls 时 tool_calls 字段不能丢失")
	}
}

// TestBUG20260520_FullConversationRoundTrip
// 端到端：模拟 hexclaw 真实场景——system + user + assistant(tool_calls) + tool + user
// 这正是用户报错时的 history 形态。整段序列化后 JSON 必须包含 tool_calls 和 tool_call_id。
func TestBUG20260520_FullConversationRoundTrip(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "you are helpful"},
		{Role: llm.RoleUser, Content: "把以上生成一个 md 文档"},
		{
			Role:    llm.RoleAssistant,
			Content: "",
			ToolCalls: []llm.ToolCallRef{
				{ID: "call_1", Name: "write_file", Arguments: `{"path":"out.md"}`},
			},
		},
		{Role: llm.RoleTool, Content: "ok", ToolCallID: "call_1"},
		{Role: llm.RoleUser, Content: "再加一段简介"},
	}

	out := convertMessages(msgs)
	body, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("JSON marshal: %v", err)
	}
	bodyStr := string(body)

	mustContain := []string{
		`"tool_calls":[`,
		`"id":"call_1"`,
		`"type":"function"`,
		`"name":"write_file"`,
		`"tool_call_id":"call_1"`,
		`"role":"tool"`,
	}
	for _, want := range mustContain {
		if !contains(bodyStr, want) {
			t.Errorf("BUG-20260520: 完整对话历史序列化必须含 %q\n实际: %s", want, bodyStr)
		}
	}
}

// --- helpers ---

func keysOf(m map[string]any) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
