// BUG-20260523 ConvertMessages 孤立 tool message 自愈测试
//
// 症状：Claude 上游返 400
//   ***.***.content.0: unexpected `tool_use_id` found in `tool_result` blocks: toolu_01MT...
//   Each `tool_result` block must have a corresponding `tool_use` block in the previous message.
//
// 根因：任何原因（上游/网关/中间层/历史持久化）导致 messages 序列里出现
//      "孤立 tool message"（其 ToolCallID 在前面最近的 assistant.ToolCalls 找不到）
//      上游 Claude 立即 400。
//
// 修法：ConvertMessages 入口先 SanitizeToolCallSequence —— 剥离所有孤立
//      tool message，确保发出去的序列契约级合规。
//
// 自愈契约：
//   每个 RoleTool message 的 ToolCallID 必须能在**前面任意位置**（直到上一条
//   user message 之前）的 RoleAssistant.ToolCalls 中找到匹配 ID。找不到 → 丢弃。
package openai

import (
	"encoding/json"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

// TestBUG20260523_OrphanToolMessageMustBeDropped
// 复现场景 1：tool message 完全孤立（前面没任何 assistant.ToolCalls）。
func TestBUG20260523_OrphanToolMessageMustBeDropped(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{Role: llm.RoleTool, Content: "Error: orphan", ToolCallID: "toolu_orphan"},
	}
	out := ConvertMessages(msgs)

	for _, m := range out {
		if m["role"] == "tool" && m["tool_call_id"] == "toolu_orphan" {
			t.Fatalf("BUG-20260523: 孤立 tool message 未被剥离 → 上游 Claude 400. out=%+v", out)
		}
	}
}

// TestBUG20260523_OrphanToolAfterUnrelatedAssistant
// 复现场景 2：前面有 assistant 但没 ToolCalls，依然孤立。
func TestBUG20260523_OrphanToolAfterUnrelatedAssistant(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{Role: llm.RoleAssistant, Content: "hello"}, // 没有 ToolCalls
		{Role: llm.RoleTool, Content: "result", ToolCallID: "toolu_orphan"},
	}
	out := ConvertMessages(msgs)

	for _, m := range out {
		if m["role"] == "tool" && m["tool_call_id"] == "toolu_orphan" {
			t.Fatalf("BUG-20260523: 孤立 tool message（前面 assistant 无 ToolCalls）未被剥离")
		}
	}
}

// TestBUG20260523_PartialOrphanInMultiToolCall
// 复现场景 3：assistant 含 2 个 ToolCalls [A, B]，但 tool message 序列含 [A, C(孤立)]
// → 保留 A 的 tool，剥离 C 的 tool（A 合法 / C 孤立）。
func TestBUG20260523_PartialOrphanInMultiToolCall(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{
			Role:    llm.RoleAssistant,
			Content: "",
			ToolCalls: []llm.ToolCallRef{
				{ID: "toolu_A", Name: "ta", Arguments: "{}"},
				{ID: "toolu_B", Name: "tb", Arguments: "{}"},
			},
		},
		{Role: llm.RoleTool, Content: "rA", ToolCallID: "toolu_A"},          // 合法
		{Role: llm.RoleTool, Content: "rC", ToolCallID: "toolu_C_orphan"},   // 孤立
	}
	out := ConvertMessages(msgs)

	var seenA, seenC bool
	for _, m := range out {
		if m["role"] == "tool" {
			switch m["tool_call_id"] {
			case "toolu_A":
				seenA = true
			case "toolu_C_orphan":
				seenC = true
			}
		}
	}
	if !seenA {
		t.Error("BUG-20260523: 合法 tool toolu_A 被误剥离")
	}
	if seenC {
		t.Error("BUG-20260523: 孤立 tool toolu_C_orphan 应被剥离")
	}
}

// TestBUG20260523_LegalToolCallSequenceMustPassThrough
// 反向断言：合法 assistant+tool 序列必须**完整保留**，sanitize 不能误伤。
func TestBUG20260523_LegalToolCallSequenceMustPassThrough(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{
			Role:    llm.RoleAssistant,
			Content: "",
			ToolCalls: []llm.ToolCallRef{
				{ID: "toolu_X", Name: "tx", Arguments: `{"p":1}`},
			},
		},
		{Role: llm.RoleTool, Content: "rX", ToolCallID: "toolu_X"},
		{Role: llm.RoleUser, Content: "next"},
	}
	out := ConvertMessages(msgs)
	if len(out) != len(msgs) {
		body, _ := json.Marshal(out)
		t.Fatalf("合法序列被误剥离：got %d, want %d. out=%s", len(out), len(msgs), body)
	}
}

// TestBUG20260523_HistoryReplayedOrphanIsDroppedNotAssistant
// 边界：assistant tool_use message 本身**不能**被当作"前面没对应"剥离 ——
// 它就是"前面的 tool_use"，是其它 tool 的归属。
func TestBUG20260523_HistoryReplayedOrphanIsDroppedNotAssistant(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{
			Role:    llm.RoleAssistant,
			Content: "",
			ToolCalls: []llm.ToolCallRef{{ID: "toolu_X", Name: "tx", Arguments: "{}"}},
		},
		// 漏 tool_X 结果，直接 user 问下一句 —— assistant 自己仍是合法保留对象
		{Role: llm.RoleUser, Content: "next"},
	}
	out := ConvertMessages(msgs)

	// assistant tool_call message 必须保留（它不是 orphan tool，它是 producer）
	var foundAssistantWithTC bool
	for _, m := range out {
		if m["role"] == "assistant" {
			if _, ok := m["tool_calls"]; ok {
				foundAssistantWithTC = true
			}
		}
	}
	if !foundAssistantWithTC {
		t.Error("BUG-20260523: 合法 assistant + tool_calls 被误剥离（它是 producer 不是 orphan）")
	}
}
