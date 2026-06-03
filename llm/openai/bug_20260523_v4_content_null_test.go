// BUG-20260523-v4 assistant + tool_calls 共存时 content 强制 null
//
// 症状：用户截图错误 `unexpected tool_use_id found in tool_result blocks: toolu_011d...`
//      上游网关端实际收到的 assistant message 是：
//        {role:"assistant", content:[{type:"text",text:"我来帮你..."}], stopReason:"tool_calls"}
//      ← content 数组只有 text，tool_use 块**丢了**。
//
// 根因：部分 OpenAI 兼容网关（new-api / one-api 同源）翻 OpenAI → Anthropic 时，
//      assistant.content="text" + tool_calls=[...] 共存场景下，**只翻 text 丢 tool_use**。
//      下一条 tool_result 找不到对应 tool_use 立即 400。
//
// 修法：assistant + ToolCalls 共存时，content 强制设为 nil（不再仅 "" 时 nil）。
//      OpenAI 规范本身允许此形态；网关接到 content=null 翻译路径单一，不易踩 bug。
//      trade-off：丢失 LLM 输出的 reasoning text，但下一轮 LLM 仍能从 tool_calls 推断意图。
package openai

import (
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestBUG20260523v4_AssistantWithTextAndToolCallsForcesContentNull(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "请帮我做某事"},
		{
			Role:    llm.RoleAssistant,
			Content: "我来帮你创建这个定时任务。先了解一下当前的工作目录结构。", // ★ 非空 text
			ToolCalls: []llm.ToolCallRef{
				{ID: "toolu_011dRAGFHg28TQejxRonRx9m", Name: "list_allowed_directories", Arguments: "{}"},
			},
		},
		{Role: llm.RoleTool, Content: "Error executing tool", ToolCallID: "toolu_011dRAGFHg28TQejxRonRx9m"},
	}

	out := ConvertMessages(msgs)
	if len(out) != 4 {
		t.Fatalf("期望 4 条消息（合法配对），实际 %d", len(out))
	}

	assistant := out[2]
	if assistant["role"] != "assistant" {
		t.Fatalf("第 3 条期望 assistant，实际 %v", assistant["role"])
	}
	// 关键断言：content 必须是 nil（不能是非空字符串）
	if c, ok := assistant["content"]; !ok {
		t.Errorf("BUG-20260523-v4: assistant.content 字段缺失")
	} else if c != nil {
		t.Errorf("BUG-20260523-v4: assistant + tool_calls 共存时 content 必须为 nil，"+
			"实际 %T=%v —— 上游网关会丢 tool_use 块导致下一条 tool_result 400", c, c)
	}
	if _, ok := assistant["tool_calls"]; !ok {
		t.Errorf("BUG-20260523-v4: tool_calls 字段缺失（不能因强制 nil 而漏写 tool_calls）")
	}
}

func TestBUG20260523v4_AssistantWithoutToolCallsKeepsContent(t *testing.T) {
	// 反向断言：assistant 没 tool_calls 时 content 必须保留原文，不能误清。
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{Role: llm.RoleUser, Content: "hi"},
		{Role: llm.RoleAssistant, Content: "hello back"},
	}
	out := ConvertMessages(msgs)
	if len(out) != 3 {
		t.Fatalf("期望 3 条消息，实际 %d", len(out))
	}
	if out[2]["content"] != "hello back" {
		t.Errorf("BUG-20260523-v4: 普通 assistant（无 tool_calls）content 被误清，实际 %v", out[2]["content"])
	}
}
