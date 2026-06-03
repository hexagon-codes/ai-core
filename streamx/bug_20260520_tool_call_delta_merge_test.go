// Package streamx 回归测试：BUG-20260520（Part 2）
//
// 症状：Claude（经 OpenRouter / 任何 OpenAI 兼容网关）流式输出 tool_call 时，
//      被 hexclaw runtime 接收到 N 次 tool 调用——1 次有完整 name，N-1 次 tool=""。
//      畸形调用回灌到第二轮 prompt 历史，导致下游 17s 后返回 500 get_channel_failed。
//
// 根因：streamx/stream.go::mergeToolCalls 按 ID 合并，但 OpenAI streaming 协议
//      只有首个 delta 带 id/name/type，后续 delta 仅含 arguments 增量（id=""、name=""）。
//      按 ID 合并时 id 空 → 没匹配到 existing → append 成新 ToolCall。
//
// 正确做法：按 **index** 合并（OpenAI 协议规定）。
//
// 本测试用真实 Claude/OpenRouter streaming 格式覆盖。
package streamx

import (
	"strings"
	"testing"
)

// TestBUG20260520_ClaudeToolCallDeltaMustMergeByIndex
// Claude/OpenRouter 流式 tool_call 的典型形态：
//   - delta1: index=0, id="call_xyz", name="write_file", arguments=""
//   - delta2: index=0, arguments="{\"path"      (无 id/name)
//   - delta3: index=0, arguments="\":\"out.md\""
//   - delta4: index=0, arguments=",\"content\":\"hi\"}"
// 期望合并成 1 个完整的 ToolCall。
func TestBUG20260520_ClaudeToolCallDeltaMustMergeByIndex(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"write_file","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"out.md\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\"content\":\"hi\"}"}}]}}]}

data: [DONE]

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("BUG-20260520: 应合并为 1 个 ToolCall，实际 %d 个: %+v", len(result.ToolCalls), result.ToolCalls)
	}

	tc := result.ToolCalls[0]
	if tc.ID != "call_xyz" {
		t.Errorf("ToolCall.ID 期望 call_xyz，实际 %q", tc.ID)
	}
	if tc.Name != "write_file" {
		t.Errorf("BUG-20260520: ToolCall.Name 必须为 write_file（不能为空），实际 %q", tc.Name)
	}
	expected := `{"path":"out.md","content":"hi"}`
	if tc.Arguments != expected {
		t.Errorf("ToolCall.Arguments 期望 %q，实际 %q", expected, tc.Arguments)
	}
}

// TestBUG20260520_MultipleToolCallsByIndex 多个并行 tool_call，按 index 区分。
func TestBUG20260520_MultipleToolCallsByIndex(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"tool_a","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"tool_b","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"y\":2}"}}]}}]}

data: [DONE]

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}

	if len(result.ToolCalls) != 2 {
		t.Fatalf("BUG-20260520: 应合并为 2 个 ToolCall，实际 %d", len(result.ToolCalls))
	}

	byID := map[string]ToolCall{}
	for _, tc := range result.ToolCalls {
		byID[tc.ID] = tc
	}
	if byID["call_a"].Name != "tool_a" || byID["call_a"].Arguments != `{"x":1}` {
		t.Errorf("call_a 合并错误: %+v", byID["call_a"])
	}
	if byID["call_b"].Name != "tool_b" || byID["call_b"].Arguments != `{"y":2}` {
		t.Errorf("call_b 合并错误: %+v", byID["call_b"])
	}
}
