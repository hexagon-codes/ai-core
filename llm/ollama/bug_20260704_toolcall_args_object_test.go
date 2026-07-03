package ollama

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// BUG-20260704：本地 Ollama 多轮工具调用返回 400。
//
// 症状：本地模型（如 qwen3.5:9b）在"问天气"等触发工具的场景下，工具执行完成后
// 第二轮请求 Ollama /api/chat 返回：
//
//	400 Bad Request, body: {"error":"Value looks like object, but can't find closing '}' symbol"}
//
// 根因：assistant 历史消息里的 tool_calls[].function.arguments 被沿用 OpenAI 协议
// 序列化成 JSON **字符串**（openai.ConvertMessages），而 Ollama 原生 /api/chat 要求
// arguments 是 JSON **对象**。适配器 convertMessagesForOllama 只翻译了 content/images，
// 漏把 arguments 反序列化回对象 → 字符串透传 → Ollama 二次解析炸。
//
// 复现依据（对本地 qwen3.5:9b 实测）：
//
//	arguments=对象 {"city":"杭州"}  → HTTP 200
//	arguments=字符串 "{\"city\":..}" → HTTP 400（本 bug）
//	arguments=空串 ""              → HTTP 400
//
// 本用例锁定：buildRequestBody 产出的 tool_calls[].function.arguments 必须是对象，
// 永久防回归。

// extractFirstToolCallArgs 从 buildRequestBody 产出的请求体里取第一个 tool_call 的
// arguments 原始 JSON，反序列化为 Go 值：对象 → map[string]any，字符串 → string。
func extractFirstToolCallArgs(t *testing.T, body []byte) any {
	t.Helper()
	var payload struct {
		Messages []struct {
			ToolCalls []struct {
				Function struct {
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("反序列化请求体失败: %v\nbody=%s", err, string(body))
	}
	for _, m := range payload.Messages {
		if len(m.ToolCalls) == 0 {
			continue
		}
		raw := m.ToolCalls[0].Function.Arguments
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			t.Fatalf("反序列化 arguments 原始值失败 %q: %v", string(raw), err)
		}
		return v
	}
	t.Fatalf("请求体里未找到 tool_calls: %s", string(body))
	return nil
}

func toolCallHistoryReq(args string) llm.CompletionRequest {
	return llm.CompletionRequest{
		Model: "qwen3.5:9b",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "杭州明天什么天气?"},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCallRef{
				{ID: "call_0", Name: "get_weather", Arguments: args},
			}},
			{Role: llm.RoleTool, ToolCallID: "call_0", Content: "晴 25C"},
		},
	}
}

func TestBuildRequestBody_ToolCallArgumentsAreObject_BUG20260704(t *testing.T) {
	p := New()
	body, err := p.buildRequestBody(toolCallHistoryReq(`{"city":"杭州"}`), false)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}

	args := extractFirstToolCallArgs(t, body)
	m, ok := args.(map[string]any)
	if !ok {
		t.Fatalf("BUG-20260704: Ollama /api/chat 要求 tool_calls.arguments 为 JSON 对象，"+
			"实际序列化为 %T（值=%v）；字符串会触发 400 \"Value looks like object, but can't find closing '}' symbol\"",
			args, args)
	}
	if m["city"] != "杭州" {
		t.Fatalf("arguments 对象内容错误: %v", m)
	}
}

func TestBuildRequestBody_EmptyToolCallArgumentsBecomeEmptyObject_BUG20260704(t *testing.T) {
	p := New()
	// 无参工具：模型可能给空串。Ollama 对空串同样报 400，必须归一化为空对象 {}。
	body, err := p.buildRequestBody(toolCallHistoryReq(""), false)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}

	args := extractFirstToolCallArgs(t, body)
	if _, ok := args.(map[string]any); !ok {
		t.Fatalf("BUG-20260704: 空参 arguments 必须序列化为空对象 {}，实际为 %T（值=%v）", args, args)
	}
}

// TestComplete_RealOllama_ToolCallHistory_NoBadRequest_BUG20260704 是真实环境 E2E harness。
//
// 走完整 Complete 路径（buildRequestBody → 真机 /api/chat → parseResponse），断言带
// 工具调用历史的第二轮请求不再返回 400。默认 skip；置 OLLAMA_E2E=1 且本机有 Ollama 时运行。
//
//	OLLAMA_E2E=1 go test ./llm/ollama/ -run RealOllama_ToolCallHistory -v
//
// 修复前该用例会以 `ollama request failed: ... 400 Bad Request, body:
// {"error":"Value looks like object, but can't find closing '}' symbol"}` 失败。
func TestComplete_RealOllama_ToolCallHistory_NoBadRequest_BUG20260704(t *testing.T) {
	if os.Getenv("OLLAMA_E2E") == "" {
		t.Skip("需真实 Ollama：设 OLLAMA_E2E=1 运行")
	}
	base := os.Getenv("OLLAMA_HOST")
	if base == "" {
		base = "http://localhost:11434"
	}
	model := os.Getenv("OLLAMA_E2E_MODEL")
	if model == "" {
		model = "qwen3.5:9b"
	}

	p := New(WithBaseURL(base), WithModel(model))
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := p.Ping(ctx); err != nil {
		t.Skipf("Ollama 不可达，跳过: %v", err)
	}

	req := toolCallHistoryReq(`{"city":"杭州"}`)
	req.MaxTokens = 16
	if _, err := p.Complete(ctx, req); err != nil {
		t.Fatalf("BUG-20260704 未修复：带工具调用历史的第二轮请求应成功，实际 error = %v", err)
	}
}
