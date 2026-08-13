// BUG-20260523-v4 E2E 真实 HTTP 验证：assistant + tool_calls 共存时
// outbound HTTP body 的 content 字段实际是 JSON null。
//
// 用户截图错误 ID toolu_011dRAGFHg28TQejxRonRx9m + 用户真实 reasoning text
// "我来帮你创建这个定时任务。先了解一下当前的工作目录结构。" 构造端到端复现。
//
// 修前：上游网关收到 {"role":"assistant","content":"我来帮你...","tool_calls":[{...}]}
//
//	翻 Anthropic 时只翻 text 丢 tool_use → 400。
//
// 修后：上游网关收到 {"role":"assistant","content":null,"tool_calls":[{...}]}
//
//	翻译路径单一，不再丢 tool_use → 不再 400。
package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestBUG20260523v4_E2E_OutboundBodyContentMustBeNull(t *testing.T) {
	var capturedBody []byte
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		capturedBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"t","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer mockServer.Close()

	p := New("k", WithBaseURL(mockServer.URL), WithModel("test-model"))

	// 完整复现用户场景：reasoning text + tool_call ID（来自真实截图）
	req := llm.CompletionRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are 小蟹"},
			{Role: llm.RoleUser, Content: "创建一个定时任务，每天上午10点采集微博 TOP10 热搜，并加入知识库"},
			{
				Role:    llm.RoleAssistant,
				Content: "我来帮你创建这个定时任务。先了解一下当前的工作目录结构。", // 用户截图记录的真实 reasoning text
				ToolCalls: []llm.ToolCallRef{
					{ID: "toolu_011dRAGFHg28TQejxRonRx9m", Name: "list_allowed_directories", Arguments: "{}"},
				},
			},
			{Role: llm.RoleTool, Content: "Error executing tool", ToolCallID: "toolu_011dRAGFHg28TQejxRonRx9m"},
		},
	}

	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("E2E Complete 失败: %v", err)
	}

	bodyStr := string(capturedBody)
	t.Logf("捕获到 outbound body: %s", bodyStr)

	// 关键断言 1：outbound JSON 必须含 `"content":null`（精确匹配 JSON 字面量）
	if !strings.Contains(bodyStr, `"content":null`) {
		t.Errorf("BUG-20260523-v4: outbound JSON 缺 `\"content\":null` —— 上游网关翻译会丢 tool_use. body=%s", bodyStr)
	}

	// 关键断言 2：outbound JSON **不能含**原 reasoning text（如果含，说明 content 没被强制 null）
	if strings.Contains(bodyStr, "我来帮你创建这个定时任务") {
		t.Errorf("BUG-20260523-v4: outbound 仍含 reasoning text —— 上游网关翻 Anthropic 会丢 tool_use → 400")
	}

	// 关键断言 3：outbound 仍含 tool_calls + 真实 ID
	if !strings.Contains(bodyStr, "toolu_011dRAGFHg28TQejxRonRx9m") {
		t.Errorf("BUG-20260523-v4: outbound 缺真实 tool_call ID. body=%s", bodyStr)
	}

	// 解析验证结构
	var parsed struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(capturedBody, &parsed); err != nil {
		t.Fatalf("解析 body 失败: %v", err)
	}
	// 第 3 条应是 assistant
	if parsed.Messages[2]["role"] != "assistant" {
		t.Fatalf("第 3 条期望 assistant，实际 %v", parsed.Messages[2]["role"])
	}
	c, exists := parsed.Messages[2]["content"]
	if !exists {
		t.Error("assistant.content 字段缺失")
	}
	if c != nil {
		t.Errorf("BUG-20260523-v4: assistant.content 必须为 JSON null，实际 %T=%v", c, c)
	}
}
