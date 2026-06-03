// BUG-20260523 端到端真实 HTTP 验证
//
// 启动 mock OpenAI 兼容 server，让 openai.Provider.Complete 真实走完
// 整条调用链（buildRequestBody → ConvertMessages → http.POST），
// 捕获 server 侧收到的 raw body，断言：
//   - 含孤立 tool message 的请求 → outbound body **不含**孤立 tool
//   - 用户截图错误中的 toolu_01MT726KsqWtAxwKGmyLb7Wq 这种 case，
//     用户走新 sidecar 后上游网关永远收不到此孤立 tool_use_id → 不再 400
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

func TestBUG20260523_E2E_OrphanToolStrippedFromHTTPBody(t *testing.T) {
	var capturedBody []byte
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		capturedBody = body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"chatcmpl-x","model":"test","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer mockServer.Close()

	p := New("test-key", WithBaseURL(mockServer.URL), WithModel("test-model"))

	// 复现用户场景：含孤立 tool message（前面没对应 assistant.ToolCalls）
	req := llm.CompletionRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are 小蟹"},
			{Role: llm.RoleUser, Content: "创建一个定时任务"},
			// 用户截图错误里的真实 ID：toolu_01MT726KsqWtAxwKGmyLb7Wq
			// 前面没有对应 assistant.ToolCalls，是 producer 丢失的孤立 case
			{Role: llm.RoleTool, Content: "Error executing tool", ToolCallID: "toolu_01MT726KsqWtAxwKGmyLb7Wq"},
		},
	}

	_, err := p.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("E2E Complete 失败: %v", err)
	}

	bodyStr := string(capturedBody)
	t.Logf("捕获到 outbound body: %s", bodyStr)

	// 强断言：outbound body **不含**孤立 tool_use_id
	if strings.Contains(bodyStr, "toolu_01MT726KsqWtAxwKGmyLb7Wq") {
		t.Errorf("BUG-20260523: outbound HTTP body 仍含孤立 tool_use_id —— "+
			"上游 Claude 仍会 400. body=%s", bodyStr)
	}

	// 验证 messages 结构：system + user，无 tool
	var parsed struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(capturedBody, &parsed); err != nil {
		t.Fatalf("解析 body 失败: %v", err)
	}
	for _, m := range parsed.Messages {
		if m["role"] == "tool" {
			t.Errorf("BUG-20260523: outbound 仍含 tool message（应被剥离）: %+v", m)
		}
	}
	if len(parsed.Messages) != 2 {
		t.Errorf("期望 2 条 message（system+user），实际 %d", len(parsed.Messages))
	}
}

func TestBUG20260523_E2E_LegalSequencePreservedInHTTPBody(t *testing.T) {
	var capturedBody []byte
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		capturedBody = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"t","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer mockServer.Close()

	p := New("k", WithBaseURL(mockServer.URL), WithModel("t"))

	// 合法序列：assistant + tool 配对
	req := llm.CompletionRequest{
		Model: "t",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "sys"},
			{Role: llm.RoleUser, Content: "hi"},
			{
				Role:    llm.RoleAssistant,
				Content: "",
				ToolCalls: []llm.ToolCallRef{
					{ID: "toolu_legit", Name: "write_file", Arguments: "{}"},
				},
			},
			{Role: llm.RoleTool, Content: "ok", ToolCallID: "toolu_legit"},
		},
	}

	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("E2E Complete 失败: %v", err)
	}

	bodyStr := string(capturedBody)
	if !strings.Contains(bodyStr, "toolu_legit") {
		t.Errorf("BUG-20260523: 合法 toolu_legit 被误剥离. body=%s", bodyStr)
	}
	if !strings.Contains(bodyStr, `"tool_calls"`) {
		t.Errorf("BUG-20260523: 合法 assistant.tool_calls 被误剥离. body=%s", bodyStr)
	}
	if !strings.Contains(bodyStr, `"tool_call_id":"toolu_legit"`) {
		t.Errorf("BUG-20260523: 合法 tool.tool_call_id 字段丢失. body=%s", bodyStr)
	}
}
