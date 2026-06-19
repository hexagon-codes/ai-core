// Package gemini 的审计测试文件 (Wave3)。
//
// 本文件针对 Gemini Provider 的公开 API 进行全场景测试，覆盖：
//   - 正常路径：请求构造 / 响应解析 / 流式 SSE 解析 / Embed
//   - 边界情况：空消息 / nil 指针 / Unicode / 0 与负数 / 超长 / 并发
//   - 错误路径：非 200 状态码 / 非法 JSON / 不可达 URL
//   - 状态一致性：Option 生效 / Header 注入 / URL 构造
//
// 测试通过 httptest.Server + WithBaseURL/WithHTTPClient 注入点拦截请求，
// 绝不访问真实 Google 网络。
package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// ptrFloat 返回 float64 指针，用于构造 Temperature/TopP 等可选字段。
func ptrFloat(f float64) *float64 { return &f }

// newTestProvider 创建一个指向 httptest.Server 的 Provider。
func newTestProvider(t *testing.T, handler http.HandlerFunc) (*Provider, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	p := New("test-key", WithBaseURL(srv.URL))
	return p, srv
}

// ============== New / Option 测试 ==============

// TestNew_Defaults 验证默认配置值。
func TestNew_Defaults(t *testing.T) {
	p := New("my-key")
	if p.apiKey != "my-key" {
		t.Errorf("apiKey = %q, want my-key", p.apiKey)
	}
	if p.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", p.baseURL, defaultBaseURL)
	}
	if p.model != defaultModel {
		t.Errorf("model = %q, want %q", p.model, defaultModel)
	}
	if p.httpClient == nil {
		t.Error("httpClient 不应为 nil")
	}
}

// TestNew_EnvFallback 验证空 apiKey 时从环境变量读取。
func TestNew_EnvFallback(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")

	t.Run("GOOGLE_API_KEY 优先", func(t *testing.T) {
		t.Setenv("GOOGLE_API_KEY", "google-key")
		t.Setenv("GEMINI_API_KEY", "gemini-key")
		p := New("")
		if p.apiKey != "google-key" {
			t.Errorf("apiKey = %q, want google-key", p.apiKey)
		}
	})

	t.Run("回退到 GEMINI_API_KEY", func(t *testing.T) {
		t.Setenv("GOOGLE_API_KEY", "")
		t.Setenv("GEMINI_API_KEY", "gemini-key")
		p := New("")
		if p.apiKey != "gemini-key" {
			t.Errorf("apiKey = %q, want gemini-key", p.apiKey)
		}
	})

	t.Run("均为空时 apiKey 为空", func(t *testing.T) {
		t.Setenv("GOOGLE_API_KEY", "")
		t.Setenv("GEMINI_API_KEY", "")
		p := New("")
		if p.apiKey != "" {
			t.Errorf("apiKey = %q, want empty", p.apiKey)
		}
	})
}

// TestOptions_Override 验证各 Option 生效。
func TestOptions_Override(t *testing.T) {
	custom := &http.Client{}
	p := New("k",
		WithBaseURL("http://custom"),
		WithModel("my-model"),
		WithHTTPClient(custom),
	)
	if p.baseURL != "http://custom" {
		t.Errorf("baseURL = %q", p.baseURL)
	}
	if p.model != "my-model" {
		t.Errorf("model = %q", p.model)
	}
	if p.httpClient != custom {
		t.Error("httpClient 未被替换为自定义客户端")
	}
}

// TestName 验证 Provider 名称。
func TestName(t *testing.T) {
	if got := New("k").Name(); got != "gemini" {
		t.Errorf("Name() = %q, want gemini", got)
	}
}

// ============== Complete 正常路径 ==============

// TestComplete_Success 验证正常补全：请求 URL / Header / 响应解析。
func TestComplete_Success(t *testing.T) {
	var capturedURL, capturedKey, capturedCT string
	var capturedBody []byte

	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.Path
		capturedKey = r.Header.Get("x-goog-api-key")
		capturedCT = r.Header.Get("Content-Type")
		capturedBody, _ = io.ReadAll(r.Body)
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"parts":[{"text":"你好世界"}],"role":"model"},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}
		}`))
	})

	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Model:    "gemini-1.5-pro",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if resp.Content != "你好世界" {
		t.Errorf("Content = %q, want 你好世界", resp.Content)
	}
	if resp.FinishReason != "STOP" {
		t.Errorf("FinishReason = %q", resp.FinishReason)
	}
	if resp.Model != "gemini-1.5-pro" {
		t.Errorf("Model = %q", resp.Model)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("TotalTokens = %d, want 15", resp.Usage.TotalTokens)
	}
	// URL 应使用指定 model 而非默认值
	if !strings.Contains(capturedURL, "gemini-1.5-pro:generateContent") {
		t.Errorf("URL path = %q, 应包含 gemini-1.5-pro:generateContent", capturedURL)
	}
	if capturedKey != "test-key" {
		t.Errorf("x-goog-api-key = %q, want test-key", capturedKey)
	}
	if capturedCT != "application/json" {
		t.Errorf("Content-Type = %q", capturedCT)
	}
	// 验证 body 是合法 JSON 且 contents 存在
	var payload map[string]any
	if err := json.Unmarshal(capturedBody, &payload); err != nil {
		t.Fatalf("请求体不是合法 JSON: %v", err)
	}
	if _, ok := payload["contents"]; !ok {
		t.Error("请求体缺少 contents 字段")
	}
}

// TestComplete_DefaultModel 验证 req.Model 为空时使用 provider 默认 model。
func TestComplete_DefaultModel(t *testing.T) {
	var capturedURL string
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.Path
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"x"}]}}]}`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if !strings.Contains(capturedURL, defaultModel+":generateContent") {
		t.Errorf("URL path = %q, 应使用默认 model %q", capturedURL, defaultModel)
	}
}

// TestComplete_SystemInstruction 验证 system 消息被转换为 systemInstruction。
func TestComplete_SystemInstruction(t *testing.T) {
	var payload map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "你是助手"},
			{Role: llm.RoleUser, Content: "hi"},
		},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if _, ok := payload["systemInstruction"]; !ok {
		t.Error("缺少 systemInstruction 字段")
	}
	// system 消息不应混入 contents
	contents, _ := payload["contents"].([]any)
	if len(contents) != 1 {
		t.Errorf("contents 长度 = %d, want 1 (system 应被剥离)", len(contents))
	}
}

// TestComplete_GenerationConfig 验证生成配置参数透传。
func TestComplete_GenerationConfig(t *testing.T) {
	var payload map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		MaxTokens:   100,
		Temperature: ptrFloat(0.7),
		TopP:        ptrFloat(0.9),
		Stop:        []string{"END"},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	gc, ok := payload["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("缺少 generationConfig")
	}
	if gc["maxOutputTokens"].(float64) != 100 {
		t.Errorf("maxOutputTokens = %v", gc["maxOutputTokens"])
	}
	if gc["temperature"].(float64) != 0.7 {
		t.Errorf("temperature = %v", gc["temperature"])
	}
	if gc["topP"].(float64) != 0.9 {
		t.Errorf("topP = %v", gc["topP"])
	}
	if _, ok := gc["stopSequences"]; !ok {
		t.Error("缺少 stopSequences")
	}
}

// TestComplete_ResponseFormat 验证 ResponseFormat 转换为 responseMimeType。
func TestComplete_ResponseFormat(t *testing.T) {
	tests := []struct {
		name       string
		format     *llm.ResponseFormat
		wantMime   bool
		wantSchema bool
	}{
		{
			name:     "json_object",
			format:   &llm.ResponseFormat{Type: "json_object"},
			wantMime: true,
		},
		{
			name: "json_schema 带 schema",
			format: &llm.ResponseFormat{
				Type: "json_schema",
				JSONSchema: &llm.ResponseFormatJSONSchema{
					Name:   "test",
					Schema: &llm.Schema{Type: "object"},
				},
			},
			wantMime:   true,
			wantSchema: true,
		},
		{
			name:   "未知类型不设置",
			format: &llm.ResponseFormat{Type: "text"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var payload map[string]any
			p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &payload)
				_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"{}"}]}}]}`))
			})
			_, err := p.Complete(context.Background(), llm.CompletionRequest{
				Messages:       []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
				ResponseFormat: tt.format,
			})
			if err != nil {
				t.Fatalf("Complete 错误: %v", err)
			}
			gc, _ := payload["generationConfig"].(map[string]any)
			if tt.wantMime {
				if gc == nil || gc["responseMimeType"] != "application/json" {
					t.Errorf("responseMimeType 未正确设置, gc=%v", gc)
				}
			} else if gc != nil {
				if _, ok := gc["responseMimeType"]; ok {
					t.Errorf("不应设置 responseMimeType, gc=%v", gc)
				}
			}
			if tt.wantSchema {
				if gc == nil || gc["responseSchema"] == nil {
					t.Error("responseSchema 未设置")
				}
			}
		})
	}
}

// TestComplete_Tools 验证工具定义被转换为 functionDeclarations。
func TestComplete_Tools(t *testing.T) {
	var payload map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Tools: []llm.ToolDefinition{
			llm.NewToolDefinition("get_weather", "查询天气", &llm.Schema{Type: "object"}),
		},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools 字段异常: %v", payload["tools"])
	}
	tool0 := tools[0].(map[string]any)
	fds, ok := tool0["functionDeclarations"].([]any)
	if !ok || len(fds) != 1 {
		t.Fatalf("functionDeclarations 异常: %v", tool0)
	}
	fd0 := fds[0].(map[string]any)
	if fd0["name"] != "get_weather" {
		t.Errorf("function name = %v", fd0["name"])
	}
}

// TestComplete_ToolCallResponse 验证响应中的 functionCall 被解析为 ToolCall。
func TestComplete_ToolCallResponse(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"parts":[
				{"functionCall":{"name":"get_weather","args":{"city":"北京"}}}
			],"role":"model"},"finishReason":"STOP"}]
		}`))
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "天气"}},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if !resp.HasToolCalls() {
		t.Fatal("应包含工具调用")
	}
	tc := resp.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q", tc.Name)
	}
	if tc.Type != "function" {
		t.Errorf("ToolCall.Type = %q", tc.Type)
	}
	// 参数应为合法 JSON
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
		t.Errorf("Arguments 不是合法 JSON: %v", err)
	}
	if args["city"] != "北京" {
		t.Errorf("args[city] = %v", args["city"])
	}
}

// TestComplete_MultiPartTextConcat 验证多个 text part 被拼接。
func TestComplete_MultiPartTextConcat(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"parts":[
				{"text":"hello "},{"text":"world"}
			]}}]
		}`))
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if resp.Content != "hello world" {
		t.Errorf("Content = %q, want 'hello world'", resp.Content)
	}
}

// TestComplete_IDAndCreated 验证 parseResponse 填充 ID 与 Created。
//
// 回归: W3-55
// Gemini 的 generateContent API 不返回 id/created，旧实现导致 CompletionResponse
// 的 ID 与 Created 恒为零值。修复后应生成非空 ID 并填充 Created 时间戳。
func TestComplete_IDAndCreated(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	})
	before := time.Now().Unix()
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if resp.ID == "" {
		t.Error("resp.ID 不应为空 (应生成响应 ID)")
	}
	after := time.Now().Unix()
	if resp.Created < before || resp.Created > after {
		t.Errorf("resp.Created = %d, 应落在 [%d, %d] 之间 (应填充创建时间戳)", resp.Created, before, after)
	}
}

// ============== Complete 边界与错误路径 ==============

// TestComplete_EmptyMessages 验证空消息列表不 panic，仍发出合法请求。
func TestComplete_EmptyMessages(t *testing.T) {
	var payload map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"candidates":[]}`))
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: nil,
	})
	if err != nil {
		t.Fatalf("空消息不应报错: %v", err)
	}
	if resp.Content != "" {
		t.Errorf("无候选时 Content 应为空, got %q", resp.Content)
	}
}

// TestComplete_NoCandidates 验证响应无 candidates 时不 panic 且返回空内容。
func TestComplete_NoCandidates(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"usageMetadata":{"totalTokenCount":3}}`))
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if resp.Content != "" {
		t.Errorf("Content 应为空, got %q", resp.Content)
	}
	if resp.Usage.TotalTokens != 3 {
		t.Errorf("usage 仍应解析, got %d", resp.Usage.TotalTokens)
	}
}

// TestComplete_Non200 验证非 200 响应返回错误并包含状态与 body。
func TestComplete_Non200(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
	}{
		{"400 错误", http.StatusBadRequest, `{"error":"invalid"}`},
		{"401 未授权", http.StatusUnauthorized, `unauthorized`},
		{"429 限流", http.StatusTooManyRequests, `rate limited`},
		{"500 服务端错误", http.StatusInternalServerError, `boom`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := p.Complete(context.Background(), llm.CompletionRequest{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			if err == nil {
				t.Fatal("非 200 状态应返回错误")
			}
			if !strings.Contains(err.Error(), "gemini api error") {
				t.Errorf("错误信息应包含 'gemini api error', got %v", err)
			}
			if !strings.Contains(err.Error(), tt.body) {
				t.Errorf("错误信息应包含响应 body %q, got %v", tt.body, err)
			}
		})
	}
}

// TestComplete_InvalidJSON 验证 200 但响应体非法 JSON 时返回解析错误。
func TestComplete_InvalidJSON(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{not valid json`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("非法 JSON 响应应返回错误")
	}
}

// TestComplete_UnreachableURL 验证不可达地址返回错误。
func TestComplete_UnreachableURL(t *testing.T) {
	// 127.0.0.1:1 几乎一定拒绝连接
	p := New("k", WithBaseURL("http://127.0.0.1:1"))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("不可达地址应返回错误")
	}
}

// TestComplete_InvalidBaseURL 验证非法 baseURL 在构造 http.Request 时报错。
func TestComplete_InvalidBaseURL(t *testing.T) {
	p := New("k", WithBaseURL("://bad-url"))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("非法 baseURL 应返回错误")
	}
}

// TestComplete_ContextCancelled 验证已取消的 context 立即返回错误。
func TestComplete_ContextCancelled(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[]}`))
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.Complete(ctx, llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("已取消 context 应返回错误")
	}
}

// TestComplete_UnicodeAndLong 验证 Unicode 与超长内容正常往返。
func TestComplete_UnicodeAndLong(t *testing.T) {
	long := strings.Repeat("龙🐉", 5000) // 含 emoji 与中文
	var captured string
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = string(body)
		resp, _ := json.Marshal(map[string]any{
			"candidates": []map[string]any{
				{"content": map[string]any{"parts": []map[string]any{{"text": long}}}},
			},
		})
		_, _ = w.Write(resp)
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: long}},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if resp.Content != long {
		t.Error("超长 Unicode 内容往返不一致")
	}
	if !json.Valid([]byte(captured)) {
		t.Error("含 Unicode 的请求体不是合法 JSON")
	}
}

// ============== buildRequestBody 角色转换 ==============

// TestRoleConversion 验证角色转换逻辑（包括 RoleTool 的处理）。
//
// 回归: W3-53
// Gemini 没有独立的 "tool" 角色，工具结果必须以 functionResponse part 的形式、
// 在 role="user"(Gemini 文档约定的 functionResponse 归属角色) 的 content 中回传，
// 而不是把工具结果文本当成普通 user 文本发出去（丢失工具语义）。
func TestRoleConversion(t *testing.T) {
	var payload struct {
		Contents []struct {
			Role  string `json:"role"`
			Parts []struct {
				Text             string `json:"text"`
				FunctionResponse *struct {
					Name     string         `json:"name"`
					Response map[string]any `json:"response"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"candidates":[]}`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "u"},
			{Role: llm.RoleAssistant, Content: "a"},
			// 工具结果消息：携带 ToolCallID 关联调用，Content 为工具返回内容
			{Role: llm.RoleTool, Content: "晴天 25 度", ToolCallID: "call_get_weather"},
		},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if len(payload.Contents) != 3 {
		t.Fatalf("contents 长度 = %d, want 3", len(payload.Contents))
	}
	if payload.Contents[0].Role != "user" {
		t.Errorf("user -> %q", payload.Contents[0].Role)
	}
	if payload.Contents[1].Role != "model" {
		t.Errorf("assistant -> %q", payload.Contents[1].Role)
	}
	// RoleTool 应被转换为携带 functionResponse part 的 content，role 为 "user"
	tool := payload.Contents[2]
	if tool.Role != "user" {
		t.Errorf("tool content role = %q, want user", tool.Role)
	}
	if len(tool.Parts) != 1 {
		t.Fatalf("tool parts 长度 = %d, want 1", len(tool.Parts))
	}
	fr := tool.Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("tool part 应为 functionResponse, got %+v (工具结果被当成普通 user 文本，丢失工具语义)", tool.Parts[0])
	}
	if fr.Name == "" {
		t.Error("functionResponse.name 不应为空 (应来自 ToolCallID 关联)")
	}
	if fr.Response == nil {
		t.Error("functionResponse.response 不应为空 (应承载工具返回内容)")
	}
}

// TestBuildRequestBody_MultiContent 验证多模态 MultiContent 被正确编码进请求体。
//
// 回归: W3-52
// Gemini 的 Models() 声明 FeatureVision，buildRequestBody 必须把 msg.MultiContent
// 的文本与图片转换为 Gemini 的 parts：文本 → {text}，图片 →
// {inlineData:{mimeType,data}}（base64 data URI）或 {fileData:{fileUri,mimeType}}（http URL）。
func TestBuildRequestBody_MultiContent(t *testing.T) {
	var payload struct {
		Contents []struct {
			Role  string `json:"role"`
			Parts []struct {
				Text       string `json:"text"`
				InlineData *struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData"`
				FileData *struct {
					FileURI  string `json:"fileUri"`
					MimeType string `json:"mimeType"`
				} `json:"fileData"`
			} `json:"parts"`
		} `json:"contents"`
	}
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("请求体非法 JSON: %v", err)
		}
		_, _ = w.Write([]byte(`{"candidates":[]}`))
	})
	const dataURI = "data:image/png;base64,iVBORw0KGgo="
	msg := llm.Message{
		Role: llm.RoleUser,
		MultiContent: []llm.ContentPart{
			llm.NewTextPart("描述这张图"),
			llm.NewImageURLPart("https://example.com/cat.png", "auto"),
			llm.NewImageURLPart(dataURI, "auto"),
		},
	}
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{msg},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if len(payload.Contents) != 1 {
		t.Fatalf("contents 长度 = %d, want 1", len(payload.Contents))
	}
	parts := payload.Contents[0].Parts
	if len(parts) != 3 {
		t.Fatalf("parts 长度 = %d, want 3 (文本 + http 图片 + base64 图片)", len(parts))
	}
	// part0: 多模态文本
	if parts[0].Text != "描述这张图" {
		t.Errorf("parts[0].Text = %q, want 描述这张图", parts[0].Text)
	}
	// part1: http URL 图片 → fileData
	if parts[1].FileData == nil {
		t.Fatalf("parts[1] 应为 fileData (http URL 图片), got %+v", parts[1])
	}
	if parts[1].FileData.FileURI != "https://example.com/cat.png" {
		t.Errorf("parts[1].fileData.fileUri = %q", parts[1].FileData.FileURI)
	}
	// part2: base64 data URI → inlineData
	if parts[2].InlineData == nil {
		t.Fatalf("parts[2] 应为 inlineData (base64 图片), got %+v", parts[2])
	}
	if parts[2].InlineData.MimeType != "image/png" {
		t.Errorf("parts[2].inlineData.mimeType = %q, want image/png", parts[2].InlineData.MimeType)
	}
	if parts[2].InlineData.Data != "iVBORw0KGgo=" {
		t.Errorf("parts[2].inlineData.data = %q, want 解析后的 base64 负载", parts[2].InlineData.Data)
	}
}

// ============== Stream 测试 ==============

// sseGeminiResponse 构造 Gemini 的 SSE 流式响应文本。
func sseGeminiResponse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: ")
		b.WriteString(e)
		b.WriteString("\n\n")
	}
	return b.String()
}

// TestStream_Success 验证流式响应解析与内容拼接。
func TestStream_Success(t *testing.T) {
	var capturedURL string
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.String()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseGeminiResponse(
			`{"candidates":[{"content":{"parts":[{"text":"你好"}],"role":"model"}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"，世界"}],"role":"model"}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"！"}],"role":"model"},"finishReason":"STOP"}]}`,
		)))
	})

	s, err := p.Stream(context.Background(), llm.CompletionRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream 错误: %v", err)
	}
	result, err := s.Collect()
	if err != nil {
		t.Fatalf("Collect 错误: %v", err)
	}
	if result.Content != "你好，世界！" {
		t.Errorf("拼接内容 = %q, want 你好，世界！", result.Content)
	}
	if result.FinishReason != "STOP" {
		t.Errorf("FinishReason = %q", result.FinishReason)
	}
	// 流式 URL 应包含 streamGenerateContent?alt=sse
	if !strings.Contains(capturedURL, "streamGenerateContent") || !strings.Contains(capturedURL, "alt=sse") {
		t.Errorf("流式 URL = %q, 应包含 streamGenerateContent?alt=sse", capturedURL)
	}
}

// TestStream_DefaultModel 验证流式默认 model 透传。
func TestStream_DefaultModel(t *testing.T) {
	var capturedURL string
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseGeminiResponse(
			`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}`,
		)))
	})
	s, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream 错误: %v", err)
	}
	_, _ = s.Collect()
	if !strings.Contains(capturedURL, defaultModel+":streamGenerateContent") {
		t.Errorf("URL = %q, 应使用默认 model", capturedURL)
	}
}

// TestStream_Non200 验证流式非 200 状态返回错误且含 body。
func TestStream_Non200(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("stream boom"))
	})
	_, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("流式非 200 应返回错误")
	}
	if !strings.Contains(err.Error(), "gemini api error") {
		t.Errorf("错误信息 = %v", err)
	}
	if !strings.Contains(err.Error(), "stream boom") {
		t.Errorf("错误信息应含响应 body, got %v", err)
	}
}

// TestStream_UnreachableURL 验证流式不可达地址返回错误。
func TestStream_UnreachableURL(t *testing.T) {
	p := New("k", WithBaseURL("http://127.0.0.1:1"))
	_, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("不可达地址流式应返回错误")
	}
}

// TestStream_InvalidBaseURL 验证非法 baseURL 流式报错。
func TestStream_InvalidBaseURL(t *testing.T) {
	p := New("k", WithBaseURL("://bad"))
	_, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("非法 baseURL 流式应返回错误")
	}
}

// ============== Embed 测试 ==============

// TestEmbed_Success 验证批量嵌入正常解析。
func TestEmbed_Success(t *testing.T) {
	var capturedURL, capturedKey string
	var payload map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.Path
		capturedKey = r.Header.Get("x-goog-api-key")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"embeddings":[{"values":[0.1,0.2,0.3]},{"values":[0.4,0.5]}]}`))
	})
	got, err := p.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed 错误: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("embeddings 数量 = %d, want 2", len(got))
	}
	if len(got[0]) != 3 || got[0][0] != 0.1 {
		t.Errorf("got[0] = %v", got[0])
	}
	if len(got[1]) != 2 {
		t.Errorf("got[1] 长度 = %d", len(got[1]))
	}
	// 默认嵌入模型应为 text-embedding-004
	if !strings.Contains(capturedURL, "text-embedding-004:batchEmbedContents") {
		t.Errorf("嵌入 URL = %q", capturedURL)
	}
	if capturedKey != "test-key" {
		t.Errorf("嵌入未注入 api key, got %q", capturedKey)
	}
	// requests 数量应与文本数量一致
	reqs, _ := payload["requests"].([]any)
	if len(reqs) != 2 {
		t.Errorf("requests 数量 = %d, want 2", len(reqs))
	}
}

// TestEmbedWithModel_CustomModel 验证自定义嵌入模型透传到 URL 与请求体。
func TestEmbedWithModel_CustomModel(t *testing.T) {
	var capturedURL string
	var payload map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		_, _ = w.Write([]byte(`{"embeddings":[{"values":[1.0]}]}`))
	})
	_, err := p.EmbedWithModel(context.Background(), "custom-embed", []string{"x"})
	if err != nil {
		t.Fatalf("EmbedWithModel 错误: %v", err)
	}
	if !strings.Contains(capturedURL, "custom-embed:batchEmbedContents") {
		t.Errorf("URL = %q, 应含 custom-embed", capturedURL)
	}
	reqs, _ := payload["requests"].([]any)
	if len(reqs) != 1 {
		t.Fatalf("requests 数量 = %d", len(reqs))
	}
	r0 := reqs[0].(map[string]any)
	if r0["model"] != "models/custom-embed" {
		t.Errorf("请求体 model = %v, want models/custom-embed", r0["model"])
	}
}

// TestEmbed_EmptyTexts 验证空文本切片：发出合法请求且不 panic。
func TestEmbed_EmptyTexts(t *testing.T) {
	var captured string
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured = string(body)
		_, _ = w.Write([]byte(`{"embeddings":[]}`))
	})
	got, err := p.Embed(context.Background(), []string{})
	if err != nil {
		t.Fatalf("空文本不应报错: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("空输入应返回空, got %d", len(got))
	}
	if !json.Valid([]byte(captured)) {
		t.Errorf("空输入请求体应为合法 JSON, got %q", captured)
	}
}

// TestEmbed_Non200 验证嵌入非 200 状态返回错误。
func TestEmbed_Non200(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte("no access"))
	})
	_, err := p.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("嵌入非 200 应返回错误")
	}
	if !strings.Contains(err.Error(), "gemini embed error") {
		t.Errorf("错误信息 = %v", err)
	}
	if !strings.Contains(err.Error(), "no access") {
		t.Errorf("错误信息应含 body, got %v", err)
	}
}

// TestEmbed_InvalidJSON 验证嵌入响应非法 JSON 返回错误。
func TestEmbed_InvalidJSON(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{bad`))
	})
	_, err := p.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("非法 JSON 应返回错误")
	}
}

// TestEmbed_UnreachableURL 验证嵌入不可达地址返回错误。
func TestEmbed_UnreachableURL(t *testing.T) {
	p := New("k", WithBaseURL("http://127.0.0.1:1"))
	_, err := p.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("不可达地址应返回错误")
	}
}

// ============== Models / CountTokens ==============

// TestModels 验证模型列表非空且字段完整。
func TestModels(t *testing.T) {
	models := New("k").Models()
	if len(models) == 0 {
		t.Fatal("Models() 不应为空")
	}
	for _, m := range models {
		if m.ID == "" {
			t.Errorf("模型 ID 为空: %+v", m)
		}
		if m.MaxTokens <= 0 {
			t.Errorf("模型 %s MaxTokens 非正", m.ID)
		}
		if !m.HasFeature(llm.FeatureStreaming) {
			t.Errorf("模型 %s 应支持 streaming", m.ID)
		}
	}
}

// TestCountTokens 验证 Token 计数逻辑。
//
// 回归: W3-54
// 不再使用 len/4 整数除法（短内容会被截断为 0），而是用 tokenizer 精确估算。
// 关键不变量：非空内容必须计出 > 0 的 token；多条消息累加单调递增。
func TestCountTokens(t *testing.T) {
	p := New("k")

	// 空列表为 0
	if got, err := p.CountTokens(nil); err != nil || got != 0 {
		t.Errorf("空列表 CountTokens = %d, err=%v, want 0", got, err)
	}

	// 短内容（"abc"）不应再被整数除法截断为 0
	short, err := p.CountTokens([]llm.Message{{Role: llm.RoleUser, Content: "abc"}})
	if err != nil {
		t.Fatalf("CountTokens 错误: %v", err)
	}
	if short <= 0 {
		t.Errorf("短内容 CountTokens = %d, want > 0 (旧实现 3/4=0 截断为零)", short)
	}

	// 单调性：更长内容应计出不少于更短内容的 token
	longMsgs := []llm.Message{
		{Role: llm.RoleUser, Content: "abcdefgh"},
		{Role: llm.RoleAssistant, Content: "abcd"},
	}
	long, err := p.CountTokens(longMsgs)
	if err != nil {
		t.Fatalf("CountTokens 错误: %v", err)
	}
	if long < short {
		t.Errorf("多条累加 = %d 应 >= 单条 %d", long, short)
	}
}

// TestCountTokens_CountsMultiContent 验证 CountTokens 计入多模态内容。
//
// 回归: W3-54
// 当消息只用 MultiContent 携带文本时，Content 为空，旧实现估算为 0；
// 修复后必须把 MultiContent 的文本计入 token 统计。
func TestCountTokens_CountsMultiContent(t *testing.T) {
	p := New("k")
	msg := llm.Message{
		Role: llm.RoleUser,
		MultiContent: []llm.ContentPart{
			llm.NewTextPart(strings.Repeat("x", 400)),
		},
	}
	got, err := p.CountTokens([]llm.Message{msg})
	if err != nil {
		t.Fatalf("CountTokens 错误: %v", err)
	}
	if got <= 0 {
		t.Errorf("CountTokens = %d, 应计入 MultiContent 文本 (旧实现忽略 MultiContent 为 0)", got)
	}
}

// ============== 并发竞态 ==============

// TestConcurrent_Complete 验证同一 Provider 并发调用 Complete 安全（-race 下）。
func TestConcurrent_Complete(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	})

	const n = 50
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := p.Complete(context.Background(), llm.CompletionRequest{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			if err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("并发 Complete 错误: %v", err)
	}
}

// TestConcurrent_MixedOps 验证 Complete/Stream/Embed/CountTokens 混合并发安全。
func TestConcurrent_MixedOps(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "batchEmbedContents") {
			_, _ = w.Write([]byte(`{"embeddings":[{"values":[0.1]}]}`))
			return
		}
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(sseGeminiResponse(
				`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}`,
			)))
			return
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	})

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			req := llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}}
			switch i % 4 {
			case 0:
				_, _ = p.Complete(ctx, req)
			case 1:
				if s, err := p.Stream(ctx, req); err == nil {
					_, _ = s.Collect()
				}
			case 2:
				_, _ = p.Embed(ctx, []string{"x"})
			case 3:
				_, _ = p.CountTokens(req.Messages)
			}
		}(i)
	}
	wg.Wait()
}
