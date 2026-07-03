// Package qwen 的审计测试（Wave3）。
//
// 本文件对 qwen Provider 做全场景黑盒 + 受控白盒审计，覆盖：
//   - 正常路径：Name/Models/New 默认值/CountTokens/EmbeddingDimension
//   - 请求构造：model 默认、temperature/top_p/max_tokens/stop/tools/tool_choice、
//     enable_search、response_format 是否正确写入请求体（通过 httptest 捕获）
//   - 响应解析：Complete JSON 解析、tool_calls 解析、Usage 统计
//   - 流式 SSE：Stream 的 OpenAI 兼容 SSE 解析（通过 httptest 注入）
//   - 错误路径：非 200 响应、JSON 解析失败、cancelled context
//   - 边界：空 texts、Unicode、空内容、多 choice 仅取首个、embedding 乱序按 index 排序
//   - 状态一致性 + 并发竞态：Models/CountTokens 多协程调用、Provider 复用
//   - Option 生效：WithBaseURL/WithModel/WithHTTPClient 注入点
//
// qwen 包暴露 WithBaseURL/WithHTTPClient 注入点（对比 Wave1 B17 中
// deepseek 早期不可切网关的问题），因此可借助 httptest 在不打真实网络的前提下
// 完整验证请求构造与响应解析。本文件全程使用 httptest.Server，绝不打真实网络。
package qwen

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

// floatPtr 返回 float64 指针，便于构造 Temperature/TopP 等可选字段。
func floatPtr(f float64) *float64 { return &f }

// newTestProvider 返回一个指向 httptest.Server 的 Provider。
// 同时返回服务端最近一次接收到的请求体（解析为 map），供断言请求构造。
func newTestProvider(t *testing.T, handler http.HandlerFunc) (*Provider, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	p := New("test-key", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	return p, srv
}

// ---------------------------------------------------------------------------
// 正常路径：基础元信息
// ---------------------------------------------------------------------------

// TestName 验证 Name 固定返回 "qwen"。
func TestName(t *testing.T) {
	if got := New("k").Name(); got != "qwen" {
		t.Fatalf("Name() = %q, 期望 %q", got, "qwen")
	}
}

// TestNew_Defaults 验证 New 的默认 baseURL / model 生效。
func TestNew_Defaults(t *testing.T) {
	p := New("k")
	if p.baseURL != defaultBaseURL {
		t.Errorf("默认 baseURL = %q, 期望 %q", p.baseURL, defaultBaseURL)
	}
	if p.model != defaultModel {
		t.Errorf("默认 model = %q, 期望 %q", p.model, defaultModel)
	}
	if p.httpClient == nil {
		t.Error("默认 httpClient 不应为 nil")
	}
}

// TestNew_EnvFallback 验证空 apiKey 时按 DASHSCOPE_API_KEY -> QWEN_API_KEY 顺序回退。
func TestNew_EnvFallback(t *testing.T) {
	t.Run("DASHSCOPE 优先", func(t *testing.T) {
		t.Setenv("DASHSCOPE_API_KEY", "dash-key")
		t.Setenv("QWEN_API_KEY", "qwen-key")
		p := New("")
		if p.apiKey != "dash-key" {
			t.Fatalf("apiKey = %q, 期望优先取 DASHSCOPE_API_KEY=dash-key", p.apiKey)
		}
	})
	t.Run("回退到 QWEN_API_KEY", func(t *testing.T) {
		t.Setenv("DASHSCOPE_API_KEY", "")
		t.Setenv("QWEN_API_KEY", "qwen-key")
		p := New("")
		if p.apiKey != "qwen-key" {
			t.Fatalf("apiKey = %q, 期望回退到 QWEN_API_KEY=qwen-key", p.apiKey)
		}
	})
	t.Run("显式 apiKey 不被覆盖", func(t *testing.T) {
		t.Setenv("DASHSCOPE_API_KEY", "env-key")
		p := New("explicit")
		if p.apiKey != "explicit" {
			t.Fatalf("apiKey = %q, 期望显式传入的 explicit", p.apiKey)
		}
	})
}

// TestModels 验证 Models 返回 6 个模型且关键字段非空。
func TestModels(t *testing.T) {
	models := New("k").Models()
	if len(models) != 6 {
		t.Fatalf("Models 数量 = %d, 期望 6", len(models))
	}
	ids := map[string]bool{}
	for _, m := range models {
		if m.ID == "" || m.Name == "" {
			t.Errorf("模型字段不应为空: %+v", m)
		}
		if m.MaxTokens <= 0 {
			t.Errorf("模型 %s MaxTokens <= 0: %d", m.ID, m.MaxTokens)
		}
		if ids[m.ID] {
			t.Errorf("模型 ID 重复: %s", m.ID)
		}
		ids[m.ID] = true
	}
	if !ids["qwen-max"] {
		t.Error("缺少默认模型 qwen-max")
	}
}

// TestCountTokens 验证 CountTokens 的简化估算公式 len(content)*2/5。
func TestCountTokens(t *testing.T) {
	tests := []struct {
		name string
		msgs []llm.Message
		want int
	}{
		{"空列表", nil, 0},
		{"空内容", []llm.Message{{Role: llm.RoleUser, Content: ""}}, 0},
		// "hello" 长度 5 -> 5*2/5 = 2
		{"英文 hello", []llm.Message{{Role: llm.RoleUser, Content: "hello"}}, 5 * 2 / 5},
		// 多条消息累加
		{"多条", []llm.Message{
			{Role: llm.RoleUser, Content: "hello"},
			{Role: llm.RoleAssistant, Content: "world!"},
		}, 5*2/5 + 6*2/5},
		// 中文：len 计 UTF-8 字节数，"你好" = 6 字节 -> 6*2/5 = 2
		{"中文你好", []llm.Message{{Role: llm.RoleUser, Content: "你好"}}, len("你好") * 2 / 5},
	}
	p := New("k")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := p.CountTokens(tt.msgs)
			if err != nil {
				t.Fatalf("CountTokens 返回错误: %v", err)
			}
			if got != tt.want {
				t.Fatalf("CountTokens = %d, 期望 %d", got, tt.want)
			}
		})
	}
}

// TestEmbeddingDimension 验证模型到维度的映射表。
func TestEmbeddingDimension(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"text-embedding-v3", 1024},
		{"text-embedding-v2", 1536},
		{"text-embedding-v1", 1536},
		{"unknown-model", 1024},
		{"", 1024},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := EmbeddingDimension(tt.model); got != tt.want {
				t.Fatalf("EmbeddingDimension(%q) = %d, 期望 %d", tt.model, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Complete：请求构造 + 响应解析
// ---------------------------------------------------------------------------

// TestComplete_RequestConstruction 验证各 Option 是否正确写入请求体。
func TestComplete_RequestConstruction(t *testing.T) {
	var captured map[string]any
	var capturedAuth, capturedCT, capturedPath string

	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		capturedCT = r.Header.Get("Content-Type")
		capturedPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"qwen-max","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`))
	})

	req := llm.CompletionRequest{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		MaxTokens:   128,
		Temperature: floatPtr(0.7),
		TopP:        floatPtr(0.9),
		Stop:        []string{"END"},
		Tools: []llm.ToolDefinition{
			llm.NewToolDefinition("get_weather", "查询天气", nil),
		},
		ToolChoice:     "auto",
		Metadata:       map[string]any{"enable_search": true},
		ResponseFormat: &llm.ResponseFormat{Type: "json_object"},
	}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}

	// 路径与 header
	if capturedPath != "/chat/completions" {
		t.Errorf("请求路径 = %q, 期望 /chat/completions", capturedPath)
	}
	if capturedAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, 期望 'Bearer test-key'", capturedAuth)
	}
	if capturedCT != "application/json" {
		t.Errorf("Content-Type = %q, 期望 application/json", capturedCT)
	}

	// 请求体字段
	if captured["model"] != "qwen-max" {
		t.Errorf("model = %v, 期望默认 qwen-max", captured["model"])
	}
	if captured["stream"] != false {
		t.Errorf("stream = %v, 期望 false", captured["stream"])
	}
	if captured["max_tokens"] != float64(128) {
		t.Errorf("max_tokens = %v, 期望 128", captured["max_tokens"])
	}
	if captured["temperature"] != 0.7 {
		t.Errorf("temperature = %v, 期望 0.7", captured["temperature"])
	}
	if captured["top_p"] != 0.9 {
		t.Errorf("top_p = %v, 期望 0.9", captured["top_p"])
	}
	if _, ok := captured["tools"]; !ok {
		t.Error("缺少 tools 字段")
	}
	if captured["tool_choice"] != "auto" {
		t.Errorf("tool_choice = %v, 期望 auto", captured["tool_choice"])
	}
	if captured["enable_search"] != true {
		t.Errorf("enable_search = %v, 期望 true", captured["enable_search"])
	}
	rf, ok := captured["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_object" {
		t.Errorf("response_format = %v, 期望 {type:json_object}", captured["response_format"])
	}
	if stop, ok := captured["stop"].([]any); !ok || len(stop) != 1 || stop[0] != "END" {
		t.Errorf("stop = %v, 期望 [END]", captured["stop"])
	}
}

// TestComplete_OmitOptionalFields 验证未设置的可选字段不会出现在请求体里。
// 重点：enable_search=false 时不应写入 enable_search 键；ResponseFormat.Type
// 非 json_object 时不应写入 response_format。
func TestComplete_OmitOptionalFields(t *testing.T) {
	var captured map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &captured)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	})

	req := llm.CompletionRequest{
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Metadata:       map[string]any{"enable_search": false},
		ResponseFormat: &llm.ResponseFormat{Type: "text"},
	}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}

	for _, k := range []string{"max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice", "enable_search", "response_format"} {
		if _, ok := captured[k]; ok {
			t.Errorf("未设置的字段 %q 不应出现在请求体: %v", k, captured[k])
		}
	}
}

// TestComplete_ModelOverride 验证 req.Model 非空时不被默认模型覆盖。
func TestComplete_ModelOverride(t *testing.T) {
	var captured map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &captured)
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	})
	req := llm.CompletionRequest{
		Model:    "qwen-turbo",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if captured["model"] != "qwen-turbo" {
		t.Fatalf("model = %v, 期望显式 qwen-turbo", captured["model"])
	}
}

// TestComplete_ResponseParsing 验证响应 JSON 解析，包含内容、Usage、tool_calls。
func TestComplete_ResponseParsing(t *testing.T) {
	body := `{
		"id":"resp-1",
		"model":"qwen-max",
		"created":1700000000,
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"你好，世界",
				"tool_calls":[
					{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"北京\"}"}}
				]
			},
			"finish_reason":"tool_calls"
		}],
		"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
	}`
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})

	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if resp.ID != "resp-1" {
		t.Errorf("ID = %q, 期望 resp-1", resp.ID)
	}
	if resp.Model != "qwen-max" {
		t.Errorf("Model = %q, 期望 qwen-max", resp.Model)
	}
	if resp.Created != 1700000000 {
		t.Errorf("Created = %d, 期望 1700000000", resp.Created)
	}
	if resp.Content != "你好，世界" {
		t.Errorf("Content = %q, 期望 '你好，世界'", resp.Content)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, 期望 tool_calls", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 20 || resp.Usage.TotalTokens != 30 {
		t.Errorf("Usage = %+v, 期望 10/20/30", resp.Usage)
	}
	if !resp.HasToolCalls() || len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls 数量 = %d, 期望 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_1" || tc.Type != "function" || tc.Name != "get_weather" || tc.Arguments != `{"city":"北京"}` {
		t.Fatalf("ToolCall 解析错误: %+v", tc)
	}
}

// TestComplete_MultiChoiceFirstOnly 验证多 choice 时只取第一个（实现固定取 [0]）。
func TestComplete_MultiChoiceFirstOnly(t *testing.T) {
	body := `{"id":"x","choices":[
		{"index":0,"message":{"content":"first"},"finish_reason":"stop"},
		{"index":1,"message":{"content":"second"},"finish_reason":"stop"}
	]}`
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if resp.Content != "first" {
		t.Fatalf("Content = %q, 期望只取首个 choice 的 first", resp.Content)
	}
}

// TestComplete_EmptyChoices 验证 choices 为空时不 panic，返回空内容。
func TestComplete_EmptyChoices(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"x","choices":[],"usage":{"total_tokens":5}}`))
	})
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if resp.Content != "" {
		t.Errorf("Content = %q, 期望空", resp.Content)
	}
	if resp.Usage.TotalTokens != 5 {
		t.Errorf("Usage.TotalTokens = %d, 期望 5（即使 choices 为空 usage 仍应解析）", resp.Usage.TotalTokens)
	}
}

// TestComplete_Non200 验证非 200 响应返回携带状态码与 body 的错误。
func TestComplete_Non200(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"invalid_request"}`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("非 200 响应应返回错误，实际为 nil")
	}
	if !strings.Contains(err.Error(), "invalid_request") {
		t.Errorf("错误应包含响应 body, 实际: %v", err)
	}
	if !strings.Contains(err.Error(), "qwen api error") {
		t.Errorf("错误应包含 'qwen api error', 实际: %v", err)
	}
}

// TestComplete_BadJSON 验证响应体非法 JSON 时返回解析错误。
func TestComplete_BadJSON(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not-a-json{{{`))
	})
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("非法 JSON 响应应返回解析错误，实际为 nil")
	}
}

// TestComplete_CancelledContext 验证已取消的 context 会让请求短路失败。
func TestComplete_CancelledContext(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.Complete(ctx, llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("已取消的 context 应导致 Complete 返回错误")
	}
}

// ---------------------------------------------------------------------------
// Stream：SSE 解析
// ---------------------------------------------------------------------------

// TestStream_SSEParsing 验证流式 SSE 的 OpenAI 兼容解析，拼接增量内容与结束原因。
func TestStream_SSEParsing(t *testing.T) {
	sse := "data: {\"id\":\"s1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"你\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"好\"}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"世界\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"

	var capturedStream any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		capturedStream = body["stream"]
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	})

	stream, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream 返回错误: %v", err)
	}
	defer stream.Close()

	// 流式请求体的 stream 字段应为 true
	if capturedStream != true {
		t.Errorf("流式请求体 stream = %v, 期望 true", capturedStream)
	}

	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect 返回错误: %v", err)
	}
	if result.Content != "你好世界" {
		t.Errorf("拼接内容 = %q, 期望 '你好世界'", result.Content)
	}
	if result.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, 期望 stop", result.FinishReason)
	}
}

// TestStream_Non200 验证流式接口非 200 时返回错误且不返回 stream。
func TestStream_Non200(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate_limited"}`))
	})
	stream, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("非 200 流式响应应返回错误")
	}
	if stream != nil {
		t.Error("出错时不应返回非 nil 的 stream")
	}
	if !strings.Contains(err.Error(), "rate_limited") {
		t.Errorf("错误应包含 body, 实际: %v", err)
	}
}

// TestStream_ToolCalls 验证流式工具调用 delta 的合并解析。
func TestStream_ToolCalls(t *testing.T) {
	sse := "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n\n" +
		"data: [DONE]\n\n"
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	})
	stream, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream 返回错误: %v", err)
	}
	defer stream.Close()
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect 返回错误: %v", err)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("ToolCalls 数量 = %d, 期望合并为 1", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Arguments != `{"a":1}` {
		t.Errorf("ToolCall arguments = %q, 期望合并为 {\"a\":1}", result.ToolCalls[0].Arguments)
	}
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

// TestEmbed_EmptyTexts 验证空 texts 直接返回 (nil, nil)，不发起 HTTP 请求。
func TestEmbed_EmptyTexts(t *testing.T) {
	called := false
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		called = true
	})
	got, err := p.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(nil) 返回错误: %v", err)
	}
	if got != nil {
		t.Errorf("Embed(nil) = %v, 期望 nil", got)
	}
	got2, err := p.Embed(context.Background(), []string{})
	if err != nil || got2 != nil {
		t.Errorf("Embed([]) = (%v, %v), 期望 (nil, nil)", got2, err)
	}
	if called {
		t.Error("空 texts 不应发起 HTTP 请求")
	}
}

// TestEmbed_DefaultModel 验证 Embed 使用默认模型 text-embedding-v3。
func TestEmbed_DefaultModel(t *testing.T) {
	var captured map[string]any
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("embedding 路径 = %q, 期望 /embeddings", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &captured)
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2]}]}`))
	})
	_, err := p.Embed(context.Background(), []string{"hi"})
	if err != nil {
		t.Fatalf("Embed 返回错误: %v", err)
	}
	if captured["model"] != defaultEmbeddingModel {
		t.Fatalf("embedding model = %v, 期望 %q", captured["model"], defaultEmbeddingModel)
	}
}

// TestEmbedWithModel_Sorting 验证响应乱序时按 index 排序，结果与输入顺序一一对应。
func TestEmbedWithModel_Sorting(t *testing.T) {
	// 故意乱序返回 index 2,0,1
	body := `{"data":[
		{"index":2,"embedding":[2.0]},
		{"index":0,"embedding":[0.0]},
		{"index":1,"embedding":[1.0]}
	]}`
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(body))
	})
	got, err := p.EmbedWithModel(context.Background(), "text-embedding-v2", []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("EmbedWithModel 返回错误: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("结果数量 = %d, 期望 3", len(got))
	}
	for i, vec := range got {
		if len(vec) != 1 || vec[0] != float32(i) {
			t.Errorf("第 %d 个向量 = %v, 期望按 index 排序后为 [%d]", i, vec, i)
		}
	}
}

// TestEmbed_Non200 验证 embedding 非 200 返回错误。
func TestEmbed_Non200(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`unauthorized`))
	})
	_, err := p.Embed(context.Background(), []string{"hi"})
	if err == nil {
		t.Fatal("embedding 非 200 应返回错误")
	}
	if !strings.Contains(err.Error(), "qwen embedding api error") {
		t.Errorf("错误应包含 'qwen embedding api error', 实际: %v", err)
	}
}

// TestEmbed_BadJSON 验证 embedding 响应非法 JSON 返回错误。
func TestEmbed_BadJSON(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{bad`))
	})
	_, err := p.Embed(context.Background(), []string{"hi"})
	if err == nil {
		t.Fatal("embedding 非法 JSON 应返回错误")
	}
}

// TestEmbed_CountMismatch 验证返回向量数与输入 texts 数不一致时返回错误，
// 防止调用方在向量与文本一一对应的假设下静默错位。
// 回归: W3-56
func TestEmbed_CountMismatch(t *testing.T) {
	// 输入 3 个文本但服务端只返回 2 个向量，破坏一一对应不变量
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[
			{"index":0,"embedding":[0.0]},
			{"index":1,"embedding":[1.0]}
		]}`))
	})
	got, err := p.EmbedWithModel(context.Background(), "text-embedding-v2", []string{"a", "b", "c"})
	if err == nil {
		t.Fatalf("返回向量数(2)与输入 texts 数(3)不一致应返回错误, 实际 got=%v err=nil", got)
	}
	if got != nil {
		t.Errorf("不一致时应返回 nil 向量, 实际 got=%v", got)
	}
}

// ---------------------------------------------------------------------------
// Option 生效 / 接口实现 / 并发
// ---------------------------------------------------------------------------

// TestWithModel_TakesEffect 验证 WithModel 真正修改默认模型并透传到请求体。
func TestWithModel_TakesEffect(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &captured)
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()), WithModel("qwen-plus"))
	if p.model != "qwen-plus" {
		t.Fatalf("WithModel 后 p.model = %q, 期望 qwen-plus", p.model)
	}
	if _, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if captured["model"] != "qwen-plus" {
		t.Fatalf("请求体 model = %v, 期望 WithModel 生效后的 qwen-plus", captured["model"])
	}
}

// TestProviderImplementsInterfaces 验证 *Provider 实现 llm.Provider 与 llm.EmbeddingProvider。
func TestProviderImplementsInterfaces(t *testing.T) {
	var _ llm.Provider = (*Provider)(nil)
	var _ llm.EmbeddingProvider = (*Provider)(nil)
}

// TestConcurrency 验证 Provider 在多协程下复用安全（Models/CountTokens/Complete）。
func TestConcurrency(t *testing.T) {
	p, _ := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}],"usage":{"total_tokens":1}}`))
	})
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = p.Models()
			_, _ = p.CountTokens([]llm.Message{{Role: llm.RoleUser, Content: "hello"}})
			_, err := p.Complete(context.Background(), llm.CompletionRequest{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Errorf("并发 Complete 出错: %v", err)
			}
		}()
	}
	wg.Wait()
}

// readModelViaReflect 通过反射读取私有 model 字段，确认 Option 真正修改了内部状态。
// 仅用于断言，不修改源码。
func readModelViaReflect(t *testing.T, p *Provider) string {
	t.Helper()
	v := reflect.ValueOf(p).Elem().FieldByName("model")
	if !v.IsValid() {
		t.Fatal("无法定位 model 字段")
	}
	return v.String()
}

// TestWithBaseURL_TakesEffect 验证 WithBaseURL 真正改变请求目标。
func TestWithBaseURL_TakesEffect(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if p.baseURL != srv.URL {
		t.Fatalf("baseURL = %q, 期望 %q", p.baseURL, srv.URL)
	}
	// 同时用反射兜底确认 model 默认值未被 WithBaseURL 误改
	if got := readModelViaReflect(t, p); got != defaultModel {
		t.Errorf("WithBaseURL 不应影响 model, 实际 = %q", got)
	}
	if _, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if !hit {
		t.Fatal("WithBaseURL 未生效，测试服务器未被命中")
	}
}

func TestWithNetworkPolicyBlocksPrivateBaseURL(t *testing.T) {
	p := New("k",
		WithBaseURL("http://127.0.0.1:1"),
		WithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true}),
	)
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage("hi")},
	})
	if err == nil {
		t.Fatal("NetworkPolicy should block private baseURL")
	}
	if !errors.Is(err, llm.ErrNetworkPolicy) {
		t.Fatalf("expected ErrNetworkPolicy, got %v", err)
	}
}
