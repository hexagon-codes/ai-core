package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// 本文件把 openai provider 的全场景单测补齐：配置/请求构造/角色转换/
// HTTP 成功与错误路径/流式/响应解析/Embedding。配合 openai_cache_test.go
// 与各 bug_*_test.go，对标 anthropic_coverage_test.go 的范式。

// ============== 配置与基础 ==============

func TestNew_DefaultsAndOptions(t *testing.T) {
	p := New("k")
	if p.apiKey != "k" || p.baseURL != defaultBaseURL || p.model != defaultModel {
		t.Fatalf("默认值错误: %+v", p)
	}
	if p.httpClient == nil {
		t.Fatal("httpClient 应非 nil")
	}
	custom := &http.Client{}
	p2 := New("k2", WithBaseURL("http://x"), WithModel("m"), WithHTTPClient(custom))
	if p2.baseURL != "http://x" || p2.model != "m" || p2.httpClient != custom {
		t.Fatalf("选项未生效: %+v", p2)
	}
}

func TestNew_EnvAPIKeyFallback(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "from-env")
	if got := New("").apiKey; got != "from-env" {
		t.Fatalf("空 key 应回退环境变量, got %q", got)
	}
}

func TestName(t *testing.T) {
	if New("k").Name() != "openai" {
		t.Fatal("Name 应为 openai")
	}
}

func TestModels(t *testing.T) {
	models := New("k").Models()
	if len(models) != 5 {
		t.Fatalf("应有 5 个模型, got %d", len(models))
	}
	if models[0].ID != "gpt-4o" || models[0].MaxTokens != 128000 {
		t.Fatalf("首个模型字段错误: %+v", models[0])
	}
	// o1 仅 streaming，校验 features 分支
	if !models[3].HasFeature(llm.FeatureStreaming) || models[3].HasFeature(llm.FeatureVision) {
		t.Fatalf("o1 features 错误: %+v", models[3])
	}
}

func TestCountTokens(t *testing.T) {
	n, err := New("k").CountTokens([]llm.Message{
		{Role: llm.RoleUser, Content: "12345678"},  // 8/4=2
		{Role: llm.RoleAssistant, Content: "1234"}, // 4/4=1
	})
	if err != nil || n != 3 {
		t.Fatalf("CountTokens = %d,%v want 3,nil", n, err)
	}
}

func TestSetHeaders(t *testing.T) {
	req, _ := http.NewRequest("POST", "http://x", nil)
	New("secret").setHeaders(req)
	if req.Header.Get("Authorization") != "Bearer secret" ||
		req.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("headers 错误: %+v", req.Header)
	}
}

// ============== buildRequestBody 全分支 ==============

func TestBuildRequestBody_AllBranches(t *testing.T) {
	p := New("k")
	temp := 0.5
	topP := 0.9
	req := llm.CompletionRequest{
		Model: "gpt-x",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "you are helpful"},
			{Role: llm.RoleUser, Content: "hi"},
		},
		MaxTokens:   256,
		Temperature: &temp,
		TopP:        &topP,
		Stop:        []string{"END"},
		User:        "user-42",
		ToolChoice:  "auto",
		Tools:       []llm.ToolDefinition{llm.NewToolDefinition("get_weather", "查天气", nil)},
	}
	body, err := p.buildRequestBody(req, true)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["model"] != "gpt-x" {
		t.Errorf("model 错误: %v", payload["model"])
	}
	if payload["stream"] != true {
		t.Errorf("stream 标志错误")
	}
	if payload["max_tokens"].(float64) != 256 {
		t.Errorf("max_tokens 覆盖失败: %v", payload["max_tokens"])
	}
	if payload["temperature"].(float64) != 0.5 || payload["top_p"].(float64) != 0.9 {
		t.Errorf("temperature/top_p 未写入")
	}
	if payload["user"] != "user-42" {
		t.Errorf("user 未写入: %v", payload["user"])
	}
	if payload["tool_choice"] != "auto" {
		t.Errorf("tool_choice 未写入: %v", payload["tool_choice"])
	}
	if _, ok := payload["stop"]; !ok {
		t.Errorf("stop 未写入")
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Errorf("tools 未写入: %v", payload["tools"])
	}
	msgs := payload["messages"].([]any)
	if len(msgs) != 2 { // system + user 都保留（OpenAI 不分离 system）
		t.Errorf("messages 应含 2 条, got %d", len(msgs))
	}
}

func TestBuildRequestBody_Minimal(t *testing.T) {
	p := New("k")
	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	if payload["stream"] != false {
		t.Errorf("非流式 stream 应为 false")
	}
	// 未设置的可选字段不应出现
	for _, k := range []string{"max_tokens", "temperature", "top_p", "stop", "user", "tools", "tool_choice", "response_format"} {
		if _, ok := payload[k]; ok {
			t.Errorf("未设置的字段 %q 不应出现", k)
		}
	}
}

func TestBuildRequestBody_ResponseFormatJSONObject(t *testing.T) {
	p := New("k")
	body, _ := p.buildRequestBody(llm.CompletionRequest{
		Model:          "m",
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: "x"}},
		ResponseFormat: &llm.ResponseFormat{Type: "json_object"},
	}, false)
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	rf, ok := payload["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_object" {
		t.Fatalf("json_object 未写入: %v", payload["response_format"])
	}
}

func TestBuildRequestBody_SiliconFlowEnableThinkingFromMetadata(t *testing.T) {
	p := New("k", WithBaseURL("https://api.siliconflow.cn/v1"))
	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "Qwen/Qwen3.6-35B-A3B",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Metadata: map[string]any{"thinking": "off"},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	got, ok := payload["enable_thinking"].(bool)
	if !ok {
		t.Fatalf("enable_thinking type = %T, want bool; payload=%v", payload["enable_thinking"], payload)
	}
	if got {
		t.Fatalf("enable_thinking = true, want false")
	}
}

func TestBuildRequestBody_DefaultOpenAIDoesNotSendEnableThinking(t *testing.T) {
	p := New("k")
	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "gpt-4o",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Metadata: map[string]any{"thinking": "off"},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["enable_thinking"]; ok {
		t.Fatalf("default OpenAI payload must not include enable_thinking: %v", payload)
	}
}

func TestBuildRequestBody_OpenAIReasoningModelMapsThinkingModeToReasoningEffort(t *testing.T) {
	tests := []struct {
		name     string
		metadata map[string]any
		want     string
	}{
		{
			name:     "thinking on uses high effort",
			metadata: map[string]any{"thinking": "on"},
			want:     "high",
		},
		{
			name:     "thinking off disables reasoning when model supports none",
			metadata: map[string]any{"thinking": "off"},
			want:     "none",
		},
		{
			name:     "explicit effort takes precedence",
			metadata: map[string]any{"thinking": "off", "reasoning_effort": "max"},
			want:     "max",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// A loopback OpenAI-compatible gateway is a valid deployment and must
			// not silently lose the standard reasoning_effort parameter.
			p := New("k", WithBaseURL("http://localhost:18080/v1"))
			body, err := p.buildRequestBody(llm.CompletionRequest{
				Model:    "gpt-5.6-sol",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
				Metadata: tt.metadata,
			}, true)
			if err != nil {
				t.Fatal(err)
			}

			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatal(err)
			}
			if got := payload["reasoning_effort"]; got != tt.want {
				t.Fatalf("reasoning_effort = %#v, want %q; payload=%v", got, tt.want, payload)
			}
			if _, ok := payload["enable_thinking"]; ok {
				t.Fatalf("OpenAI reasoning model must use reasoning_effort, not vendor enable_thinking: %v", payload)
			}
		})
	}
}

func TestBuildRequestBody_NonReasoningOpenAIModelDoesNotInferReasoningEffort(t *testing.T) {
	p := New("k", WithBaseURL("http://localhost:18080/v1"))
	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "gpt-4o",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Metadata: map[string]any{"thinking": "on"},
	}, true)
	if err != nil {
		t.Fatal(err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["reasoning_effort"]; ok {
		t.Fatalf("non-reasoning model must not receive inferred reasoning_effort: %v", payload)
	}
}

func TestBuildRequestBody_ResponseFormatJSONSchema(t *testing.T) {
	p := New("k")
	body, _ := p.buildRequestBody(llm.CompletionRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}},
		ResponseFormat: &llm.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &llm.ResponseFormatJSONSchema{
				Name:        "result",
				Description: "结构化结果",
				Schema:      &llm.Schema{Type: "object"},
				Strict:      true,
			},
		},
	}, false)
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	rf, ok := payload["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_schema" {
		t.Fatalf("json_schema 未写入: %v", payload["response_format"])
	}
	js := rf["json_schema"].(map[string]any)
	if js["name"] != "result" || js["strict"] != true || js["description"] != "结构化结果" {
		t.Fatalf("json_schema 字段错误: %v", js)
	}
}

// TestBuildRequestBody_JSONSchemaNoDescription 覆盖 Description 为空的分支
// （不写入 description 字段）。
func TestBuildRequestBody_JSONSchemaNoDescription(t *testing.T) {
	p := New("k")
	body, _ := p.buildRequestBody(llm.CompletionRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}},
		ResponseFormat: &llm.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &llm.ResponseFormatJSONSchema{
				Name:   "r",
				Schema: &llm.Schema{Type: "object"},
			},
		},
	}, false)
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	js := payload["response_format"].(map[string]any)["json_schema"].(map[string]any)
	if _, ok := js["description"]; ok {
		t.Fatalf("Description 为空时不应写入 description: %v", js)
	}
}

// TestBuildRequestBody_JSONSchemaNilInner 覆盖 Type=json_schema 但
// JSONSchema 为 nil 的分支（不写入 response_format）。
func TestBuildRequestBody_JSONSchemaNilInner(t *testing.T) {
	p := New("k")
	body, _ := p.buildRequestBody(llm.CompletionRequest{
		Model:          "m",
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: "x"}},
		ResponseFormat: &llm.ResponseFormat{Type: "json_schema"},
	}, false)
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	if _, ok := payload["response_format"]; ok {
		t.Fatalf("JSONSchema 为 nil 时不应写入 response_format")
	}
}

// TestBuildRequestBody_ResponseFormatUnknownType 覆盖 ResponseFormat
// 非空但 Type 未命中任何 case 的分支（不写入 response_format）。
func TestBuildRequestBody_ResponseFormatUnknownType(t *testing.T) {
	p := New("k")
	body, _ := p.buildRequestBody(llm.CompletionRequest{
		Model:          "m",
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: "x"}},
		ResponseFormat: &llm.ResponseFormat{Type: "text"},
	}, false)
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	if _, ok := payload["response_format"]; ok {
		t.Fatalf("未知 Type 不应写入 response_format")
	}
}

// ============== ConvertMessages 多模态/角色分支 ==============

func TestConvertMessages_MultiContentAndRoles(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "sys"},
		{
			Role: llm.RoleUser,
			MultiContent: []llm.ContentPart{
				llm.NewTextPart("看图"),
				llm.NewImageURLPart("https://x/img.png", "high"),
				{Type: "image_url"}, // ImageURL 为 nil，覆盖 nil 分支
			},
		},
		{
			Role:    llm.RoleAssistant,
			Content: "ignored when tool calls present",
			ToolCalls: []llm.ToolCallRef{
				{ID: "call_1", Name: "get_weather", Arguments: `{"city":"SF"}`},
			},
		},
		{Role: llm.RoleTool, Content: "晴", ToolCallID: "call_1", Name: "get_weather"},
	}
	out := ConvertMessages(msgs)
	if len(out) != 4 {
		t.Fatalf("应有 4 条消息, got %d", len(out))
	}

	// 多模态 content 数组
	parts, ok := out[1]["content"].([]map[string]any)
	if !ok || len(parts) != 3 {
		t.Fatalf("多模态 content 错误: %v", out[1]["content"])
	}
	if parts[0]["type"] != "text" || parts[0]["text"] != "看图" {
		t.Errorf("text part 错误: %v", parts[0])
	}
	if parts[1]["type"] != "image_url" {
		t.Errorf("image part 错误: %v", parts[1])
	}
	img := parts[1]["image_url"].(map[string]any)
	if img["url"] != "https://x/img.png" || img["detail"] != "high" {
		t.Errorf("image_url 字段错误: %v", img)
	}
	// 第三个 part：ImageURL 为 nil，不应有 image_url 键
	if _, has := parts[2]["image_url"]; has {
		t.Errorf("nil ImageURL 不应写入 image_url 键: %v", parts[2])
	}

	// assistant tool_calls：content 强制 nil
	if out[2]["content"] != nil {
		t.Errorf("assistant+tool_calls 时 content 应为 nil, got %v", out[2]["content"])
	}
	calls := out[2]["tool_calls"].([]map[string]any)
	if len(calls) != 1 || calls[0]["id"] != "call_1" {
		t.Errorf("tool_calls 错误: %v", calls)
	}
	fn := calls[0]["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function.name 错误: %v", fn)
	}

	// tool message：带 tool_call_id 与 name
	if out[3]["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id 未写入: %v", out[3])
	}
	if out[3]["name"] != "get_weather" {
		t.Errorf("name 未写入: %v", out[3])
	}
}

// TestConvertMessages_ImageURLNoDetail 覆盖 ImageURL 非 nil 但 Detail 为空
// 的分支（不写入 detail 键）。
func TestConvertMessages_ImageURLNoDetail(t *testing.T) {
	out := ConvertMessages([]llm.Message{
		{
			Role: llm.RoleUser,
			MultiContent: []llm.ContentPart{
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "u"}},
			},
		},
	})
	parts := out[0]["content"].([]map[string]any)
	img := parts[0]["image_url"].(map[string]any)
	if _, ok := img["detail"]; ok {
		t.Fatalf("Detail 为空时不应写入 detail: %v", img)
	}
}

// TestConvertMessages_DefaultPartType 覆盖 MultiContent part Type 既非 text
// 也非 image_url（走 default 当作 text）的分支。
func TestConvertMessages_DefaultPartType(t *testing.T) {
	out := ConvertMessages([]llm.Message{
		{
			Role:         llm.RoleUser,
			MultiContent: []llm.ContentPart{{Type: "weird", Text: "fallback"}},
		},
	})
	parts := out[0]["content"].([]map[string]any)
	if parts[0]["type"] != "text" || parts[0]["text"] != "fallback" {
		t.Fatalf("未知 part type 应当作 text: %v", parts[0])
	}
}

// ============== Complete HTTP 路径 ==============

func TestComplete_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("缺 api key")
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("路径错误: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"id":"id_1","model":"gpt-x","created":123,
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},
			"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL))
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello" || resp.Usage.TotalTokens != 15 ||
		resp.FinishReason != "stop" || resp.Model != "gpt-x" || resp.Created != 123 {
		t.Fatalf("响应解析错误: %+v", resp)
	}
}

// TestComplete_DefaultModelInjected 验证 req.Model 为空时回退到 provider 默认模型。
func TestComplete_DefaultModelInjected(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		gotModel, _ = payload["model"].(string)
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL), WithModel("custom-default"))
	if _, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatal(err)
	}
	if gotModel != "custom-default" {
		t.Fatalf("空 Model 应回退默认模型, got %q", gotModel)
	}
}

func TestComplete_ToolCallsResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"x","choices":[{"message":{
			"role":"assistant","content":"",
			"tool_calls":[{"id":"t1","type":"function","function":{
			"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},
			"finish_reason":"tool_calls"}]}`))
	}))
	defer srv.Close()
	resp, err := New("k", WithBaseURL(srv.URL)).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "weather?"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.HasToolCalls() || len(resp.ToolCalls) != 1 {
		t.Fatalf("应有 1 个 tool call: %+v", resp.ToolCalls)
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "t1" || tc.Type != "function" || tc.Name != "get_weather" ||
		!strings.Contains(tc.Arguments, "SF") {
		t.Fatalf("tool_calls 解析错误: %+v", tc)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish_reason 错误: %q", resp.FinishReason)
	}
}

// TestComplete_EmptyChoices 覆盖 parseResponse 中 Choices 为空的分支。
func TestComplete_EmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[],
			"usage":{"prompt_tokens":3,"completion_tokens":0,"total_tokens":3}}`))
	}))
	defer srv.Close()
	resp, err := New("k", WithBaseURL(srv.URL)).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "" || resp.HasToolCalls() {
		t.Fatalf("空 choices 应返回空内容: %+v", resp)
	}
	if resp.Usage.PromptTokens != 3 {
		t.Errorf("usage 仍应解析: %+v", resp.Usage)
	}
}

func TestComplete_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limit"}`))
	}))
	defer srv.Close()
	_, err := New("k", WithBaseURL(srv.URL)).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "rate limit") {
		t.Fatalf("应返回含 body 的错误, got %v", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("应返回 ProviderError 且带状态码, got %#v", err)
	}
}

func TestComplete_WithHeadersAppliesOnlySafeOverrides(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer k" {
			t.Fatalf("Authorization = %q, want provider key", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Fatalf("Content-Type = %q, want provider content type", got)
		}
		if got := r.Header.Get("X-Debug-Trace"); got != "trace-1" {
			t.Fatalf("safe custom header missing, got %q", got)
		}
		_, _ = w.Write([]byte(`{"id":"id_1","model":"gpt-x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	p := New("k",
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{
			"Authorization": "Bearer attacker",
			"Content-Type":  "text/plain",
			"X-Debug-Trace": "trace-1",
		}),
	)
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "ok" {
		t.Fatalf("响应解析错误: %+v", resp)
	}
}

func TestComplete_WithNetworkPolicyBlocksPrivateBaseURL(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		_, _ = w.Write([]byte(`{"id":"id_1","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	_, err := New("k",
		WithBaseURL(srv.URL),
		WithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true}),
	).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected network policy error")
	}
	if called {
		t.Fatal("request should be blocked before reaching local server")
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || !strings.Contains(providerErr.Cause.Error(), "private/internal") {
		t.Fatalf("expected private host ProviderError, got %T %[1]v", err)
	}
}

func TestComplete_WithNetworkPolicyReusesGuardedTransport(t *testing.T) {
	var dials atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		_, _ = w.Write([]byte(`{"id":"id_1","model":"gpt-x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				dials.Add(1)
				return (&net.Dialer{}).DialContext(ctx, network, srv.Listener.Addr().String())
			},
		},
	}

	p := New("k",
		WithBaseURL("http://example.com/v1"),
		WithHTTPClient(client),
		WithNetworkPolicy(llm.NetworkPolicy{
			AllowHTTP:    true,
			AllowPrivate: true,
		}),
	)
	for i := 0; i < 2; i++ {
		resp, err := p.Complete(context.Background(), llm.CompletionRequest{
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Content != "ok" {
			t.Fatalf("响应解析错误: %+v", resp)
		}
	}
	if got := dials.Load(); got != 1 {
		t.Fatalf("guarded transport should preserve keep-alive reuse, got %d dials", got)
	}
}

func TestComplete_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{not json`))
	}))
	defer srv.Close()
	_, err := New("k", WithBaseURL(srv.URL)).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("畸形 JSON 应返回解析错误")
	}
}

func TestComplete_RetriesUnexpectedEOFAfterHTTP200(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			_, _ = w.Write([]byte(`{"choices":[`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"ok","model":"m","choices":[{"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	resp, err := New("k", WithBaseURL(srv.URL)).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("截断响应应重试后成功，got err=%v", err)
	}
	if resp.Content != "recovered" {
		t.Fatalf("content = %q, want recovered", resp.Content)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func TestComplete_NonIdempotentUnexpectedEOFDoesNotReplay(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		calls.Add(1)
		_, _ = w.Write([]byte(`{"choices":[`))
	}))
	defer srv.Close()
	ctx := llm.WithOperationSafety(context.Background(), llm.OperationSafetyNonIdempotent)

	_, err := New("k", WithBaseURL(srv.URL)).Complete(ctx, llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("截断响应应原样返回解析错误")
	}
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("error = %v, want unexpected EOF", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("physical POST attempts = %d, want 1", calls.Load())
	}
}

func TestComplete_RequestBuildAndDoErrors(t *testing.T) {
	req := llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}}

	// 1) NewRequestWithContext 失败：baseURL 缺协议头 → URL 解析错误
	badURL := New("k", WithBaseURL("://invalid"))
	if _, err := badURL.Complete(context.Background(), req); err == nil {
		t.Error("Complete 非法 URL 应报错")
	}

	// 2) httpClient.Do 失败：指向不可达端口 → 连接拒绝
	down := New("k", WithBaseURL("http://127.0.0.1:1"))
	if _, err := down.Complete(context.Background(), req); err == nil {
		t.Error("Complete 不可达上游应报错")
	}
}

// ============== Stream HTTP 路径 ==============

func TestStream_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
			"data: [DONE]\n\n"))
	}))
	defer srv.Close()
	s, err := New("k", WithBaseURL(srv.URL)).Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil || s == nil {
		t.Fatalf("流式成功应返回非 nil stream: %v", err)
	}
}

// TestStream_DefaultModelInjected 验证流式请求 req.Model 为空时回退默认模型。
func TestStream_DefaultModelInjected(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		gotModel, _ = payload["model"].(string)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL), WithModel("stream-default"))
	if _, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatal(err)
	}
	if gotModel != "stream-default" {
		t.Fatalf("流式空 Model 应回退默认模型, got %q", gotModel)
	}
}

func TestStream_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer srv.Close()
	_, err := New("k", WithBaseURL(srv.URL)).Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("流式错误状态应返回错误, got %v", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.StatusCode != http.StatusInternalServerError {
		t.Fatalf("流式错误应返回 ProviderError 且带状态码, got %#v", err)
	}
}

func TestStream_RequestBuildAndDoErrors(t *testing.T) {
	req := llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}}

	badURL := New("k", WithBaseURL("://invalid"))
	if _, err := badURL.Stream(context.Background(), req); err == nil {
		t.Error("Stream 非法 URL 应报错")
	}

	down := New("k", WithBaseURL("http://127.0.0.1:1"))
	if _, err := down.Stream(context.Background(), req); err == nil {
		t.Error("Stream 不可达上游应报错")
	}
}

func TestStream_IdleTimeoutPropagatesProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
		time.Sleep(80 * time.Millisecond)
		_, _ = w.Write([]byte("data: late\n\n"))
	}))
	defer srv.Close()

	s, err := New("k", WithBaseURL(srv.URL), WithStreamIdleTimeout(10*time.Millisecond)).
		Stream(context.Background(), llm.CompletionRequest{
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		})
	if err != nil {
		t.Fatal(err)
	}

	_, err = s.Collect()
	if err == nil {
		t.Fatal("expected stream idle timeout")
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Action != "stream.stream_read" {
		t.Fatalf("expected stream read ProviderError, got %T %[1]v", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded cause, got %v", err)
	}
}

// ============== Embedding ==============

func TestEmbeddingDimension(t *testing.T) {
	cases := map[string]int{
		"text-embedding-3-large": 3072,
		"text-embedding-3-small": 1536,
		"text-embedding-ada-002": 1536,
		"unknown-model":          1536,
	}
	for model, want := range cases {
		if got := EmbeddingDimension(model); got != want {
			t.Errorf("EmbeddingDimension(%q) = %d, want %d", model, got, want)
		}
	}
}

func TestEmbed_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("路径错误: %s", r.URL.Path)
		}
		var payload embeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if payload.Model != defaultEmbeddingModel {
			t.Errorf("Embed 应使用默认模型, got %q", payload.Model)
		}
		// 故意乱序返回，验证按 index 排序
		_, _ = w.Write([]byte(`{"object":"list","data":[
			{"object":"embedding","index":1,"embedding":[0.3,0.4]},
			{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
			"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":2,"total_tokens":2}}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL))
	vecs, err := p.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 {
		t.Fatalf("应返回 2 个向量, got %d", len(vecs))
	}
	// index 0 排在前面
	if vecs[0][0] != 0.1 || vecs[1][0] != 0.3 {
		t.Fatalf("乱序结果未按 index 排序: %v", vecs)
	}
}

func TestEmbedWithModel_CustomModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload embeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if payload.Model != "text-embedding-3-large" {
			t.Errorf("应使用指定模型, got %q", payload.Model)
		}
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[1.0]}]}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL))
	vecs, err := p.EmbedWithModel(context.Background(), "text-embedding-3-large", []string{"x"})
	if err != nil || len(vecs) != 1 || vecs[0][0] != 1.0 {
		t.Fatalf("EmbedWithModel 错误: %v %v", vecs, err)
	}
}

// TestEmbed_EmptyInput 覆盖空输入直接返回 nil,nil 的分支。
func TestEmbed_EmptyInput(t *testing.T) {
	vecs, err := New("k").Embed(context.Background(), nil)
	if err != nil || vecs != nil {
		t.Fatalf("空输入应返回 nil,nil, got %v %v", vecs, err)
	}
}

// TestEmbed_CtxCancelled 覆盖 ctx 已取消的提前返回分支。
func TestEmbed_CtxCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := New("k").Embed(ctx, []string{"x"})
	if err == nil {
		t.Fatal("已取消 ctx 应返回错误")
	}
}

func TestEmbed_ErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad embed"}`))
	}))
	defer srv.Close()
	_, err := New("k", WithBaseURL(srv.URL)).Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "bad embed") {
		t.Fatalf("embedding 错误状态应返回含 body 的错误, got %v", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("embedding 错误应返回 ProviderError 且带状态码, got %#v", err)
	}
}

func TestEmbed_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{not json`))
	}))
	defer srv.Close()
	_, err := New("k", WithBaseURL(srv.URL)).Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "解析 embedding 响应失败") {
		t.Fatalf("畸形 JSON 应返回解析错误, got %v", err)
	}
}

func TestEmbed_RetriesUnexpectedEOFAfterHTTP200(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			_, _ = w.Write([]byte(`{"data":[`))
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2]}]}`))
	}))
	defer srv.Close()

	vecs, err := New("k", WithBaseURL(srv.URL)).EmbedWithModel(context.Background(), "m", []string{"x"})
	if err != nil {
		t.Fatalf("截断 embedding 响应应重试后成功，got err=%v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 2 {
		t.Fatalf("embedding 解析错误: %#v", vecs)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

// ============== SanitizeToolCallSequence 边界 ==============

// TestSanitize_EmptyInput 覆盖空 slice 直接返回的分支。
func TestSanitize_EmptyInput(t *testing.T) {
	if got := SanitizeToolCallSequence(nil); got != nil {
		t.Fatalf("空输入应原样返回 nil, got %v", got)
	}
}

// TestSanitize_ToolMessageWithoutID 覆盖 RoleTool 但 ToolCallID 为空 → 丢弃
// 的分支（无法对账）。
func TestSanitize_ToolMessageWithoutID(t *testing.T) {
	out := SanitizeToolCallSequence([]llm.Message{
		{Role: llm.RoleUser, Content: "hi"},
		{Role: llm.RoleTool, Content: "无 ID 的孤立 tool"}, // ToolCallID 为空 → 丢
	})
	if len(out) != 1 || out[0].Role != llm.RoleUser {
		t.Fatalf("无 ID 的 tool message 应被丢弃, got %+v", out)
	}
}

func TestEmbed_RequestBuildAndDoErrors(t *testing.T) {
	// NewRequestWithContext 失败：非法 URL
	badURL := New("k", WithBaseURL("://invalid"))
	if _, err := badURL.Embed(context.Background(), []string{"x"}); err == nil {
		t.Error("Embed 非法 URL 应报错")
	}
	// httpClient.Do 失败：不可达上游
	down := New("k", WithBaseURL("http://127.0.0.1:1"))
	if _, err := down.Embed(context.Background(), []string{"x"}); err == nil {
		t.Error("Embed 不可达上游应报错")
	}
}
