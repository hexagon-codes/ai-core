// Package ark 的审计测试 (Wave3)。
//
// 本文件对豆包 (Ark) Provider 做全场景黑盒 + 受控白盒审计，覆盖：
//   - 正常路径：Name/Models/New 默认值/Complete/Stream
//   - 边界：空 apiKey + 环境变量、Unicode、超长内容、空消息、0/负数参数、并发竞态
//   - 错误路径：非 200 响应、坏 JSON、nil context、cancelled context
//   - 未闭环逻辑：WithModel/WithBaseURL/WithEndpointID/WithHTTPClient 是否真正生效
//   - 状态一致性：endpointID 覆盖 model、请求体字段省略规则
//
// ark 包暴露了 WithBaseURL/WithHTTPClient 注入点（New 中默认用 httpx.RawClient，
// 但 Option 可覆盖），因此可借助 httptest 在不打真实网络的前提下完整验证
// 请求构造 / 响应解析 (JSON) / 非 200 错误处理 / 流式 SSE 解析 / Option 生效。
package ark

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/hexagon-codes/ai-core/llm"
)

// ---------- 反射辅助：读取 Provider 私有字段 ----------
//
// ark.Provider 未暴露 model/apiKey/baseURL/endpointID 的读取入口，
// 为在不打网络的前提下观测 New 与各 Option 的真实效果，
// 这里通过 reflect + unsafe 读取私有字段。仅用于测试观测，不修改被测源码。

func readUnexportedString(t *testing.T, obj any, field string) string {
	t.Helper()
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	f := v.FieldByName(field)
	if !f.IsValid() {
		t.Fatalf("字段 %q 不存在于 %T", field, obj)
	}
	fv := reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem()
	return fv.String()
}

// ============== 正常路径 ==============

// TestName 验证 Name 返回固定的 "ark"。
func TestName(t *testing.T) {
	p := New("test-key")
	if got := p.Name(); got != "ark" {
		t.Fatalf("Name() = %q, 期望 %q", got, "ark")
	}
}

// TestNew_Defaults 验证 New 的默认 baseURL 与默认 model。
func TestNew_Defaults(t *testing.T) {
	p := New("test-key")
	if got := readUnexportedString(t, p, "baseURL"); got != defaultBaseURL {
		t.Errorf("默认 baseURL = %q, 期望 %q", got, defaultBaseURL)
	}
	if got := readUnexportedString(t, p, "model"); got != defaultModel {
		t.Errorf("默认 model = %q, 期望 %q", got, defaultModel)
	}
	if got := readUnexportedString(t, p, "apiKey"); got != "test-key" {
		t.Errorf("apiKey = %q, 期望 %q", got, "test-key")
	}
}

// TestModels 验证 Models 返回的模型列表内容与文档一致。
func TestModels(t *testing.T) {
	p := New("test-key")
	models := p.Models()

	// 与 Models() 源码列出的 7 个模型对齐
	if len(models) != 7 {
		t.Fatalf("Models() 返回 %d 个模型, 期望 7", len(models))
	}

	byID := make(map[string]llm.ModelInfo, len(models))
	for _, m := range models {
		if m.ID == "" {
			t.Errorf("存在 ID 为空的模型: %+v", m)
		}
		if m.Name == "" {
			t.Errorf("模型 %q Name 为空", m.ID)
		}
		if _, dup := byID[m.ID]; dup {
			t.Errorf("模型 ID %q 重复", m.ID)
		}
		byID[m.ID] = m
	}

	tests := []struct {
		id           string
		maxTokens    int
		wantFeatures []string
	}{
		{"doubao-pro-32k", 32768, []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming}},
		{"doubao-pro-128k", 131072, []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming}},
		{"doubao-pro-256k", 262144, []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming}},
		{"doubao-lite-32k", 32768, []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming}},
		{"doubao-lite-128k", 131072, []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming}},
		{"doubao-vision-pro-32k", 32768, []string{llm.FeatureVision, llm.FeatureStreaming}},
		{"doubao-character-pro-32k", 32768, []string{llm.FeatureStreaming}},
	}
	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			m, ok := byID[tt.id]
			if !ok {
				t.Fatalf("模型列表缺少 %q", tt.id)
			}
			if m.MaxTokens != tt.maxTokens {
				t.Errorf("%q MaxTokens = %d, 期望 %d", tt.id, m.MaxTokens, tt.maxTokens)
			}
			for _, f := range tt.wantFeatures {
				if !m.HasFeature(f) {
					t.Errorf("模型 %q 缺少特性 %q", tt.id, f)
				}
			}
		})
	}
}

// TestModels_Concurrent 验证 Models 在并发调用下无数据竞争（配合 -race）。
func TestModels_Concurrent(t *testing.T) {
	p := New("test-key")
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if len(p.Models()) != 7 {
				t.Errorf("并发 Models() 返回数量异常")
			}
			_ = p.Name()
		}()
	}
	wg.Wait()
}

// TestInterfaceCompliance 验证 *Provider 满足 llm.Provider 接口。
func TestInterfaceCompliance(t *testing.T) {
	var _ llm.Provider = New("k")
}

// ============== Option 生效 (未闭环逻辑) ==============

// TestWithModel_TakesEffect 验证 WithModel 真正修改默认模型。
func TestWithModel_TakesEffect(t *testing.T) {
	const want = "doubao-pro-128k"
	p := New("k", WithModel(want))
	if got := readUnexportedString(t, p, "model"); got != want {
		t.Fatalf("WithModel(%q) 后 model = %q, 期望 %q", want, got, want)
	}
}

// TestWithBaseURL_TakesEffect 验证 WithBaseURL 真正修改 baseURL。
func TestWithBaseURL_TakesEffect(t *testing.T) {
	const want = "https://gateway.internal/api/v3"
	p := New("k", WithBaseURL(want))
	if got := readUnexportedString(t, p, "baseURL"); got != want {
		t.Fatalf("WithBaseURL(%q) 后 baseURL = %q, 期望 %q", want, got, want)
	}
}

// TestWithEndpointID_TakesEffect 验证 WithEndpointID 真正写入 endpointID。
func TestWithEndpointID_TakesEffect(t *testing.T) {
	const want = "ep-20240101-abcde"
	p := New("k", WithEndpointID(want))
	if got := readUnexportedString(t, p, "endpointID"); got != want {
		t.Fatalf("WithEndpointID(%q) 后 endpointID = %q, 期望 %q", want, got, want)
	}
}

// ============== 环境变量读取 (边界) ==============

// TestNew_EmptyAPIKeyReadsArkEnv 验证空 apiKey 时从 ARK_API_KEY 读取。
func TestNew_EmptyAPIKeyReadsArkEnv(t *testing.T) {
	t.Setenv("ARK_API_KEY", "ark-env-key")
	t.Setenv("VOLC_ACCESSKEY", "")
	p := New("")
	if got := readUnexportedString(t, p, "apiKey"); got != "ark-env-key" {
		t.Fatalf("apiKey = %q, 期望从 ARK_API_KEY 读取 %q", got, "ark-env-key")
	}
}

// TestNew_EmptyAPIKeyFallsBackToVolc 验证 ARK_API_KEY 为空时回退到 VOLC_ACCESSKEY。
func TestNew_EmptyAPIKeyFallsBackToVolc(t *testing.T) {
	t.Setenv("ARK_API_KEY", "")
	t.Setenv("VOLC_ACCESSKEY", "volc-env-key")
	p := New("")
	if got := readUnexportedString(t, p, "apiKey"); got != "volc-env-key" {
		t.Fatalf("apiKey = %q, 期望回退到 VOLC_ACCESSKEY %q", got, "volc-env-key")
	}
}

// TestNew_ExplicitAPIKeyOverridesEnv 验证显式 apiKey 优先于环境变量。
func TestNew_ExplicitAPIKeyOverridesEnv(t *testing.T) {
	t.Setenv("ARK_API_KEY", "ark-env-key")
	p := New("explicit")
	if got := readUnexportedString(t, p, "apiKey"); got != "explicit" {
		t.Fatalf("apiKey = %q, 期望 %q", got, "explicit")
	}
}

// TestNew_EndpointIDFromEnv 验证 ARK_ENDPOINT_ID 环境变量被读取。
func TestNew_EndpointIDFromEnv(t *testing.T) {
	t.Setenv("ARK_ENDPOINT_ID", "ep-from-env")
	p := New("k")
	if got := readUnexportedString(t, p, "endpointID"); got != "ep-from-env" {
		t.Fatalf("endpointID = %q, 期望从环境变量读取 %q", got, "ep-from-env")
	}
}

// TestNew_OptionEndpointIDOverridesEnv 验证 WithEndpointID 覆盖环境变量。
//
// 源码中先读环境变量再 apply opts，因此 Option 应胜出。
func TestNew_OptionEndpointIDOverridesEnv(t *testing.T) {
	t.Setenv("ARK_ENDPOINT_ID", "ep-from-env")
	p := New("k", WithEndpointID("ep-from-opt"))
	if got := readUnexportedString(t, p, "endpointID"); got != "ep-from-opt" {
		t.Fatalf("endpointID = %q, 期望 Option 覆盖环境变量为 %q", got, "ep-from-opt")
	}
}

// ============== Complete: 请求构造 + 响应解析 (httptest) ==============

// TestComplete_RoundTrip 用 httptest 验证非流式请求构造与响应解析全链路。
func TestComplete_RoundTrip(t *testing.T) {
	var gotPath, gotAuth, gotContentType, gotMethod string
	var reqBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		gotMethod = r.Method
		_ = json.NewDecoder(r.Body).Decode(&reqBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "ark-resp-1",
			"model": "doubao-pro-32k",
			"created": 1700000000,
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "你好世界"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
		}`))
	}))
	defer srv.Close()

	p := New("secret", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))

	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}

	// 请求构造断言
	if gotMethod != "POST" {
		t.Errorf("请求 method = %q, 期望 POST", gotMethod)
	}
	if gotPath != "/chat/completions" {
		t.Errorf("请求 path = %q, 期望 /chat/completions", gotPath)
	}
	if gotAuth != "Bearer secret" {
		t.Errorf("Authorization = %q, 期望 %q", gotAuth, "Bearer secret")
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, 期望 application/json", gotContentType)
	}
	if reqBody["model"] != "doubao-pro-32k" {
		t.Errorf("请求 model = %v, 期望默认 doubao-pro-32k", reqBody["model"])
	}
	if reqBody["stream"] != false {
		t.Errorf("请求 stream = %v, 期望 false", reqBody["stream"])
	}

	// 响应解析断言
	if resp.ID != "ark-resp-1" {
		t.Errorf("resp.ID = %q, 期望 ark-resp-1", resp.ID)
	}
	if resp.Model != "doubao-pro-32k" {
		t.Errorf("resp.Model = %q", resp.Model)
	}
	if resp.Created != 1700000000 {
		t.Errorf("resp.Created = %d, 期望 1700000000", resp.Created)
	}
	if resp.Content != "你好世界" {
		t.Errorf("resp.Content = %q, 期望 你好世界", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("resp.FinishReason = %q, 期望 stop", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 5 || resp.Usage.CompletionTokens != 4 || resp.Usage.TotalTokens != 9 {
		t.Errorf("resp.Usage = %+v, 期望 {5,4,9}", resp.Usage)
	}
}

// TestComplete_ToolCallsParsing 验证响应中的 tool_calls 被正确解析。
func TestComplete_ToolCallsParsing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "ark-tc",
			"model": "doubao-pro-32k",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "",
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}
				]}, "finish_reason": "tool_calls"}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage("天气")},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}
	if !resp.HasToolCalls() {
		t.Fatalf("期望有 tool_calls, 实际无")
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_1" || tc.Type != "function" || tc.Name != "get_weather" {
		t.Errorf("tool_call 解析错误: %+v", tc)
	}
	if tc.Arguments != `{"city":"北京"}` {
		t.Errorf("tool_call Arguments = %q", tc.Arguments)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, 期望 tool_calls", resp.FinishReason)
	}
}

// TestComplete_RequestFieldsOmission 验证可选字段的省略/包含规则。
//
// 钉死 buildRequestBody 的契约：未设置的可选字段不应出现在请求体；
// 设置后应正确出现。覆盖 max_tokens/temperature/top_p/stop/tools/tool_choice/response_format。
func TestComplete_RequestFieldsOmission(t *testing.T) {
	temp := 0.7
	topP := 0.9
	tests := []struct {
		name    string
		req     llm.CompletionRequest
		present []string // 应出现的字段
		absent  []string // 不应出现的字段
	}{
		{
			name:    "仅消息_全部可选字段省略",
			req:     llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}},
			present: []string{"model", "messages", "stream"},
			absent:  []string{"max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice", "response_format"},
		},
		{
			name: "全部可选字段设置",
			req: llm.CompletionRequest{
				Messages:    []llm.Message{llm.UserMessage("a")},
				MaxTokens:   100,
				Temperature: &temp,
				TopP:        &topP,
				Stop:        []string{"END"},
				Tools:       []llm.ToolDefinition{llm.NewToolDefinition("f", "d", nil)},
				ToolChoice:  "auto",
				ResponseFormat: &llm.ResponseFormat{Type: "json_object"},
			},
			present: []string{"max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice", "response_format"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var reqBody map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&reqBody)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
			}))
			defer srv.Close()

			p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
			if _, err := p.Complete(context.Background(), tt.req); err != nil {
				t.Fatalf("Complete 错误: %v", err)
			}
			for _, k := range tt.present {
				if _, ok := reqBody[k]; !ok {
					t.Errorf("请求体应包含字段 %q, 实际缺失: %v", k, reqBody)
				}
			}
			for _, k := range tt.absent {
				if _, ok := reqBody[k]; ok {
					t.Errorf("请求体不应包含字段 %q, 实际存在: %v", k, reqBody[k])
				}
			}
		})
	}
}

// TestComplete_ResponseFormatJSONSchemaIgnored 验证 json_schema 类型不被透传。
//
// buildRequestBody 仅在 Type == "json_object" 时设置 response_format，
// json_schema 类型应被忽略（豆包不支持）。这是状态一致性测试。
func TestComplete_ResponseFormatJSONSchemaIgnored(t *testing.T) {
	var reqBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages:       []llm.Message{llm.UserMessage("a")},
		ResponseFormat: &llm.ResponseFormat{Type: "json_schema"},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if _, ok := reqBody["response_format"]; ok {
		t.Errorf("json_schema 类型不应透传 response_format, 实际存在: %v", reqBody["response_format"])
	}
}

// TestComplete_EndpointIDOverridesModel 验证设置 endpointID 后请求体 model 使用 endpointID。
//
// buildRequestBody 中：若 endpointID 非空，则 model 字段用 endpointID 替换。
// 这是状态一致性测试，验证火山引擎端点路由。
func TestComplete_EndpointIDOverridesModel(t *testing.T) {
	var reqBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()), WithEndpointID("ep-123"), WithModel("doubao-pro-128k"))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage("a")},
		Model:    "doubao-lite-32k",
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if reqBody["model"] != "ep-123" {
		t.Errorf("请求 model = %v, 期望 endpointID 覆盖为 ep-123", reqBody["model"])
	}
}

// TestComplete_UnicodeAndLongContent 验证 Unicode 与超长内容能正确序列化与解析。
func TestComplete_UnicodeAndLongContent(t *testing.T) {
	longInput := strings.Repeat("日本語テスト🎉emoji混排", 5000) // 超长 Unicode
	var gotMessages []any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotMessages, _ = body["messages"].([]any)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"返回：` + "🎉" + `中文"}}]}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage(longInput)},
	})
	if err != nil {
		t.Fatalf("Complete 错误: %v", err)
	}
	if len(gotMessages) != 1 {
		t.Fatalf("服务端收到 %d 条消息, 期望 1", len(gotMessages))
	}
	m := gotMessages[0].(map[string]any)
	if m["content"] != longInput {
		t.Errorf("超长 Unicode 内容在传输中被损坏")
	}
	if !strings.Contains(resp.Content, "🎉") {
		t.Errorf("响应 Unicode 解析错误: %q", resp.Content)
	}
}

// TestComplete_EmptyMessages 验证空消息列表不会 panic，请求 messages 为 [] 而非 null。
//
// 边界测试：ConvertMessages 对空切片返回 len 0 的切片，JSON 序列化为 []。
func TestComplete_EmptyMessages(t *testing.T) {
	var rawBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":""}}]}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{Messages: nil})
	if err != nil {
		t.Fatalf("空消息 Complete 错误: %v", err)
	}
	if !strings.Contains(string(rawBody), `"messages":[]`) {
		t.Errorf("空消息请求体应含 \"messages\":[]，实际: %s", rawBody)
	}
}

// TestComplete_EmptyChoices 验证响应 choices 为空时不 panic，返回空 Content。
//
// 边界测试：parseResponse 中 len(resp.Choices) > 0 守卫，空 choices 应安全返回。
func TestComplete_EmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[],"usage":{"prompt_tokens":1}}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
	if err != nil {
		t.Fatalf("空 choices Complete 错误: %v", err)
	}
	if resp.Content != "" {
		t.Errorf("空 choices 应返回空 Content, 实际 %q", resp.Content)
	}
	if resp.Usage.PromptTokens != 1 {
		t.Errorf("Usage 仍应解析, PromptTokens = %d", resp.Usage.PromptTokens)
	}
}

// ============== 错误路径 ==============

// TestComplete_Non200Error 验证非 200 响应返回包含状态码与 body 的错误。
func TestComplete_Non200Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
	if err == nil {
		t.Fatal("非 200 响应应返回错误, 实际 nil")
	}
	if !strings.Contains(err.Error(), "ark api error") {
		t.Errorf("错误信息应含 'ark api error', 实际: %v", err)
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("错误信息应含响应 body, 实际: %v", err)
	}
}

// TestComplete_BadJSON 验证 200 但响应体非法 JSON 时返回解码错误。
func TestComplete_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{not valid json`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	_, err := p.Complete(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
	if err == nil {
		t.Fatal("非法 JSON 响应应返回错误, 实际 nil")
	}
}

// TestComplete_CancelledContext 验证已取消 context 下 Complete 返回错误且不 panic。
func TestComplete_CancelledContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.Complete(ctx, llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
	if err == nil {
		t.Fatal("已取消 context 下 Complete 应返回错误")
	}
}

// TestComplete_NilContext 验证 nil context 的行为。
//
// 边界测试：http.NewRequestWithContext(nil,...) 在 Go 中会 panic("nil context")。
// 本测试用 recover 捕获，记录 Provider 是否对 nil context 做防御。
func TestComplete_NilContext(t *testing.T) {
	p := New("k", WithBaseURL("http://127.0.0.1:0"))
	defer func() {
		// nil context 触发底层 panic 属于已知 Go 行为；这里仅确认不会静默错误地继续执行。
		_ = recover()
	}()
	//nolint:staticcheck // 故意传 nil context 测试边界
	_, _ = p.Complete(nil, llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
}

// ============== Stream: SSE 解析 (httptest) ==============

// TestStream_RoundTrip 用 httptest 验证流式 SSE 解析全链路。
func TestStream_RoundTrip(t *testing.T) {
	var reqBody map[string]any
	sse := "data: {\"id\":\"ark-s\",\"model\":\"doubao-pro-32k\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"id\":\"ark-s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你\"}}]}\n\n" +
		"data: {\"id\":\"ark-s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"好\"}}]}\n\n" +
		"data: {\"id\":\"ark-s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	stream, err := p.Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Stream 返回错误: %v", err)
	}
	defer stream.Close()

	// 验证请求体 stream=true
	if reqBody["stream"] != true {
		t.Errorf("流式请求 stream = %v, 期望 true", reqBody["stream"])
	}

	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect 错误: %v", err)
	}
	if result.Content != "你好" {
		t.Errorf("流式拼接 Content = %q, 期望 你好", result.Content)
	}
	if result.ID != "ark-s" {
		t.Errorf("流式 ID = %q, 期望 ark-s", result.ID)
	}
	if result.FinishReason != "stop" {
		t.Errorf("流式 FinishReason = %q, 期望 stop", result.FinishReason)
	}
}

// TestStream_Non200Error 验证流式请求非 200 时返回错误且不返回 stream。
func TestStream_Non200Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad model"}`))
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	stream, err := p.Stream(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
	if err == nil {
		t.Fatal("流式非 200 应返回错误, 实际 nil")
	}
	if stream != nil {
		t.Error("流式非 200 时不应返回 stream")
	}
	if !strings.Contains(err.Error(), "ark api error") || !strings.Contains(err.Error(), "bad model") {
		t.Errorf("错误信息不完整: %v", err)
	}
}

// TestStream_EmptyBody 验证流式响应体为空（无任何 data 行）时 Collect 返回空内容不阻塞。
func TestStream_EmptyBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// 不写任何内容，直接结束
	}))
	defer srv.Close()

	p := New("k", WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	stream, err := p.Stream(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserMessage("a")}})
	if err != nil {
		t.Fatalf("Stream 错误: %v", err)
	}
	defer stream.Close()
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("空流 Collect 错误: %v", err)
	}
	if result.Content != "" {
		t.Errorf("空流应返回空 Content, 实际 %q", result.Content)
	}
}

// ============== CountTokens (边界 + 算法挑刺) ==============

// TestCountTokens 验证 CountTokens 的估算公式 len(content)*2/5。
//
// 注意：len() 计算的是 UTF-8 字节数而非 rune 数。中文每字 3 字节，
// 因此"你好"= 6 字节 → 6*2/5 = 2 token。本测试钉死当前实现行为，
// 并暴露算法对中文/Unicode 的字节级估算特性（详见 findings）。
func TestCountTokens(t *testing.T) {
	p := New("k")
	tests := []struct {
		name string
		msgs []llm.Message
		want int
	}{
		{"nil列表", nil, 0},
		{"空内容", []llm.Message{llm.UserMessage("")}, 0},
		{"5字节英文", []llm.Message{llm.UserMessage("hello")}, 5 * 2 / 5}, // = 2
		{"中文你好", []llm.Message{llm.UserMessage("你好")}, len("你好") * 2 / 5},
		{"多条消息累加", []llm.Message{llm.UserMessage("abcde"), llm.UserMessage("fghij")}, (5 + 5) * 2 / 5},
		{"emoji", []llm.Message{llm.UserMessage("🎉")}, len("🎉") * 2 / 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := p.CountTokens(tt.msgs)
			if err != nil {
				t.Fatalf("CountTokens 错误: %v", err)
			}
			if got != tt.want {
				t.Errorf("CountTokens = %d, 期望 %d", got, tt.want)
			}
		})
	}
}

// TestCountTokens_IgnoresMultiContent 暴露 CountTokens 仅统计 Content 文本，
// 忽略 MultiContent（多模态）内容的局限。
//
// 这是一个挑刺测试：一条仅含图片的多模态消息（Content 为空），
// CountTokens 会返回 0，可能导致 Vision 场景下 token 预算严重低估。
func TestCountTokens_IgnoresMultiContent(t *testing.T) {
	p := New("k")
	msg := llm.Message{
		Role: llm.RoleUser,
		MultiContent: []llm.ContentPart{
			{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,AAAA"}},
		},
	}
	got, err := p.CountTokens([]llm.Message{msg})
	if err != nil {
		t.Fatalf("CountTokens 错误: %v", err)
	}
	// 记录现状：多模态消息 token 估算为 0（暴露低估问题）。
	if got != 0 {
		t.Logf("CountTokens 对多模态消息返回 %d（若非 0 说明实现已改进）", got)
	}
}
