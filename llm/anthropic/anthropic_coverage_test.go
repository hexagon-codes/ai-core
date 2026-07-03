package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// 本文件把 anthropic provider 的全场景单测补齐：配置/请求构造/角色转换/
// HTTP 成功与错误路径/流式/响应解析。配合 anthropic_cache_test.go。

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
	t.Setenv("ANTHROPIC_API_KEY", "from-env")
	if got := New("").apiKey; got != "from-env" {
		t.Fatalf("空 key 应回退环境变量, got %q", got)
	}
}

func TestName(t *testing.T) {
	if New("k").Name() != "anthropic" {
		t.Fatal("Name 应为 anthropic")
	}
}

func TestModels(t *testing.T) {
	models := New("k").Models()
	if len(models) != 5 {
		t.Fatalf("应有 5 个模型, got %d", len(models))
	}
	if models[0].ID != "claude-opus-4-20250514" || models[0].MaxTokens != 200000 {
		t.Fatalf("首个模型字段错误: %+v", models[0])
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

func TestConvertRole(t *testing.T) {
	cases := map[llm.Role]string{
		llm.RoleUser:      "user",
		llm.RoleAssistant: "assistant",
		llm.RoleSystem:    "user", // default 分支
		llm.Role("weird"): "user",
	}
	for in, want := range cases {
		if got := convertRole(in); got != want {
			t.Errorf("convertRole(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSetHeaders(t *testing.T) {
	req, _ := http.NewRequest("POST", "http://x", nil)
	New("secret").setHeaders(req)
	if req.Header.Get("x-api-key") != "secret" ||
		req.Header.Get("anthropic-version") != anthropicVersion ||
		req.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("headers 错误: %+v", req.Header)
	}
}

func TestBuildRequestBody_AllBranches(t *testing.T) {
	p := New("k")
	temp := 0.5
	topP := 0.9
	req := llm.CompletionRequest{
		Model: "claude-x",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "you are helpful"},
			{Role: llm.RoleUser, Content: "hi"},
		},
		MaxTokens:   256,
		Temperature: &temp,
		TopP:        &topP,
		Stop:        []string{"END"},
		Tools:       []llm.ToolDefinition{llm.NewToolDefinition("get_weather", "查天气", nil)},
	}
	body, sys, err := p.buildRequestBody(req, true)
	if err != nil {
		t.Fatal(err)
	}
	if sys != "you are helpful" {
		t.Fatalf("system 分离错误: %q", sys)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["system"] != "you are helpful" {
		t.Errorf("system 未写入 payload")
	}
	if payload["max_tokens"].(float64) != 256 {
		t.Errorf("max_tokens 覆盖失败: %v", payload["max_tokens"])
	}
	if payload["temperature"].(float64) != 0.5 || payload["top_p"].(float64) != 0.9 {
		t.Errorf("temperature/top_p 未写入")
	}
	if payload["stream"] != true {
		t.Errorf("stream 标志错误")
	}
	if _, ok := payload["stop_sequences"]; !ok {
		t.Errorf("stop_sequences 未写入")
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Errorf("tools 未写入: %v", payload["tools"])
	}
	tool := tools[0].(map[string]any)
	inputSchema, ok := tool["input_schema"].(map[string]any)
	if !ok || inputSchema["type"] != "object" {
		t.Fatalf("nil tool schema 应兜底为空 object, got %#v", tool["input_schema"])
	}
	msgs := payload["messages"].([]any)
	if len(msgs) != 1 { // system 被分离，只剩 user
		t.Errorf("messages 应只含 1 条 user, got %d", len(msgs))
	}
}

func TestBuildRequestBody_DefaultMaxTokensNoSystem(t *testing.T) {
	p := New("k")
	body, sys, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}},
	}, false)
	if err != nil || sys != "" {
		t.Fatalf("无 system 时 sys 应为空: %q %v", sys, err)
	}
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	if payload["max_tokens"].(float64) != 4096 {
		t.Errorf("默认 max_tokens 应 4096, got %v", payload["max_tokens"])
	}
	if _, ok := payload["system"]; ok {
		t.Errorf("无 system 时不应有 system 字段")
	}
}

func TestBuildRequestBody_MultiContent(t *testing.T) {
	p := New("k")
	body, _, err := p.buildRequestBody(llm.CompletionRequest{
		Model: "claude-x",
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			MultiContent: []llm.ContentPart{
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.com/a.png", Detail: "high"}},
				{Type: "text", Text: "describe this"},
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,aGVsbG8="}},
			},
		}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	messages := payload["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	if len(content) != 3 {
		t.Fatalf("expected 3 content blocks, got %#v", content)
	}

	urlBlock := content[0].(map[string]any)
	urlSource := urlBlock["source"].(map[string]any)
	if urlBlock["type"] != "image" || urlSource["type"] != "url" ||
		urlSource["url"] != "https://example.com/a.png" {
		t.Fatalf("URL image block not encoded for Anthropic: %#v", urlBlock)
	}
	textBlock := content[1].(map[string]any)
	if textBlock["type"] != "text" || textBlock["text"] != "describe this" {
		t.Fatalf("text block not encoded: %#v", textBlock)
	}
	base64Block := content[2].(map[string]any)
	base64Source := base64Block["source"].(map[string]any)
	if base64Block["type"] != "image" || base64Source["type"] != "base64" ||
		base64Source["media_type"] != "image/png" || base64Source["data"] != "aGVsbG8=" {
		t.Fatalf("base64 image block not encoded for Anthropic: %#v", base64Block)
	}
}

func TestComplete_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "k" {
			t.Errorf("缺 api key")
		}
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-x","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hello"}],
			"usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer srv.Close()
	p := New("k", WithBaseURL(srv.URL))
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello" || resp.Usage.TotalTokens != 15 || resp.FinishReason != "end_turn" {
		t.Fatalf("响应解析错误: %+v", resp)
	}
}

func TestComplete_ToolUseResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"x","content":[
			{"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"SF"}}]}`))
	}))
	defer srv.Close()
	resp, err := New("k", WithBaseURL(srv.URL)).Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "weather?"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "get_weather" ||
		!strings.Contains(resp.ToolCalls[0].Arguments, "SF") {
		t.Fatalf("tool_use 解析错误: %+v", resp.ToolCalls)
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
		if got := r.Header.Get("x-api-key"); got != "k" {
			t.Fatalf("x-api-key = %q, want provider key", got)
		}
		if got := r.Header.Get("anthropic-version"); got != anthropicVersion {
			t.Fatalf("anthropic-version = %q, want provider version", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Fatalf("Content-Type = %q, want provider content type", got)
		}
		if got := r.Header.Get("anthropic-beta"); got != "tools-2024-04-04" {
			t.Fatalf("safe anthropic-beta header missing, got %q", got)
		}
		if got := r.Header.Get("X-Debug-Trace"); got != "trace-1" {
			t.Fatalf("safe custom header missing, got %q", got)
		}
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-x","stop_reason":"end_turn",
			"content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	p := New("k",
		WithBaseURL(srv.URL),
		WithHeaders(map[string]string{
			"x-api-key":         "attacker-key",
			"anthropic-version": "2099-01-01",
			"Content-Type":      "text/plain",
			"anthropic-beta":    "tools-2024-04-04",
			"X-Debug-Trace":     "trace-1",
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
		_, _ = w.Write([]byte(`{"id":"msg_1","content":[{"type":"text","text":"ok"}]}`))
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

func TestComplete_Stream_RequestBuildAndDoErrors(t *testing.T) {
	req := llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}}}

	// 1) NewRequestWithContext 失败：baseURL 缺协议头 → URL 解析错误
	badURL := New("k", WithBaseURL("://invalid"))
	if _, err := badURL.Complete(context.Background(), req); err == nil {
		t.Error("Complete 非法 URL 应报错")
	}
	if _, err := badURL.Stream(context.Background(), req); err == nil {
		t.Error("Stream 非法 URL 应报错")
	}

	// 2) httpClient.Do 失败：指向不可达端口 → 连接拒绝
	down := New("k", WithBaseURL("http://127.0.0.1:1"))
	if _, err := down.Complete(context.Background(), req); err == nil {
		t.Error("Complete 不可达上游应报错")
	}
	if _, err := down.Stream(context.Background(), req); err == nil {
		t.Error("Stream 不可达上游应报错")
	}
}

func TestStream_SuccessAndError(t *testing.T) {
	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message_stop\ndata: {}\n\n"))
	}))
	defer okSrv.Close()
	s, err := New("k", WithBaseURL(okSrv.URL)).Stream(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil || s == nil {
		t.Fatalf("流式成功应返回非 nil stream: %v", err)
	}

	errSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer errSrv.Close()
	_, err = New("k", WithBaseURL(errSrv.URL)).Stream(context.Background(), llm.CompletionRequest{
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
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\n"))
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
