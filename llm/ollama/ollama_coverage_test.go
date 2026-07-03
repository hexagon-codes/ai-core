package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

// ollama provider 全场景单测：配置/请求构造/HTTP 成功与错误/流式/模型发现/
// embedding/ping/pull/thinking-metadata。配合 ollama_test.go（已覆盖 thinking 分支）。

func userReq(content string) llm.CompletionRequest {
	return llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: content}}}
}

func TestNew_EnvAndOptions(t *testing.T) {
	if p := New(); p.baseURL != defaultBaseURL || p.model != defaultModel {
		t.Fatalf("默认值错误: %+v", p)
	}
	p := New()
	baseTransport, ok := p.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("default http client transport type = %T, want *http.Transport", p.httpClient.Transport)
	}
	streamTransport, ok := p.streamHTTPClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("stream http client transport type = %T, want *http.Transport", p.streamHTTPClient.Transport)
	}
	if baseTransport.ResponseHeaderTimeout != ollamaDefaultResponseHeaderWait ||
		streamTransport.ResponseHeaderTimeout != ollamaStreamResponseHeaderWait {
		t.Fatalf("response header timeout mismatch: base=%v stream=%v",
			baseTransport.ResponseHeaderTimeout, streamTransport.ResponseHeaderTimeout)
	}
	t.Setenv("OLLAMA_HOST", "http://envhost:1234")
	t.Setenv("OLLAMA_MODEL", "envmodel")
	if p = New(); p.baseURL != "http://envhost:1234" || p.model != "envmodel" {
		t.Fatalf("环境变量未生效: %+v", p)
	}
	custom := &http.Client{}
	p = New(WithBaseURL("http://x"), WithModel("m"), WithHTTPClient(custom))
	if p.baseURL != "http://x" || p.model != "m" || p.httpClient != custom || p.streamHTTPClient != custom {
		t.Fatalf("选项未生效（应覆盖环境变量）: %+v", p)
	}
}

func TestName_CountTokens(t *testing.T) {
	p := New()
	if p.Name() != "ollama" {
		t.Fatal("Name 应为 ollama")
	}
	n, err := p.CountTokens([]llm.Message{{Content: "12345678"}, {Content: "1234"}})
	if err != nil || n != 3 {
		t.Fatalf("CountTokens=%d,%v want 3", n, err)
	}
}

func TestOllamaThinkFromMetadata(t *testing.T) {
	cases := []struct {
		meta map[string]any
		val  any
		ok   bool
	}{
		{nil, false, false},
		{map[string]any{}, false, false},
		{map[string]any{"thinking": true}, true, true},
		{map[string]any{"think": false}, false, true},
		{map[string]any{"thinking": "on"}, true, true},
		{map[string]any{"thinking": "ENABLED"}, true, true},
		{map[string]any{"think": "off"}, false, true},
		{map[string]any{"think": "no"}, false, true},
		{map[string]any{"think": "HIGH"}, "high", true},
		{map[string]any{"thinking": "garbage"}, false, false},
		{map[string]any{"other": "x"}, false, false},
		{map[string]any{"thinking": 123}, false, false}, // 非 bool/string
	}
	for i, c := range cases {
		v, ok := ollamaThinkFromMetadata(c.meta)
		if v != c.val || ok != c.ok {
			t.Errorf("case %d: got (%v,%v) want (%v,%v)", i, v, ok, c.val, c.ok)
		}
	}
}

func TestBuildRequestBody_Options(t *testing.T) {
	p := New()
	temp, topP := 0.3, 0.8
	req := llm.CompletionRequest{
		Model:          "m",
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Temperature:    &temp,
		TopP:           &topP,
		MaxTokens:      100,
		Stop:           []string{"X"},
		Tools:          []llm.ToolDefinition{llm.NewToolDefinition("f", "d", nil)},
		ResponseFormat: &llm.ResponseFormat{Type: "json_object"},
	}
	body, err := p.buildRequestBody(req, true)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	if payload["stream"] != true || payload["format"] != "json" {
		t.Errorf("stream/format 错误: %v", payload)
	}
	opts := payload["options"].(map[string]any)
	if opts["temperature"].(float64) != 0.3 || opts["top_p"].(float64) != 0.8 ||
		opts["num_predict"].(float64) != 100 || opts["stop"] == nil {
		t.Errorf("options 错误: %v", opts)
	}
	if opts["num_ctx"].(float64) != defaultOllamaNumCtx {
		t.Errorf("num_ctx 默认值错误: %v", opts)
	}
	if payload["tools"] == nil {
		t.Errorf("tools 未写入")
	}
}

func TestBuildRequestBody_NumCtxFromMetadataAndModelContext(t *testing.T) {
	p := New()
	p.models = []llm.ModelInfo{{ID: "qwen3.5:9b", Name: "qwen3.5:9b", MaxTokens: 262144}}

	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if unmarshalErr := json.Unmarshal(body, &payload); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	opts := payload["options"].(map[string]any)
	if opts["num_ctx"].(float64) != maxAutomaticNumCtx {
		t.Fatalf("model context should clamp to automatic cap: %v", opts)
	}

	body, err = p.buildRequestBody(llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		Metadata: map[string]any{"num_ctx": "16384"},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	if unmarshalErr := json.Unmarshal(body, &payload); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	opts = payload["options"].(map[string]any)
	if opts["num_ctx"].(float64) != 16384 {
		t.Fatalf("metadata num_ctx should override model context: %v", opts)
	}
}

func TestBuildRequestBody_ConvertsOpenAIMultimodalToOllamaImages(t *testing.T) {
	p := New()
	body, err := p.buildRequestBody(llm.CompletionRequest{
		Model: "qwen3.5:9b",
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				MultiContent: []llm.ContentPart{
					{Type: "text", Text: "识别图片文字"},
					{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,QUJDRA==", Detail: "high"}},
				},
			},
		},
	}, false)
	if err != nil {
		t.Fatal(err)
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	messages := payload["messages"].([]any)
	msg := messages[0].(map[string]any)
	if msg["content"] != "识别图片文字" {
		t.Fatalf("content = %#v, want text content", msg["content"])
	}
	images := msg["images"].([]any)
	if len(images) != 1 || images[0] != "QUJDRA==" {
		t.Fatalf("images = %#v, want base64 payload", msg["images"])
	}
}

func TestBuildRequestBody_RejectsNonBase64OllamaImage(t *testing.T) {
	p := New()
	_, err := p.buildRequestBody(llm.CompletionRequest{
		Model: "qwen3.5:9b",
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				MultiContent: []llm.ContentPart{
					{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.com/image.png"}},
				},
			},
		},
	}, false)
	if err == nil || !strings.Contains(err.Error(), "base64") {
		t.Fatalf("expected base64 validation error, got %v", err)
	}
}

func TestComplete_Success_ToolCalls_Done(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","created_at":"t1","done":true,
			"message":{"role":"assistant","content":"hi there",
				"tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"SF"}}}]},
			"prompt_eval_count":10,"eval_count":5}`))
	}))
	defer srv.Close()
	resp, err := New(WithBaseURL(srv.URL)).Complete(context.Background(), userReq("q"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hi there" || resp.FinishReason != "stop" || resp.Usage.TotalTokens != 15 {
		t.Fatalf("解析错误: %+v", resp)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "get_weather" ||
		!strings.Contains(resp.ToolCalls[0].Arguments, "SF") {
		t.Fatalf("tool call 错误: %+v", resp.ToolCalls)
	}
}

func TestComplete_DefaultModelInjected(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p map[string]any
		_ = json.NewDecoder(r.Body).Decode(&p)
		gotModel, _ = p["model"].(string)
		_, _ = w.Write([]byte(`{"done":true,"message":{"content":"ok"}}`))
	}))
	defer srv.Close()
	_, err := New(WithBaseURL(srv.URL), WithModel("my-default")).Complete(context.Background(),
		llm.CompletionRequest{Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}}})
	if err != nil || gotModel != "my-default" {
		t.Fatalf("默认模型未注入: %q %v", gotModel, err)
	}
}

func TestComplete_Errors(t *testing.T) {
	// 非 200
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("model not found"))
	}))
	defer bad.Close()
	if _, err := New(WithBaseURL(bad.URL)).Complete(context.Background(), userReq("q")); err == nil ||
		!strings.Contains(err.Error(), "model not found") {
		t.Errorf("非200 应含 body, got %v", err)
	}
	// 畸形 JSON
	mal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{bad`))
	}))
	defer mal.Close()
	if _, err := New(WithBaseURL(mal.URL)).Complete(context.Background(), userReq("q")); err == nil {
		t.Error("畸形 JSON 应报错")
	}
	// 非法 URL + 不可达
	if _, err := New(WithBaseURL("://bad")).Complete(context.Background(), userReq("q")); err == nil {
		t.Error("非法 URL 应报错")
	}
	if _, err := New(WithBaseURL("http://127.0.0.1:1")).Complete(context.Background(), userReq("q")); err == nil {
		t.Error("不可达应报错")
	}
}

func TestStream_SuccessAndErrors(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"done":true,"message":{"content":"x"}}` + "\n"))
	}))
	defer ok.Close()
	s, err := New(WithBaseURL(ok.URL)).Stream(context.Background(), userReq("q"))
	if err != nil || s == nil {
		t.Fatalf("流式成功应返回 stream: %v", err)
	}
	result, err := s.Collect()
	if err != nil {
		t.Fatalf("Ollama JSON Lines 流解析失败: %v", err)
	}
	if result.Content != "x" || result.FinishReason != "stop" {
		t.Fatalf("Ollama JSON Lines 流解析错误: %+v", result)
	}

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("boom"))
	}))
	defer bad.Close()
	if _, err := New(WithBaseURL(bad.URL)).Stream(context.Background(), userReq("q")); err == nil ||
		!strings.Contains(err.Error(), "boom") {
		t.Errorf("流式非200 应报错, got %v", err)
	}
	if _, err := New(WithBaseURL("://bad")).Stream(context.Background(), userReq("q")); err == nil {
		t.Error("流式非法 URL 应报错")
	}
	if _, err := New(WithBaseURL("http://127.0.0.1:1")).Stream(context.Background(), userReq("q")); err == nil {
		t.Error("流式不可达应报错")
	}
}

func TestModels_FetchCacheAndFallback(t *testing.T) {
	// fetchLocalModels 成功 → 缓存
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b","details":{"family":"qwen","parameter_size":"7B"}}]}`))
	}))
	defer srv.Close()
	p := New(WithBaseURL(srv.URL))
	models := p.Models()
	if len(models) != 1 || models[0].ID != "qwen2.5:7b" {
		t.Fatalf("fetchLocalModels 失败: %+v", models)
	}
	// 第二次走缓存分支
	if got := p.Models(); len(got) != 1 {
		t.Fatalf("缓存分支错误: %+v", got)
	}
	// fallback：上游不可达 → 默认列表
	fb := New(WithBaseURL("http://127.0.0.1:1")).Models()
	if len(fb) < 5 {
		t.Fatalf("fallback 默认列表应非空: %d", len(fb))
	}
}

func TestModels_MapsOllamaCapabilitiesFromTags(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Fatalf("unexpected endpoint %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen3.5:9b","capabilities":["vision","completion","tools","thinking"],"details":{"family":"qwen35","parameter_size":"9.7B","context_length":262144}}]}`))
	}))
	defer srv.Close()

	models := New(WithBaseURL(srv.URL)).Models()
	if len(models) != 1 {
		t.Fatalf("models length = %d, want 1: %+v", len(models), models)
	}
	if !models[0].HasFeature(llm.FeatureVision) {
		t.Fatalf("qwen3.5:9b should expose FeatureVision, features=%v", models[0].Features)
	}
	if !models[0].HasFeature(llm.FeatureFunctions) {
		t.Fatalf("qwen3.5:9b should expose FeatureFunctions, features=%v", models[0].Features)
	}
	if !models[0].HasFeature(llm.FeatureStreaming) {
		t.Fatalf("ollama models should keep FeatureStreaming, features=%v", models[0].Features)
	}
	if models[0].MaxTokens != 262144 {
		t.Fatalf("context_length should be mapped to MaxTokens, got %d", models[0].MaxTokens)
	}
}

func TestModels_ReadsModelFieldFromTags(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Fatalf("unexpected endpoint %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"models":[{"model":"qwen3.5:9b","capabilities":["vision"],"details":{"family":"qwen35","parameter_size":"9.7B"}}]}`))
	}))
	defer srv.Close()

	models := New(WithBaseURL(srv.URL)).Models()
	if len(models) != 1 || models[0].ID != "qwen3.5:9b" {
		t.Fatalf("models should read model field from /api/tags: %+v", models)
	}
	if !models[0].HasFeature(llm.FeatureVision) {
		t.Fatalf("model field response should still map capabilities: %+v", models[0])
	}
}

func TestModels_FallsBackToShowCapabilities(t *testing.T) {
	var showCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen3.5:9b","details":{"family":"qwen35","parameter_size":"9.7B"}}]}`))
		case "/api/show":
			showCalls++
			var req map[string]string
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode /api/show request: %v", err)
			}
			if req["model"] != "qwen3.5:9b" {
				t.Fatalf("/api/show model = %q, want qwen3.5:9b", req["model"])
			}
			_, _ = w.Write([]byte(`{"capabilities":["completion","vision"]}`))
		default:
			t.Fatalf("unexpected endpoint %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	models := New(WithBaseURL(srv.URL)).Models()
	if showCalls != 1 {
		t.Fatalf("/api/show calls = %d, want 1", showCalls)
	}
	if len(models) != 1 || !models[0].HasFeature(llm.FeatureVision) {
		t.Fatalf("show capabilities should mark model as vision-capable: %+v", models)
	}
}

func TestFetchLocalModels_Errors(t *testing.T) {
	// 不可达 → error（经 Models fallback，但直接调也覆盖 error 分支）
	if _, err := New(WithBaseURL("http://127.0.0.1:1")).fetchLocalModels(); err == nil {
		t.Error("不可达应报错")
	}
	// 畸形 JSON
	mal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{bad`))
	}))
	defer mal.Close()
	if _, err := New(WithBaseURL(mal.URL)).fetchLocalModels(); err == nil {
		t.Error("畸形 JSON 应报错")
	}
}

func TestPing(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer ok.Close()
	if err := New(WithBaseURL(ok.URL)).Ping(context.Background()); err != nil {
		t.Errorf("Ping 成功应无错: %v", err)
	}
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer bad.Close()
	if err := New(WithBaseURL(bad.URL)).Ping(context.Background()); err == nil {
		t.Error("Ping 非200 应报错")
	}
	if err := New(WithBaseURL("://bad")).Ping(context.Background()); err == nil {
		t.Error("Ping 非法 URL 应报错")
	}
	if err := New(WithBaseURL("http://127.0.0.1:1")).Ping(context.Background()); err == nil {
		t.Error("Ping 不可达应报错")
	}
}

func TestPullModel(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer ok.Close()
	if err := New(WithBaseURL(ok.URL)).PullModel(context.Background(), "llama3.2"); err != nil {
		t.Errorf("PullModel 成功应无错: %v", err)
	}
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("pull failed"))
	}))
	defer bad.Close()
	if err := New(WithBaseURL(bad.URL)).PullModel(context.Background(), "m"); err == nil ||
		!strings.Contains(err.Error(), "pull failed") {
		t.Errorf("PullModel 非200 应报错, got %v", err)
	}
	if err := New(WithBaseURL("://bad")).PullModel(context.Background(), "m"); err == nil {
		t.Error("PullModel 非法 URL 应报错")
	}
	if err := New(WithBaseURL("http://127.0.0.1:1")).PullModel(context.Background(), "m"); err == nil {
		t.Error("PullModel 不可达应报错")
	}
}

func TestEmbed(t *testing.T) {
	// 空输入 → nil,nil
	if v, err := New().Embed(context.Background(), nil); v != nil || err != nil {
		t.Errorf("空输入应返回 nil,nil")
	}
	// 成功
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"nomic-embed-text","embeddings":[[0.1,0.2],[0.3,0.4]]}`))
	}))
	defer srv.Close()
	v, err := New(WithBaseURL(srv.URL)).Embed(context.Background(), []string{"a", "b"})
	if err != nil || len(v) != 2 || len(v[0]) != 2 || v[0][0] != float32(0.1) {
		t.Fatalf("Embed 解析错误: %+v %v", v, err)
	}
	// EmbedWithModel 指定模型
	if _, err := New(WithBaseURL(srv.URL)).EmbedWithModel(context.Background(), "custom", []string{"a"}); err != nil {
		t.Errorf("EmbedWithModel 应成功: %v", err)
	}
	// 非200 / 畸形 / 非法 URL / 不可达
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("embed err"))
	}))
	defer bad.Close()
	if _, err := New(WithBaseURL(bad.URL)).Embed(context.Background(), []string{"a"}); err == nil {
		t.Error("Embed 非200 应报错")
	}
	mal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{bad`))
	}))
	defer mal.Close()
	if _, err := New(WithBaseURL(mal.URL)).Embed(context.Background(), []string{"a"}); err == nil {
		t.Error("Embed 畸形 JSON 应报错")
	}
	if _, err := New(WithBaseURL("://bad")).Embed(context.Background(), []string{"a"}); err == nil {
		t.Error("Embed 非法 URL 应报错")
	}
	if _, err := New(WithBaseURL("http://127.0.0.1:1")).Embed(context.Background(), []string{"a"}); err == nil {
		t.Error("Embed 不可达应报错")
	}
}
