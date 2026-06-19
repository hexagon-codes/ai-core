// Package deepseek 的审计测试。
//
// 本文件对 deepseek Provider 做全场景黑盒 + 受控白盒审计，覆盖：
//   - 正常路径：Name/Models/New 默认值
//   - 边界：空 apiKey + 环境变量、空 texts 的 Embed、nil/cancelled context
//   - 错误路径：embedding 在 context 取消时的短路返回
//   - 未闭环逻辑：WithModel Option 是否真正生效
//   - 状态一致性 + 并发竞态：Models 多协程调用、Provider 复用
//
// deepseek 包是 openai.Provider 的薄封装。自 B17 起，deepseek 暴露了
// WithBaseURL/WithHTTPClient 注入点，因此可借助 httptest 在不打真实网络的前提下
// 完整验证请求构造与响应解析；本文件对此也有覆盖（见 TestWithBaseURL_And_HTTPClient_*）。
package deepseek

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"unsafe"

	"github.com/hexagon-codes/ai-core/llm"
)

// TestNew_DefaultModel 验证 New 默认使用 deepseek-chat 模型。
//
// 由于 deepseek 包不暴露 model 字段的读取入口，这里通过反射读取
// 内嵌 openai.Provider 的私有 model 字段来观测真实生效的默认值。
func TestNew_DefaultModel(t *testing.T) {
	p := New("test-key")
	if got := readModel(t, p); got != defaultModel {
		t.Fatalf("默认模型 = %q, 期望 %q", got, defaultModel)
	}
}

// TestWithModel_TakesEffect 验证 WithModel Option 能真正修改默认模型。
//
// 这是对"未闭环逻辑"的针对性测试：deepseek.go 中 WithModel 的函数体为空，
// 仅有一句注释，实际不做任何事。期望传入 deepseek-reasoner 后默认模型随之改变。
// 若该测试失败，说明 WithModel 是一个对外宣称可用、实际是 no-op 的死接口。
func TestWithModel_TakesEffect(t *testing.T) {
	// 回归: B16 WithModel 曾是空 no-op，模型配置不生效；修复后透传给 openai.WithModel。
	const want = "deepseek-reasoner"
	p := New("test-key", WithModel(want))
	if got := readModel(t, p); got != want {
		t.Fatalf("WithModel(%q) 后默认模型 = %q, 期望 %q —— WithModel 未生效", want, got, want)
	}
}

// TestName 验证 Name 覆盖了内嵌 openai.Provider 的 "openai"，返回 "deepseek"。
func TestName(t *testing.T) {
	p := New("test-key")
	if got := p.Name(); got != "deepseek" {
		t.Fatalf("Name() = %q, 期望 %q", got, "deepseek")
	}
}

// TestModels 验证 Models 返回的模型列表内容、特性、成本与文档一致。
func TestModels(t *testing.T) {
	p := New("test-key")
	models := p.Models()

	if len(models) != 2 {
		t.Fatalf("Models() 返回 %d 个模型, 期望 2", len(models))
	}

	// 按 ID 建索引，便于断言
	byID := make(map[string]llm.ModelInfo, len(models))
	for _, m := range models {
		byID[m.ID] = m
	}

	tests := []struct {
		id           string
		maxTokens    int
		inputCost    float64
		outputCost   float64
		wantFeatures []string
	}{
		{
			id:           "deepseek-chat",
			maxTokens:    64000,
			inputCost:    0.14,
			outputCost:   0.28,
			wantFeatures: []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			id:           "deepseek-reasoner",
			maxTokens:    64000,
			inputCost:    0.55,
			outputCost:   2.19,
			wantFeatures: []string{llm.FeatureStreaming},
		},
	}

	for _, tt := range tests {
		t.Run(tt.id, func(t *testing.T) {
			m, ok := byID[tt.id]
			if !ok {
				t.Fatalf("模型列表缺少 %q", tt.id)
			}
			if m.MaxTokens != tt.maxTokens {
				t.Errorf("MaxTokens = %d, 期望 %d", m.MaxTokens, tt.maxTokens)
			}
			if m.InputCost != tt.inputCost {
				t.Errorf("InputCost = %v, 期望 %v", m.InputCost, tt.inputCost)
			}
			if m.OutputCost != tt.outputCost {
				t.Errorf("OutputCost = %v, 期望 %v", m.OutputCost, tt.outputCost)
			}
			for _, f := range tt.wantFeatures {
				if !m.HasFeature(f) {
					t.Errorf("模型 %q 缺少特性 %q", tt.id, f)
				}
			}
			if m.Name == "" {
				t.Errorf("模型 %q Name 为空", tt.id)
			}
		})
	}
}

// TestModels_JSONModeFeatureConsistency 验证 Models 声明的 JSON 特性常量与 llm 包一致。
//
// deepseek.go 使用 llm.FeatureJSON，其值为 "json_mode"。本测试钉死该契约，
// 防止特性常量被悄悄改名后模型能力声明与上层判断错配。
func TestModels_JSONModeFeatureConsistency(t *testing.T) {
	p := New("test-key")
	var chat *llm.ModelInfo
	for i := range p.Models() {
		m := p.Models()[i]
		if m.ID == "deepseek-chat" {
			chat = &m
			break
		}
	}
	if chat == nil {
		t.Fatal("未找到 deepseek-chat 模型")
	}
	if !chat.HasFeature(llm.FeatureJSON) {
		t.Errorf("deepseek-chat 应支持 %q 特性", llm.FeatureJSON)
	}
	if llm.FeatureJSON != "json_mode" {
		t.Errorf("llm.FeatureJSON 常量值 = %q, 期望 %q", llm.FeatureJSON, "json_mode")
	}
}

// TestNew_EmptyAPIKeyReadsEnv 验证空 apiKey 时从环境变量 DEEPSEEK_API_KEY 读取。
func TestNew_EmptyAPIKeyReadsEnv(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "env-secret-key")
	p := New("")
	if got := readAPIKey(t, p); got != "env-secret-key" {
		t.Fatalf("空 apiKey 时 apiKey = %q, 期望从环境变量读取 %q", got, "env-secret-key")
	}
}

// TestNew_ExplicitAPIKeyOverridesEnv 验证显式 apiKey 优先于环境变量。
func TestNew_ExplicitAPIKeyOverridesEnv(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "env-secret-key")
	p := New("explicit-key")
	if got := readAPIKey(t, p); got != "explicit-key" {
		t.Fatalf("显式 apiKey = %q, 期望 %q", got, "explicit-key")
	}
}

// TestNew_EmptyAPIKeyNoEnv 验证空 apiKey 且无环境变量时不 panic，apiKey 为空。
func TestNew_EmptyAPIKeyNoEnv(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	p := New("")
	if p == nil {
		t.Fatal("New 返回 nil")
	}
	if got := readAPIKey(t, p); got != "" {
		t.Fatalf("无 apiKey 时 apiKey = %q, 期望空", got)
	}
}

// TestEmbed_EmptyTexts 验证空 texts 时 Embed 短路返回 nil,nil，不发起网络请求。
func TestEmbed_EmptyTexts(t *testing.T) {
	p := New("test-key")
	embeds, err := p.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(nil) 返回错误: %v", err)
	}
	if embeds != nil {
		t.Fatalf("Embed(nil) = %v, 期望 nil", embeds)
	}

	embeds, err = p.Embed(context.Background(), []string{})
	if err != nil {
		t.Fatalf("Embed([]) 返回错误: %v", err)
	}
	if embeds != nil {
		t.Fatalf("Embed([]) = %v, 期望 nil", embeds)
	}
}

// TestEmbedWithModel_CancelledContext 验证 context 已取消且 texts 非空时短路返回 ctx.Err()。
//
// 这是错误路径测试：embedding.go 在构造请求前检查 ctx.Err()，
// 已取消的 context 应在不打网络的前提下立即返回错误。
func TestEmbedWithModel_CancelledContext(t *testing.T) {
	p := New("test-key")
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 立即取消

	_, err := p.EmbedWithModel(ctx, "deepseek-embed", []string{"hello"})
	if err == nil {
		t.Fatal("已取消 context 下 EmbedWithModel 应返回错误, 实际返回 nil")
	}
	if err != context.Canceled {
		t.Fatalf("EmbedWithModel 错误 = %v, 期望 context.Canceled", err)
	}
}

// TestEmbed_EmptyTextsBeforeContextCheck 验证空 texts 的检查优先于 context 检查。
//
// 边界一致性测试：当 texts 为空且 context 已取消时，按 embedding.go 的代码顺序
// （先判 len(texts)==0 再判 ctx.Err()），应返回 nil,nil 而非 ctx.Err()。
func TestEmbed_EmptyTextsBeforeContextCheck(t *testing.T) {
	p := New("test-key")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	embeds, err := p.EmbedWithModel(ctx, "m", nil)
	if err != nil {
		t.Fatalf("空 texts + 已取消 context: 期望 nil error, 实际 %v", err)
	}
	if embeds != nil {
		t.Fatalf("空 texts: 期望 nil 切片, 实际 %v", embeds)
	}
}

// TestInterfaceCompliance 验证 *Provider 同时满足 Provider 与 EmbeddingProvider 接口。
func TestInterfaceCompliance(t *testing.T) {
	var _ llm.Provider = New("k")
	var _ llm.EmbeddingProvider = New("k")
}

// TestCountTokens_InheritedEstimation 验证继承自 openai 的 CountTokens 估算逻辑。
//
// openai.CountTokens 以约 4 字符 = 1 token 估算，纯本地计算无网络。
func TestCountTokens_InheritedEstimation(t *testing.T) {
	p := New("test-key")
	tests := []struct {
		name string
		msgs []llm.Message
		want int
	}{
		{name: "空列表", msgs: nil, want: 0},
		{name: "空内容", msgs: []llm.Message{llm.UserMessage("")}, want: 0},
		{name: "4字符", msgs: []llm.Message{llm.UserMessage("abcd")}, want: 1},
		{name: "中文unicode", msgs: []llm.Message{llm.UserMessage("你好")}, want: len("你好") / 4},
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

// TestModels_Concurrent 验证 Models 在并发调用下无数据竞争（配合 -race）。
func TestModels_Concurrent(t *testing.T) {
	p := New("test-key")
	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			ms := p.Models()
			if len(ms) != 2 {
				t.Errorf("并发 Models() 返回 %d 个, 期望 2", len(ms))
			}
			_ = p.Name()
		}()
	}
	wg.Wait()
}

// TestNew_OptionOrderStability 验证多次传入 WithModel 不 panic 且 Provider 可用。
//
// 即便 WithModel 当前为 no-op，也应保证 Option 链式调用的健壮性。
func TestNew_OptionOrderStability(t *testing.T) {
	p := New("k", WithModel("a"), WithModel("b"), WithModel("c"))
	if p == nil {
		t.Fatal("多 Option 下 New 返回 nil")
	}
	if p.Name() != "deepseek" {
		t.Fatalf("多 Option 下 Name() = %q", p.Name())
	}
}

// TestWithBaseURL_TakesEffect 验证 WithBaseURL 能透传给底层 openai.Provider。
//
// 回归: B17 deepseek 此前无 baseURL 注入点，只能打 api.deepseek.com。
// 通过反射读取内嵌 openai.Provider 的私有 baseURL 字段，确认配置真正生效。
func TestWithBaseURL_TakesEffect(t *testing.T) {
	const want = "https://gateway.internal/v1"
	p := New("test-key", WithBaseURL(want))
	if got := readBaseURL(t, p); got != want {
		t.Fatalf("WithBaseURL(%q) 后 baseURL = %q, 期望 %q —— WithBaseURL 未生效", want, got, want)
	}
}

// TestWithBaseURL_And_HTTPClient_RoundTrip 用 httptest 验证请求构造与响应解析全链路。
//
// 回归: B17 在不打真实网络的前提下，借助 WithBaseURL/WithHTTPClient 注入点把请求
// 指向本地 httptest 服务器，断言：
//   - 请求落到了注入的 baseURL（path = /chat/completions）
//   - Authorization 头携带了 apiKey
//   - 省略 Model 时使用 WithModel 设定的默认模型（联动验证 B16）
//   - 响应能被底层 openai.Provider 正确解析回 CompletionResponse
func TestWithBaseURL_And_HTTPClient_RoundTrip(t *testing.T) {
	var gotPath, gotAuth, gotModel string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")

		var reqBody struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		gotModel = reqBody.Model

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"model": "deepseek-reasoner",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
		}`))
	}))
	defer srv.Close()

	p := New(
		"secret-key",
		WithBaseURL(srv.URL),
		WithHTTPClient(srv.Client()),
		WithModel("deepseek-reasoner"),
	)

	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{llm.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Complete 返回错误: %v", err)
	}

	if gotPath != "/chat/completions" {
		t.Errorf("请求 path = %q, 期望 %q", gotPath, "/chat/completions")
	}
	if gotAuth != "Bearer secret-key" {
		t.Errorf("Authorization = %q, 期望 %q", gotAuth, "Bearer secret-key")
	}
	if gotModel != "deepseek-reasoner" {
		t.Errorf("请求 model = %q, 期望默认透传 %q（B16 联动）", gotModel, "deepseek-reasoner")
	}
	if resp.Content != "你好" {
		t.Errorf("响应 Content = %q, 期望 %q", resp.Content, "你好")
	}
	if resp.ID != "chatcmpl-test" {
		t.Errorf("响应 ID = %q, 期望 %q", resp.ID, "chatcmpl-test")
	}
}

// ---------- 反射辅助：读取内嵌 openai.Provider 的私有字段 ----------
//
// deepseek 包未暴露 model / apiKey 的读取入口，为在不打网络的前提下观测
// New 与 WithModel 的真实效果，这里通过 reflect + unsafe 读取内嵌结构体的私有字段。
// 仅用于测试观测，不修改任何被测源码。

func readModel(t *testing.T, p *Provider) string {
	t.Helper()
	return readUnexportedString(t, p.Provider, "model")
}

func readAPIKey(t *testing.T, p *Provider) string {
	t.Helper()
	return readUnexportedString(t, p.Provider, "apiKey")
}

func readBaseURL(t *testing.T, p *Provider) string {
	t.Helper()
	return readUnexportedString(t, p.Provider, "baseURL")
}

// readUnexportedString 读取目标结构体指针上指定名称的私有 string 字段。
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
	// 私有字段需借助 unsafe 才能读取
	fv := reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem()
	return fv.String()
}
