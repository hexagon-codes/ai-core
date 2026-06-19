package voicechat

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
)

// ============================================================================
// 本文件为 Wave8 审计测试，针对 voicechat 包的真实公开 API 编写全场景用例：
//   - Service 路由/校验/状态一致性/并发
//   - openAIVoiceChat 的请求构造、响应解析、错误路径（用 httptest 注入，不打真实网络）
// 所有用例针对真实公开函数，不 mock 掉被测逻辑。
// ============================================================================

// ----------------------------------------------------------------------------
// 一、Service 路由与解析逻辑（resolve / Chat / NewService）
// ----------------------------------------------------------------------------

// TestService_Resolve_Priority 验证 resolve 的优先级：name > model > default。
// table-driven 覆盖各种命中/未命中组合。
func TestService_Resolve_Priority(t *testing.T) {
	pOpenAI := &mockVoiceProvider{name: "openai", models: []string{"gpt-4o-audio-preview"}, result: &Result{Provider: "openai"}}
	pZhipu := &mockVoiceProvider{name: "zhipu", models: []string{"glm-4-voice"}, result: &Result{Provider: "zhipu"}}

	cases := []struct {
		desc      string
		defaultNm string
		reqName   string
		reqModel  string
		wantProv  string // "" 表示期望 resolve 返回 nil
	}{
		{"显式 name 命中优先于 model", "openai", "zhipu", "gpt-4o-audio-preview", "zhipu"},
		{"name 为空按 model 路由", "", "", "glm-4-voice", "zhipu"},
		{"name 不存在但 model 命中", "", "nonexistent", "glm-4-voice", "zhipu"},
		{"name 与 model 都不命中回退 default", "openai", "x", "y", "openai"},
		{"name 与 model 都不命中且无 default", "", "x", "y", ""},
		{"name 命中但 model 为空", "", "zhipu", "", "zhipu"},
		{"全部为空但有 default", "openai", "", "", "openai"},
		{"全部为空且无 default", "", "", "", ""},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			svc := NewService(map[string]Provider{"openai": pOpenAI, "zhipu": pZhipu}, c.defaultNm)
			p := svc.resolve(c.reqName, c.reqModel)
			if c.wantProv == "" {
				if p != nil {
					t.Fatalf("期望 resolve 返回 nil, 实际=%s", p.Name())
				}
				return
			}
			if p == nil {
				t.Fatalf("期望 resolve 返回 %s, 实际=nil", c.wantProv)
			}
			if p.Name() != c.wantProv {
				t.Errorf("期望 resolve=%s, 实际=%s", c.wantProv, p.Name())
			}
		})
	}
}

// TestService_Chat_EmptyInputValidation 验证空输入校验。
// 仅当 audio 和 text 同时为空时才报错；任一非空即可。
func TestService_Chat_EmptyInputValidation(t *testing.T) {
	p := &mockVoiceProvider{name: "x", models: []string{"m"}, result: &Result{Provider: "x"}}

	cases := []struct {
		desc    string
		req     Request
		wantErr bool
	}{
		{"audio+text 皆空报错", Request{}, true},
		{"仅 text", Request{Text: "hi"}, false},
		{"仅 audio", Request{AudioB64: "AAAA"}, false},
		{"both", Request{Text: "hi", AudioB64: "AAAA"}, false},
		{"空白字符 text 仍视为非空(不 trim)", Request{Text: " "}, false},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			svc := NewService(map[string]Provider{"x": p}, "x")
			_, err := svc.Chat(context.Background(), "x", c.req)
			if c.wantErr && err == nil {
				t.Error("期望报错但未报错")
			}
			if !c.wantErr && err != nil {
				t.Errorf("期望成功但报错: %v", err)
			}
		})
	}
}

// TestService_Chat_NoProviderResolved 校验通过但 resolve 返回 nil 时应报错，且不应 panic。
func TestService_Chat_NoProviderResolved(t *testing.T) {
	// 注册了 provider，但请求的 name/model 都不命中且无 default。
	p := &mockVoiceProvider{name: "openai", models: []string{"gpt-4o-audio-preview"}, result: &Result{Provider: "openai"}}
	svc := NewService(map[string]Provider{"openai": p}, "")

	_, err := svc.Chat(context.Background(), "unknown", Request{Text: "hi", Model: "unknown-model"})
	if err == nil {
		t.Fatal("无法解析 provider 时应报错")
	}
	if !strings.Contains(err.Error(), "未找到可用") {
		t.Errorf("错误信息应说明未找到 provider, 实际=%v", err)
	}
	if p.calls != 0 {
		t.Errorf("未解析到 provider 时不应调用任何 provider, calls=%d", p.calls)
	}
}

// TestService_Chat_EmptyInputCheckedBeforeResolve 验证空输入校验先于 provider 解析。
// 即使没有任何 provider，空输入也应返回"audio 与 text 至少需要一个"而非"未找到 provider"。
func TestService_Chat_EmptyInputCheckedBeforeResolve(t *testing.T) {
	svc := NewService(nil, "")
	_, err := svc.Chat(context.Background(), "", Request{})
	if err == nil {
		t.Fatal("空输入应报错")
	}
	if !strings.Contains(err.Error(), "至少需要一个") {
		t.Errorf("应优先返回输入校验错误, 实际=%v", err)
	}
}

// TestNewService_ModelIndexFirstWins 验证多 provider 声明同一 model 时，冲突解析是确定的。
// 此前实现用 map 遍历顺序表达"先注册优先"，但 Go map 无序导致结果非确定。
// 修复后按 provider key 的字典序确定冲突归属（确定性策略）。
// 回归: modelIndex 在多 provider 声明同一 model 时结果非确定（依赖 map 遍历顺序）。
func TestNewService_ModelIndexFirstWins(t *testing.T) {
	pA := &mockVoiceProvider{name: "a", models: []string{"shared"}, result: &Result{Provider: "a"}}
	pB := &mockVoiceProvider{name: "b", models: []string{"shared"}, result: &Result{Provider: "b"}}

	// 多次构建，统计 "shared" model 解析到哪个 provider。
	seen := map[string]int{}
	for i := 0; i < 200; i++ {
		svc := NewService(map[string]Provider{"a": pA, "b": pB}, "")
		p := svc.resolve("", "shared")
		if p == nil {
			t.Fatal("shared model 应被索引")
		}
		seen[p.Name()]++
	}
	// 必须确定性：每次都解析到同一个 provider。
	if len(seen) != 1 {
		t.Fatalf("modelIndex 在 model 冲突时结果非确定（依赖 map 遍历顺序）, 分布=%v", seen)
	}
	// key 字典序 "a" < "b"，"a" 先胜出。
	if _, ok := seen["a"]; !ok {
		t.Fatalf("冲突时应按 key 字典序选择 provider=a, 实际分布=%v", seen)
	}
}

// TestNewService_SkipsNilProvider 验证 nil provider 被跳过且不污染 modelIndex。
func TestNewService_SkipsNilProvider(t *testing.T) {
	pA := &mockVoiceProvider{name: "a", models: []string{"m1", "m2"}}
	svc := NewService(map[string]Provider{
		"a":    pA,
		"nilP": nil,
	}, "a")

	if len(svc.Providers()) != 1 {
		t.Errorf("nil provider 应被跳过, providers=%v", svc.Providers())
	}
	if len(svc.Models()) != 2 {
		t.Errorf("应仅索引有效 provider 的 2 个 model, models=%v", svc.Models())
	}
	// HasProvider 应为 true（有 1 个有效）。
	if !svc.HasProvider() {
		t.Error("有有效 provider 时 HasProvider 应为 true")
	}
}

// TestNewService_DefaultPointsToNilProvider 边界：defaultName 指向被跳过的 nil provider。
// 此时 resolve 回退 default 会从 providers map 取——但该 key 已被跳过，应返回 nil 而非 panic。
func TestNewService_DefaultPointsToNilProvider(t *testing.T) {
	svc := NewService(map[string]Provider{
		"good": &mockVoiceProvider{name: "good", models: []string{"gm"}, result: &Result{Provider: "good"}},
		"bad":  nil,
	}, "bad") // default 指向被跳过的 nil

	// name/model 都不命中 -> 回退 default="bad" -> providers["bad"] 不存在 -> nil。
	p := svc.resolve("", "")
	if p != nil {
		t.Errorf("default 指向被跳过的 provider, resolve 应返回 nil, 实际=%v", p.Name())
	}

	// 经由 Chat 应得到"未找到 provider"错误而非 panic。
	_, err := svc.Chat(context.Background(), "", Request{Text: "hi"})
	if err == nil {
		t.Fatal("default 失效应报错")
	}
}

// TestNewService_DefaultNotRegistered 边界：defaultName 指向一个根本未注册的名字。
func TestNewService_DefaultNotRegistered(t *testing.T) {
	svc := NewService(map[string]Provider{
		"good": &mockVoiceProvider{name: "good", models: []string{"gm"}, result: &Result{Provider: "good"}},
	}, "ghost")

	p := svc.resolve("", "")
	if p != nil {
		t.Errorf("default 指向未注册名字, resolve 应返回 nil, 实际=%v", p.Name())
	}
}

// TestService_ProvidersModels_Snapshot 验证 Providers()/Models() 返回快照不影响内部。
func TestService_ProvidersModels_Snapshot(t *testing.T) {
	pA := &mockVoiceProvider{name: "a", models: []string{"m1"}}
	svc := NewService(map[string]Provider{"a": pA}, "a")

	provs := svc.Providers()
	if len(provs) != 1 {
		t.Fatalf("期望 1 个 provider, 实际=%v", provs)
	}
	// 修改返回的切片不应影响后续调用。
	provs[0] = "tampered"
	if svc.Providers()[0] == "tampered" {
		t.Error("Providers() 返回的切片被外部篡改影响了内部状态")
	}

	models := svc.Models()
	if len(models) != 1 {
		t.Fatalf("期望 1 个 model, 实际=%v", models)
	}
	models[0] = "tampered"
	if svc.Models()[0] == "tampered" {
		t.Error("Models() 返回的切片被外部篡改影响了内部状态")
	}
}

// TestService_EmptyService 全空 Service 的边界行为。
func TestService_EmptyService(t *testing.T) {
	svc := NewService(map[string]Provider{}, "")
	if svc.HasProvider() {
		t.Error("空 Service HasProvider 应为 false")
	}
	if len(svc.Providers()) != 0 {
		t.Errorf("空 Service Providers 应为空, 实际=%v", svc.Providers())
	}
	if len(svc.Models()) != 0 {
		t.Errorf("空 Service Models 应为空, 实际=%v", svc.Models())
	}
}

// TestService_NilServiceConstructed_NilMap 用 nil map 构建。
func TestService_NilMapConstructed(t *testing.T) {
	svc := NewService(nil, "x")
	if svc.HasProvider() {
		t.Error("nil map 构建的 Service HasProvider 应为 false")
	}
	_, err := svc.Chat(context.Background(), "x", Request{Text: "hi"})
	if err == nil {
		t.Fatal("nil map 构建的 Service 调用应报错")
	}
}

// TestService_ResultPassthrough 验证 provider 返回的 *Result（含 nil）被原样透传。
func TestService_ResultPassthrough(t *testing.T) {
	// provider 返回 nil result 且无 error —— Service 应原样透传 (nil, nil)。
	p := &mockVoiceProvider{name: "x", models: []string{"m"}, result: nil, err: nil}
	svc := NewService(map[string]Provider{"x": p}, "x")

	res, err := svc.Chat(context.Background(), "x", Request{Text: "hi"})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if res != nil {
		t.Errorf("provider 返回 nil result, Service 应透传 nil, 实际=%+v", res)
	}
}

// TestService_ContextPropagation 验证 ctx 透传到 provider。
func TestService_ContextPropagation(t *testing.T) {
	type ctxKey string
	key := ctxKey("trace")
	var gotVal any
	p := &ctxCaptureProvider{
		name:   "x",
		models: []string{"m"},
		onChat: func(ctx context.Context) { gotVal = ctx.Value(key) },
	}
	svc := NewService(map[string]Provider{"x": p}, "x")
	ctx := context.WithValue(context.Background(), key, "abc-123")
	_, _ = svc.Chat(ctx, "x", Request{Text: "hi"})
	if gotVal != "abc-123" {
		t.Errorf("ctx 未透传到 provider, 实际=%v", gotVal)
	}
}

// ctxCaptureProvider 捕获 Chat 收到的 ctx。
type ctxCaptureProvider struct {
	name   string
	models []string
	onChat func(ctx context.Context)
}

func (c *ctxCaptureProvider) Name() string              { return c.name }
func (c *ctxCaptureProvider) SupportedModels() []string { return c.models }
func (c *ctxCaptureProvider) Chat(ctx context.Context, _ Request) (*Result, error) {
	if c.onChat != nil {
		c.onChat(ctx)
	}
	return &Result{Provider: c.name}, nil
}

// ----------------------------------------------------------------------------
// 二、并发安全：Service 的读操作（resolve/Chat/Providers/Models/HasProvider）
//   Service 构建后 map 只读，理论上并发读安全。-race 跑实证。
// ----------------------------------------------------------------------------

// TestService_ConcurrentRead 并发读不应触发 data race。
//
// 关键设计：mockVoiceProvider.calls++（service_test.go:21）非并发安全，
// 若多个 goroutine 调用同一 mock 会触发 mock 自身的竞态（与被测代码无关）。
// 为让 -race 仅检测 Service 内部读路径（resolve / map 读 / 切片构造），
// 此处用一个不带可变状态的 readOnlyProvider，避免 mock 污染结果。
func TestService_ConcurrentRead(t *testing.T) {
	pA := &readOnlyProvider{name: "a", models: []string{"m1"}}
	pB := &readOnlyProvider{name: "b", models: []string{"m2"}}
	svc := NewService(map[string]Provider{"a": pA, "b": pB}, "a")

	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_ = svc.HasProvider()
			_ = svc.Providers()
			_ = svc.Models()
			name := "a"
			if i%2 == 0 {
				name = "b"
			}
			// 并发解析（按 name 与按 model 两条路径都覆盖）。
			_, _ = svc.Chat(context.Background(), name, Request{Text: "hi"})
			_, _ = svc.Chat(context.Background(), "", Request{Text: "hi", Model: "m1"})
		}(i)
	}
	wg.Wait()
}

// readOnlyProvider 无可变状态的只读 provider，专供并发读测试，
// 不会引入 mock 自身的竞态，从而让 -race 聚焦于 Service 读路径。
type readOnlyProvider struct {
	name   string
	models []string
}

func (r *readOnlyProvider) Name() string              { return r.name }
func (r *readOnlyProvider) SupportedModels() []string { return r.models }
func (r *readOnlyProvider) Chat(_ context.Context, _ Request) (*Result, error) {
	return &Result{Provider: r.name}, nil
}

// ----------------------------------------------------------------------------
// 三、openAIVoiceChat：基础属性 + API Key 缺失
// ----------------------------------------------------------------------------

// TestOpenAI_BasicProps 验证 Name / SupportedModels / 默认 baseURL。
func TestOpenAI_BasicProps(t *testing.T) {
	p := NewOpenAIVoiceChat("sk-test", "")
	if p.Name() != "openai-gpt4o-audio" {
		t.Errorf("Name 不符, 实际=%s", p.Name())
	}
	models := p.SupportedModels()
	if len(models) != 2 || models[0] != "gpt-4o-audio-preview" {
		t.Errorf("SupportedModels 不符, 实际=%v", models)
	}
	// 默认 baseURL 应回退到官方地址。
	concrete, ok := p.(*openAIVoiceChat)
	if !ok {
		t.Fatal("应可断言为 *openAIVoiceChat")
	}
	if concrete.baseURL != "https://api.openai.com/v1" {
		t.Errorf("默认 baseURL 不符, 实际=%s", concrete.baseURL)
	}
}

// TestOpenAI_BaseURLTrimSlash 验证 baseURL 尾部斜杠被裁剪。
func TestOpenAI_BaseURLTrimSlash(t *testing.T) {
	p := NewOpenAIVoiceChat("sk", "https://example.com/v1///")
	concrete := p.(*openAIVoiceChat)
	if concrete.baseURL != "https://example.com/v1" {
		t.Errorf("尾部斜杠应被裁剪, 实际=%s", concrete.baseURL)
	}
}

// TestOpenAI_NoAPIKey 缺失 API Key 应在发请求前报错。
func TestOpenAI_NoAPIKey(t *testing.T) {
	p := NewOpenAIVoiceChat("", "https://example.com")
	_, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err == nil {
		t.Fatal("缺失 API Key 应报错")
	}
	if !strings.Contains(err.Error(), "API Key") {
		t.Errorf("错误应说明 API Key 缺失, 实际=%v", err)
	}
}

// ----------------------------------------------------------------------------
// 四、openAIVoiceChat：通过 httptest 注入 baseURL 测请求构造 / 响应解析 / 错误路径
//   httpc 是 http.DefaultClient，但 baseURL 可注入指向本地 server，故无需打真实网络。
// ----------------------------------------------------------------------------

// captured 用于在 handler 中捕获请求体便于断言。
type captured struct {
	mu     sync.Mutex
	path   string
	auth   string
	ctype  string
	body   oaChatReq
	rawSet bool
}

func (c *captured) set(r *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.path = r.URL.Path
	c.auth = r.Header.Get("Authorization")
	c.ctype = r.Header.Get("Content-Type")
	raw, _ := io.ReadAll(r.Body)
	_ = json.Unmarshal(raw, &c.body)
	c.rawSet = true
}

// newOpenAIWithServer 构造一个指向 httptest server 的 provider。
func newOpenAIWithServer(t *testing.T, apiKey string, h http.HandlerFunc) (Provider, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	p := NewOpenAIVoiceChat(apiKey, srv.URL)
	return p, srv
}

// TestOpenAI_Chat_RequestConstruction 验证请求构造：路径、Header、modalities、audio config、messages。
func TestOpenAI_Chat_RequestConstruction(t *testing.T) {
	cap := &captured{}
	p, _ := newOpenAIWithServer(t, "sk-abc", func(w http.ResponseWriter, r *http.Request) {
		cap.set(r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hi","audio":{"id":"a1","data":"ZGF0YQ==","transcript":"你好"}}}]}`))
	})

	res, err := p.Chat(context.Background(), Request{
		Model:    "gpt-4o-audio-preview",
		AudioB64: "QUFBQQ==",
		Text:     "讲个笑话",
		Voice:    "echo",
		Format:   "mp3",
		System:   "你是助手",
		History:  []Turn{{Role: "user", Text: "之前"}, {Role: "assistant", Text: "回复"}, {Role: "user", Text: ""}},
	})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if !cap.rawSet {
		t.Fatal("未捕获到请求")
	}
	if cap.path != "/chat/completions" {
		t.Errorf("路径错误, 实际=%s", cap.path)
	}
	if cap.auth != "Bearer sk-abc" {
		t.Errorf("Authorization 错误, 实际=%s", cap.auth)
	}
	if cap.ctype != "application/json" {
		t.Errorf("Content-Type 错误, 实际=%s", cap.ctype)
	}
	if cap.body.Model != "gpt-4o-audio-preview" {
		t.Errorf("model 错误, 实际=%s", cap.body.Model)
	}
	// modalities 必须含 text+audio。
	if len(cap.body.Modalities) != 2 {
		t.Errorf("modalities 应为 [text,audio], 实际=%v", cap.body.Modalities)
	}
	// audio config。
	if cap.body.Audio == nil || cap.body.Audio.Voice != "echo" || cap.body.Audio.Format != "mp3" {
		t.Errorf("audio config 错误, 实际=%+v", cap.body.Audio)
	}
	// messages 结构：system + 2 条有效历史(空 text 被跳过) + 1 条 user = 4。
	if len(cap.body.Messages) != 4 {
		t.Fatalf("期望 4 条消息(system+2历史+user), 实际=%d: %+v", len(cap.body.Messages), cap.body.Messages)
	}
	if cap.body.Messages[0].Role != "system" {
		t.Errorf("首条应为 system, 实际=%s", cap.body.Messages[0].Role)
	}
	// 最后一条 user 应同时含 text + input_audio 两个 part。
	last := cap.body.Messages[len(cap.body.Messages)-1]
	if last.Role != "user" {
		t.Errorf("末条应为 user, 实际=%s", last.Role)
	}
	if len(last.Content) != 2 {
		t.Errorf("user 应含 text+input_audio 两 part, 实际=%d: %+v", len(last.Content), last.Content)
	}
	var hasText, hasAudio bool
	for _, part := range last.Content {
		if part.Type == "text" && part.Text == "讲个笑话" {
			hasText = true
		}
		if part.Type == "input_audio" && part.InputAudio != nil && part.InputAudio.Data == "QUFBQQ==" && part.InputAudio.Format == "mp3" {
			hasAudio = true
		}
	}
	if !hasText || !hasAudio {
		t.Errorf("user part 内容错误 hasText=%v hasAudio=%v: %+v", hasText, hasAudio, last.Content)
	}

	// 响应解析：audio.transcript 优先覆盖 content。
	if res.Transcript != "你好" {
		t.Errorf("transcript 应取 audio.transcript, 实际=%s", res.Transcript)
	}
	if res.ResponseB64 != "ZGF0YQ==" {
		t.Errorf("ResponseB64 应取 audio.data, 实际=%s", res.ResponseB64)
	}
	if res.Format != "mp3" {
		t.Errorf("Format 应回显请求格式, 实际=%s", res.Format)
	}
	if res.Provider != "openai-gpt4o-audio" {
		t.Errorf("Provider 错误, 实际=%s", res.Provider)
	}
	if res.UsageMs < 0 {
		t.Errorf("UsageMs 不应为负, 实际=%d", res.UsageMs)
	}
}

// TestOpenAI_Chat_Defaults 验证缺省 model/voice/format 时回退默认值。
func TestOpenAI_Chat_Defaults(t *testing.T) {
	cap := &captured{}
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		cap.set(r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	})

	res, err := p.Chat(context.Background(), Request{Text: "hi"}) // 不带 model/voice/format
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	cap.mu.Lock()
	defer cap.mu.Unlock()
	if cap.body.Model != "gpt-4o-audio-preview" {
		t.Errorf("model 缺省应回退 gpt-4o-audio-preview, 实际=%s", cap.body.Model)
	}
	if cap.body.Audio.Voice != "alloy" {
		t.Errorf("voice 缺省应为 alloy, 实际=%s", cap.body.Audio.Voice)
	}
	if cap.body.Audio.Format != "wav" {
		t.Errorf("format 缺省应为 wav, 实际=%s", cap.body.Audio.Format)
	}
	// 无 audio 字段时 Result.Transcript 取 content。
	if res.Transcript != "ok" {
		t.Errorf("无 audio 时 transcript 应取 content, 实际=%s", res.Transcript)
	}
	if res.Format != "wav" {
		t.Errorf("Result.Format 应回显 wav, 实际=%s", res.Format)
	}
}

// TestOpenAI_Chat_TextOnly 仅文本输入时 user content 应只含 text part。
func TestOpenAI_Chat_TextOnly(t *testing.T) {
	cap := &captured{}
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		cap.set(r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	_, err := p.Chat(context.Background(), Request{Text: "只打字"})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	cap.mu.Lock()
	defer cap.mu.Unlock()
	last := cap.body.Messages[len(cap.body.Messages)-1]
	if len(last.Content) != 1 || last.Content[0].Type != "text" {
		t.Errorf("仅文本时 user content 应只含 text part, 实际=%+v", last.Content)
	}
}

// TestOpenAI_Chat_AudioOnly 仅音频输入时 user content 应只含 input_audio part。
func TestOpenAI_Chat_AudioOnly(t *testing.T) {
	cap := &captured{}
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		cap.set(r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	_, err := p.Chat(context.Background(), Request{AudioB64: "QUFBQQ=="})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	cap.mu.Lock()
	defer cap.mu.Unlock()
	last := cap.body.Messages[len(cap.body.Messages)-1]
	if len(last.Content) != 1 || last.Content[0].Type != "input_audio" {
		t.Errorf("仅音频时 user content 应只含 input_audio part, 实际=%+v", last.Content)
	}
}

// TestOpenAI_Chat_EmptyUserContentBug 验证 provider 层对 audio/text 皆空的输入自校验。
// 即使绕过 Service 直接调用 provider.Chat，全空输入也应在发请求前报错，
// 而不是构造一条 content 为空数组的 user 消息并发出请求。
// 回归: provider.Chat 不做 audio/text 皆空校验, 会发出 content 为空的 user 消息。
func TestOpenAI_Chat_EmptyUserContentBug(t *testing.T) {
	var reqSent bool
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		reqSent = true
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	// 直接调用 provider，绕过 Service 校验：全空输入应被 provider 自身拦截。
	_, err := p.Chat(context.Background(), Request{}) // 全空
	if err == nil {
		t.Fatal("provider.Chat 全空输入应报错, 实际未报错")
	}
	if !strings.Contains(err.Error(), "至少需要一个") {
		t.Errorf("错误应说明 audio/text 至少需要一个, 实际=%v", err)
	}
	if reqSent {
		t.Error("空输入校验失败时不应向上游发出请求")
	}
}

// TestOpenAI_Chat_HTTPError4xxWithBody 非 200 且响应体含 error.message。
func TestOpenAI_Chat_HTTPError4xxWithBody(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid audio format"}}`))
	})
	_, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err == nil {
		t.Fatal("4xx 应报错")
	}
	if !strings.Contains(err.Error(), "invalid audio format") {
		t.Errorf("错误应含上游 message, 实际=%v", err)
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("错误应含状态码 400, 实际=%v", err)
	}
}

// TestOpenAI_Chat_HTTPError5xxNoBody 非 200 且响应体无法解析出 error。
func TestOpenAI_Chat_HTTPError5xxNoBody(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`gateway timeout html`))
	})
	_, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err == nil {
		t.Fatal("5xx 应报错")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("错误应含状态码 500, 实际=%v", err)
	}
}

// TestOpenAI_Chat_HTTP200WithErrorField 200 状态但 body 含 error 字段。
func TestOpenAI_Chat_HTTP200WithErrorField(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	})
	_, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err == nil {
		t.Fatal("body 含 error 字段应报错")
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("错误应含 error.message, 实际=%v", err)
	}
}

// TestOpenAI_Chat_InvalidJSON 200 状态但 body 非法 JSON。
func TestOpenAI_Chat_InvalidJSON(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{not json`))
	})
	_, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err == nil {
		t.Fatal("非法 JSON 应报错")
	}
	if !strings.Contains(err.Error(), "decode response") {
		t.Errorf("错误应为 decode 错误, 实际=%v", err)
	}
}

// TestOpenAI_Chat_EmptyChoices 200 状态但 choices 为空。
func TestOpenAI_Chat_EmptyChoices(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[]}`))
	})
	_, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err == nil {
		t.Fatal("空 choices 应报错")
	}
	if !strings.Contains(err.Error(), "无回应") {
		t.Errorf("错误应说明无回应, 实际=%v", err)
	}
}

// TestOpenAI_Chat_AudioWithoutTranscript 音频存在但 transcript 为空时，保留 content 作为 transcript。
func TestOpenAI_Chat_AudioWithoutTranscript(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		// audio.transcript 为空，content 有值。
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"内容兜底","audio":{"id":"a1","data":"ZGF0YQ==","transcript":""}}}]}`))
	})
	res, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if res.Transcript != "内容兜底" {
		t.Errorf("audio.transcript 为空时应保留 content, 实际=%s", res.Transcript)
	}
	if res.ResponseB64 != "ZGF0YQ==" {
		t.Errorf("ResponseB64 应取 audio.data, 实际=%s", res.ResponseB64)
	}
}

// TestOpenAI_Chat_ContextCanceled ctx 已取消时应在发请求时报错。
func TestOpenAI_Chat_ContextCanceled(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 立即取消
	_, err := p.Chat(ctx, Request{Text: "hi"})
	if err == nil {
		t.Fatal("已取消的 ctx 应报错")
	}
}

// TestOpenAI_Chat_ServerHang 验证 server 阻塞时 ctx 超时能正确中断。
// 用一个 50ms 后才返回的 handler + 短超时 ctx。
func TestOpenAI_Chat_ServerHang(t *testing.T) {
	release := make(chan struct{})
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		<-release // 阻塞直到测试结束
	})
	defer close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := p.Chat(ctx, Request{Text: "hi"})
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("server 阻塞 + ctx 超时应报错")
	}
	// 应在远小于 provider 内置 90s 超时内返回（受外部 ctx 控制）。
	if elapsed > 5*time.Second {
		t.Errorf("ctx 超时未生效, 耗时=%v", elapsed)
	}
}

// TestOpenAI_Chat_UnicodeAndLongInput Unicode + 超长输入应正确编码进请求体。
func TestOpenAI_Chat_UnicodeAndLongInput(t *testing.T) {
	cap := &captured{}
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		cap.set(r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	})
	longText := strings.Repeat("你好🌍emoji🎵", 5000) // 超长 + emoji + 中文
	_, err := p.Chat(context.Background(), Request{Text: longText})
	if err != nil {
		t.Fatalf("超长 Unicode 输入不应报错: %v", err)
	}
	cap.mu.Lock()
	defer cap.mu.Unlock()
	last := cap.body.Messages[len(cap.body.Messages)-1]
	if len(last.Content) == 0 || last.Content[0].Text != longText {
		t.Error("超长 Unicode 文本未被正确编码进请求体")
	}
}

// TestOpenAI_SupportedModels_NotShared 验证 SupportedModels 做了防御性拷贝，
// 外部修改返回的切片不会污染 provider 内部 models。
// 回归: SupportedModels 返回内部切片引用导致外部可污染 provider 内部状态。
func TestOpenAI_SupportedModels_NotShared(t *testing.T) {
	p := NewOpenAIVoiceChat("sk", "")
	models := p.SupportedModels()
	orig := models[0]
	models[0] = "tampered"
	// 再次取，若被污染说明返回的是内部切片引用（缺陷）。
	again := p.SupportedModels()
	if again[0] == "tampered" {
		t.Fatalf("SupportedModels 返回内部切片引用, 外部修改污染了 provider 内部状态 (orig=%s)", orig)
	}
	if again[0] != orig {
		t.Fatalf("SupportedModels 内容异常, 期望=%s 实际=%s", orig, again[0])
	}
}

// ----------------------------------------------------------------------------
// 五、补充：响应解析的剩余边界 + 文档承诺未闭环
// ----------------------------------------------------------------------------

// TestOpenAI_Chat_ContentNull 响应 content 为 JSON null。
// OpenAI audio 模式下 message.content 经常为 null，仅 audio.transcript 有值。
// 此处验证 content=null 不会破坏解析，且 transcript 取自 audio.transcript。
func TestOpenAI_Chat_ContentNull(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		// content 显式为 null，audio.transcript 提供文本。
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":null,"audio":{"id":"a1","data":"ZGF0YQ==","transcript":"语音回应"}}}]}`))
	})
	res, err := p.Chat(context.Background(), Request{AudioB64: "QUFBQQ=="})
	if err != nil {
		t.Fatalf("content=null 不应破坏解析: %v", err)
	}
	if res.Transcript != "语音回应" {
		t.Errorf("transcript 应取 audio.transcript, 实际=%s", res.Transcript)
	}
	if res.ResponseB64 != "ZGF0YQ==" {
		t.Errorf("ResponseB64 应取 audio.data, 实际=%s", res.ResponseB64)
	}
}

// TestOpenAI_Chat_NoContentNoAudio 响应既无 content 也无 audio。
// 这是一个未闭环点：当上游返回一条空 message（无 content 且无 audio）时，
// Chat 不报错，返回 Transcript="" 且 ResponseB64=""，调用方无法区分
// "模型确实没说话" 与 "解析丢了内容"。记录该静默退化行为。
func TestOpenAI_Chat_NoContentNoAudio(t *testing.T) {
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{}}]}`))
	})
	res, err := p.Chat(context.Background(), Request{Text: "hi"})
	if err != nil {
		t.Fatalf("空 message 当前实现不报错: %v", err)
	}
	if res.Transcript != "" || res.ResponseB64 != "" {
		t.Errorf("空 message 应得到空结果, 实际 transcript=%q audio=%q", res.Transcript, res.ResponseB64)
	}
	t.Logf("未闭环: 上游返回空 message(无 content 无 audio) 时静默返回空 Result, 不报错也无信号")
}

// TestOpenAI_Chat_MarshalAlwaysSucceeds 记录 openai.go:148 处
// `payload, _ := json.Marshal(body)` 丢弃了 marshal 错误。
// Request 全部为可序列化的基本类型(string/slice/struct)，json.Marshal 永不返回错误，
// 因此该忽略在当前结构下安全；但若未来 Request 引入不可序列化字段(如 chan/func)，
// 错误将被静默吞掉并发出空 body 请求。这是一处防御性缺口，本测试固化当前安全前提。
func TestOpenAI_Chat_MarshalAlwaysSucceeds(t *testing.T) {
	cap := &captured{}
	p, _ := newOpenAIWithServer(t, "sk", func(w http.ResponseWriter, r *http.Request) {
		cap.set(r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	})
	// 含各种特殊字符/Unicode 的输入，确认仍能 marshal 成有效 body。
	_, err := p.Chat(context.Background(), Request{
		Text:   "引号\"反斜杠\\换行\n制表\t中文🎵",
		System: "<script>&特殊字符</script>",
	})
	if err != nil {
		t.Fatalf("特殊字符输入不应导致 marshal/请求失败: %v", err)
	}
	cap.mu.Lock()
	defer cap.mu.Unlock()
	if !cap.rawSet || cap.body.Model == "" {
		t.Error("请求体应被正确构造并发出(非空 body)")
	}
}

// TestVoicechat_ZhipuGLM4Voice_NotImplemented 文档/功能未闭环修复验证。
// 此前 voicechat.go 包注释声称支持"智谱 glm-4-voice / Anthropic / Gemini"，
// 但本包仅实现了 openAIVoiceChat —— 文档与实现不一致。
//
// 修复方式（最小闭环）：用 BuiltinModels() 作为"内置支持模型"的唯一权威来源，
// 并据此修正包注释，使文档只声明真正落地的模型；OpenAI Provider 也复用同一来源，
// 杜绝清单漂移。本测试把"文档不得声称未实现模型为内置支持"固化为可回归契约。
// 回归: 包文档声称支持 glm-4-voice / Anthropic / Gemini 但仅实现 OpenAI。
func TestVoicechat_ZhipuGLM4Voice_NotImplemented(t *testing.T) {
	builtin := BuiltinModels()
	if len(builtin) == 0 {
		t.Fatal("BuiltinModels 不应为空，应至少包含 OpenAI 内置模型")
	}

	// 内置模型集合中不得出现任何未实现的 Provider 模型。
	unimplemented := []string{"glm-4-voice", "claude-voice", "gemini-live"}
	builtinSet := map[string]bool{}
	for _, m := range builtin {
		builtinSet[m] = true
		if !strings.HasPrefix(m, "gpt-4o") {
			t.Errorf("BuiltinModels 含非内置实现的模型 %q（仅 OpenAI gpt-4o* 系列已落地）", m)
		}
	}
	for _, m := range unimplemented {
		if builtinSet[m] {
			t.Fatalf("文档/实现不一致: 声称内置支持未实现的模型 %q", m)
		}
	}

	// 唯一内置 Provider（OpenAI）原生支持的模型必须与权威来源完全一致，杜绝漂移。
	openai := NewOpenAIVoiceChat("sk", "")
	got := openai.SupportedModels()
	if len(got) != len(builtin) {
		t.Fatalf("OpenAI SupportedModels 与 BuiltinModels 数量不一致: got=%v builtin=%v", got, builtin)
	}
	for i := range builtin {
		if got[i] != builtin[i] {
			t.Fatalf("OpenAI 内置模型清单与权威来源漂移: got=%v builtin=%v", got, builtin)
		}
	}

	// glm-4-voice 不应被任何内置 Provider 声明支持。
	svc := NewService(map[string]Provider{"openai": openai}, "openai")
	for _, m := range svc.Models() {
		if m == "glm-4-voice" {
			t.Fatalf("意外：内置 Provider 声称支持 glm-4-voice (model=%s)", m)
		}
	}
}
