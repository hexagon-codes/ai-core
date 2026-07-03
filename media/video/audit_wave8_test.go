package video

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// ============================================================================
// Wave8 审计测试：覆盖 Service 路由、TaskID 包装/解析、zhipu Provider 的
// 请求构造 / 响应解析 / 错误路径 / 状态机映射，以及 SubmitAndWait 同步包装。
//
// 云 Provider 测试策略：用 httptest.Server 注入 baseURL，httpc 为
// http.DefaultClient 默认可访问 httptest URL，因此无需打真实网络。
// ============================================================================

// -------------------------- NewService 构建边界 --------------------------

// TestNewService_NilProviderSkipped nil Provider 应被跳过，不计入 providers/modelIndex。
func TestNewService_NilProviderSkipped(t *testing.T) {
	good := &mockVideoProvider{name: "good", models: []string{"m1"}}
	svc := NewService(map[string]Provider{
		"good": good,
		"bad":  nil, // nil 应被跳过
	}, "good")

	if got := len(svc.Providers()); got != 1 {
		t.Errorf("nil provider 应被跳过, 期望 1 个 provider, 实际=%d (%v)", got, svc.Providers())
	}
	if !svc.HasProvider() {
		t.Error("有有效 provider 时 HasProvider 应为 true")
	}
}

// TestNewService_NilMap 完全 nil 输入应得到空但可用的 Service。
func TestNewService_NilMap(t *testing.T) {
	svc := NewService(nil, "")
	if svc.HasProvider() {
		t.Error("空 Service HasProvider 应为 false")
	}
	if len(svc.Providers()) != 0 || len(svc.Models()) != 0 {
		t.Errorf("空 Service Providers/Models 应为空, 实际 providers=%v models=%v", svc.Providers(), svc.Models())
	}
	// resolveByTaskID/Poll 在空 Service 上应安全报错而非 panic
	if _, err := svc.Poll(context.Background(), "x::y"); err == nil {
		t.Error("空 Service Poll 应报未知 provider 错误")
	}
}

// TestNewService_ModelIndexDuplicateModel 同一 model 被多个 Provider 声明时，
// 路由必须是【确定性】的——不能因 map 迭代顺序随机而在不同构造间漂移。
//
// 回归: 共享 model 被多 Provider 声明时路由非确定性（map 迭代顺序随机）。
// 修复采用 Provider 名字典序作为冲突时的稳定优先级（alpha < beta），
// 因此无论构造多少次，"shared" 必须恒定解析到 alpha。
func TestNewService_ModelIndexDuplicateModel(t *testing.T) {
	pA := &mockVideoProvider{name: "alpha", models: []string{"shared"}, submitID: "A"}
	pB := &mockVideoProvider{name: "beta", models: []string{"shared"}, submitID: "B"}

	// 多次重建 Service，路由结果必须恒定，否则即为非确定性缺陷。
	const iterations = 200
	for i := 0; i < iterations; i++ {
		svc := NewService(map[string]Provider{"alpha": pA, "beta": pB}, "")
		got := svc.resolveProvider("", "shared")
		if got == nil {
			t.Fatalf("第 %d 次: 共享 model 应能解析到某个 provider", i)
		}
		// 字典序优先 -> 恒定 alpha。任何一次解析到 beta 都说明路由非确定。
		if got.Name() != "alpha" {
			t.Fatalf("第 %d 次: 共享 model 路由非确定性, 期望恒定=alpha 实际=%s", i, got.Name())
		}
		if len(svc.Models()) != 1 {
			t.Fatalf("第 %d 次: 去重后 model 数应为 1, 实际=%d (%v)", i, len(svc.Models()), svc.Models())
		}
	}
}

// -------------------------- resolveProvider 优先级 / 边界 --------------------------

// TestResolveProvider_UnknownNameFallsThrough 指定了不存在的 name 时，
// 应继续按 model -> default 回退，而非直接返回 nil。
func TestResolveProvider_UnknownNameFallsThrough(t *testing.T) {
	p := &mockVideoProvider{name: "real", models: []string{"m1"}, submitID: "r"}
	svc := NewService(map[string]Provider{"real": p}, "real")

	// name 不存在，但 model 命中 -> 应回退到 model 命中的 provider
	got := svc.resolveProvider("ghost", "m1")
	if got == nil || got.Name() != "real" {
		t.Errorf("未知 name 应回退 model 命中, 实际=%v", got)
	}

	// name 和 model 都不命中 -> 回退 default
	got = svc.resolveProvider("ghost", "ghost-model")
	if got == nil || got.Name() != "real" {
		t.Errorf("name/model 都不命中应回退 default, 实际=%v", got)
	}
}

// TestResolveProvider_NoDefaultNoMatch 无 default 且无命中 -> nil。
func TestResolveProvider_NoDefaultNoMatch(t *testing.T) {
	p := &mockVideoProvider{name: "real", models: []string{"m1"}}
	svc := NewService(map[string]Provider{"real": p}, "") // 无 default
	if got := svc.resolveProvider("ghost", "ghost-model"); got != nil {
		t.Errorf("无 default 且无命中应返回 nil, 实际=%v", got)
	}
}

// TestResolveProvider_DefaultPointsToMissing default 指向不存在的 key 时返回 nil（map 读零值）。
func TestResolveProvider_DefaultPointsToMissing(t *testing.T) {
	p := &mockVideoProvider{name: "real", models: []string{"m1"}}
	svc := NewService(map[string]Provider{"real": p}, "nonexistent")
	if got := svc.resolveProvider("", ""); got != nil {
		t.Errorf("default 指向缺失 key 应返回 nil, 实际=%v", got)
	}
}

// -------------------------- resolveByTaskID 边界 --------------------------

// TestResolveByTaskID_Edges 覆盖空 provider 前缀、空 raw、多重 :: 等边界。
func TestResolveByTaskID_Edges(t *testing.T) {
	p := &mockVideoProvider{name: "zhipu", models: []string{"m"}}
	svc := NewService(map[string]Provider{"zhipu": p}, "zhipu")

	tests := []struct {
		name    string
		taskID  string
		wantErr bool
		wantRaw string
	}{
		{"正常", "zhipu::raw-1", false, "raw-1"},
		{"无前缀", "raw-1", true, ""},
		{"空前缀", "::raw-1", true, ""},            // name="" 查不到 provider
		{"空raw", "zhipu::", false, ""},          // raw 为空但 provider 合法（degenerate, 见 findings）
		{"多重双冒号", "zhipu::a::b", false, "a::b"}, // strings.Cut 只切第一个
		{"未知provider", "ghost::raw", true, ""},
		{"完全空串", "", true, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, raw, err := svc.resolveByTaskID(tt.taskID)
			if (err != nil) != tt.wantErr {
				t.Fatalf("taskID=%q wantErr=%v 实际 err=%v", tt.taskID, tt.wantErr, err)
			}
			if err == nil && raw != tt.wantRaw {
				t.Errorf("taskID=%q 期望 raw=%q 实际=%q", tt.taskID, tt.wantRaw, raw)
			}
		})
	}
}

// -------------------------- Submit/Poll 边界 --------------------------

// TestSubmit_WhitespacePromptRejected 仅空白字符的 prompt 是退化输入，
// 必须在校验层被拒，绝不能绕过校验向上游发请求（浪费配额、可能触发风控）。
//
// 回归: Submit 参数校验过松——仅空白字符的 prompt 可绕过校验并发起上游请求。
// 覆盖 ASCII 空格、Tab、换行、全角空格(U+3000) 等常见空白退化输入。
func TestSubmit_WhitespacePromptRejected(t *testing.T) {
	for _, ws := range []string{"   ", "\t", "\n", " \t\n ", "　", " 　\t"} {
		p := &mockVideoProvider{name: "x", models: []string{"m"}, submitID: "r"}
		svc := NewService(map[string]Provider{"x": p}, "x")

		// 纯空白 prompt 且无 image_url -> 必须报错且不调用上游
		_, err := svc.Submit(context.Background(), "", Request{Prompt: ws, Model: "m"})
		if err == nil {
			t.Errorf("纯空白 prompt(%q) 应被校验拒绝, 实际通过", ws)
		}
		if p.submitCall != 0 {
			t.Errorf("纯空白 prompt(%q) 校验失败时不应调用上游 Submit, 实际调用=%d", ws, p.submitCall)
		}
	}
}

// TestSubmit_UnicodePromptPreserved Unicode/emoji prompt 应原样进入 provider。
func TestSubmit_UnicodePromptPreserved(t *testing.T) {
	var captured Request
	p := &capturingProvider{name: "x", models: []string{"m"}, submitID: "r", capture: &captured}
	svc := NewService(map[string]Provider{"x": p}, "x")

	prompt := "一只猫在月球上跳舞 🐱🌙 héllo"
	id, err := svc.Submit(context.Background(), "", Request{Prompt: prompt, Model: "m"})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if captured.Prompt != prompt {
		t.Errorf("Unicode prompt 应原样透传, 期望=%q 实际=%q", prompt, captured.Prompt)
	}
	if !strings.HasPrefix(id, "x::") {
		t.Errorf("ID 应带前缀, 实际=%s", id)
	}
}

// TestPoll_RawTaskIDUsesEmptyTaskIDFromProvider Service.Poll 应把带前缀 ID 覆盖回去，
// 即使 provider 返回的 TaskStatus.TaskID 为空。
func TestPoll_OverridesTaskIDEvenIfProviderSetsOwn(t *testing.T) {
	p := &mockVideoProvider{
		name:   "zhipu",
		models: []string{"m"},
		status: TaskStatus{TaskID: "provider-set-id", Status: "running"},
	}
	svc := NewService(map[string]Provider{"zhipu": p}, "zhipu")

	st, err := svc.Poll(context.Background(), "zhipu::raw-9")
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if st.TaskID != "zhipu::raw-9" {
		t.Errorf("Service 应覆盖 TaskID 为带前缀 ID, 实际=%s", st.TaskID)
	}
}

// -------------------------- Providers/Models 列表 --------------------------

// TestProvidersAndModels_NoDuplicateRegistration 多 provider 列表正确性。
func TestProvidersAndModels(t *testing.T) {
	pA := &mockVideoProvider{name: "alpha", models: []string{"m1", "m2"}}
	pB := &mockVideoProvider{name: "beta", models: []string{"m3"}}
	svc := NewService(map[string]Provider{"alpha": pA, "beta": pB}, "alpha")

	if len(svc.Providers()) != 2 {
		t.Errorf("应有 2 个 provider, 实际=%v", svc.Providers())
	}
	if len(svc.Models()) != 3 {
		t.Errorf("应有 3 个 model, 实际=%v", svc.Models())
	}
}

// ============================================================================
//                          zhipu Provider 测试（httptest 注入）
// ============================================================================

// newZhipuWithServer 用 httptest.Server 的 URL 注入 baseURL，
// 返回的具体类型便于设置 apiKey 等内部字段；httpc 默认即可访问 httptest。
func newZhipuWithServer(apiKey, baseURL string) *zhipuCogVideoX {
	p := NewZhipuCogVideoX(apiKey, baseURL).(*zhipuCogVideoX)
	return p
}

// TestZhipu_HTTPClientInjectable 验证 zhipu Provider 提供 httpClient 注入点。
//
// 回归: zhipuCogVideoX 固定使用 http.DefaultClient，无 httpClient 注入点
// （共享全局 client、超时不可控）。修复后应通过 WithHTTPClient 注入自定义
// client，且默认仍可用（不破坏既有 NewZhipuCogVideoX 签名）。
func TestZhipu_HTTPClientInjectable(t *testing.T) {
	// 1) 默认应得到非 nil 的 client（不再硬编码裸全局单例的隐式依赖）。
	def := NewZhipuCogVideoX("k", "").(*zhipuCogVideoX)
	if def.httpc == nil {
		t.Fatal("默认 httpc 不应为 nil")
	}

	// 2) 注入的自定义 client 必须被实际采用（用带超时的 client 区分）。
	custom := &http.Client{Timeout: 7 * time.Second}
	p := NewZhipuCogVideoX("k", "", WithHTTPClient(custom)).(*zhipuCogVideoX)
	if p.httpc != custom {
		t.Errorf("WithHTTPClient 注入的 client 未生效, 期望=%p 实际=%p", custom, p.httpc)
	}

	// 3) 传入 nil client 应被忽略，回退到默认非 nil client（防御性）。
	p2 := NewZhipuCogVideoX("k", "", WithHTTPClient(nil)).(*zhipuCogVideoX)
	if p2.httpc == nil {
		t.Error("WithHTTPClient(nil) 应被忽略并回退默认 client, 实际为 nil")
	}
}

// TestZhipu_NewDefaults 默认值与接口断言。
func TestZhipu_NewDefaults(t *testing.T) {
	p := newZhipuWithServer("k", "")
	if p.baseURL != defaultZhipuVideoBase {
		t.Errorf("空 baseURL 应回退默认, 实际=%s", p.baseURL)
	}
	if p.Name() != "zhipu-cogvideox" {
		t.Errorf("Name 不符, 实际=%s", p.Name())
	}
	if len(p.SupportedModels()) == 0 {
		t.Error("应有支持模型")
	}
	// 尾部斜杠应被裁剪
	p2 := newZhipuWithServer("k", "http://x.test/api/")
	if strings.HasSuffix(p2.baseURL, "/") {
		t.Errorf("baseURL 尾斜杠应裁剪, 实际=%s", p2.baseURL)
	}
}

// TestZhipu_Submit_NoAPIKey 未配置 key 应立即报错且不发请求。
func TestZhipu_Submit_NoAPIKey(t *testing.T) {
	var hit bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
	}))
	defer srv.Close()
	p := newZhipuWithServer("", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "hi", Model: "cogvideox-2"})
	if err == nil || !strings.Contains(err.Error(), "API Key") {
		t.Errorf("无 key 应报 API Key 错误, 实际=%v", err)
	}
	if hit {
		t.Error("无 key 不应发起 HTTP 请求")
	}
}

func TestZhipu_NetworkPolicyBlocksPrivateBaseURL(t *testing.T) {
	p := NewZhipuCogVideoX("secret-key", "http://127.0.0.1:1",
		WithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true}),
	)
	_, err := p.Submit(context.Background(), Request{Prompt: "hi", Model: "cogvideox-2"})
	if err == nil {
		t.Fatal("NetworkPolicy should block private baseURL")
	}
	if !errors.Is(err, llm.ErrNetworkPolicy) {
		t.Fatalf("expected ErrNetworkPolicy, got %v", err)
	}
}

// TestZhipu_Submit_RequestConstruction 验证 method/path/header/body 构造。
func TestZhipu_Submit_RequestConstruction(t *testing.T) {
	var (
		gotMethod string
		gotPath   string
		gotAuth   string
		gotCT     string
		gotBody   zhipuSubmitReq
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotCT = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"task-xyz","task_status":"PROCESSING"}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("secret-key", srv.URL)

	id, err := p.Submit(context.Background(), Request{
		Model: "cogvideox-2", Prompt: "a dog", Quality: "quality",
		WithAudio: true, Size: "1280x720", FPS: 30, Duration: 5, UserID: "u1",
	})
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if id != "task-xyz" {
		t.Errorf("应返回上游 task id, 实际=%s", id)
	}
	if gotMethod != http.MethodPost {
		t.Errorf("应为 POST, 实际=%s", gotMethod)
	}
	if !strings.HasSuffix(gotPath, "/videos/generations") {
		t.Errorf("path 不符, 实际=%s", gotPath)
	}
	if gotAuth != "Bearer secret-key" {
		t.Errorf("Authorization 不符, 实际=%s", gotAuth)
	}
	if gotCT != "application/json" {
		t.Errorf("Content-Type 不符, 实际=%s", gotCT)
	}
	if gotBody.Model != "cogvideox-2" || gotBody.Prompt != "a dog" || !gotBody.WithAudio ||
		gotBody.FPS != 30 || gotBody.Duration != 5 || gotBody.UserID != "u1" || gotBody.Size != "1280x720" {
		t.Errorf("请求体字段映射不全, 实际=%+v", gotBody)
	}
}

// TestZhipu_Submit_DefaultModelFilled Model 为空时应填入 models[0]。
func TestZhipu_Submit_DefaultModelFilled(t *testing.T) {
	var gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b zhipuSubmitReq
		_ = json.NewDecoder(r.Body).Decode(&b)
		gotModel = b.Model
		_, _ = w.Write([]byte(`{"id":"t1","task_status":"PROCESSING"}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "x"}) // Model 留空
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if gotModel != "cogvideox-2" { // models[0]
		t.Errorf("空 Model 应填默认 models[0]=cogvideox-2, 实际=%s", gotModel)
	}
}

// TestZhipu_Submit_HTTPError_WithErrorBody 非200且 body 含 error 应提取 message。
func TestZhipu_Submit_HTTPError_WithErrorBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"code":"1002","message":"余额不足"}}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil {
		t.Fatal("非200应报错")
	}
	if !strings.Contains(err.Error(), "余额不足") || !strings.Contains(err.Error(), "400") {
		t.Errorf("错误应含上游 message 与状态码, 实际=%v", err)
	}
}

// TestZhipu_Submit_HTTPError_PlainBody 非200且 body 非结构化 error 时回退原始 body。
func TestZhipu_Submit_HTTPError_PlainBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`Internal Server Error`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil || !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "Internal Server Error") {
		t.Errorf("非结构化错误应回退原始 body, 实际=%v", err)
	}
}

// TestZhipu_Submit_Success_ErrorInBody 200 但 body 含 error 字段应报错。
func TestZhipu_Submit_Success_ErrorInBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"error":{"code":"x","message":"内容违规"}}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil || !strings.Contains(err.Error(), "内容违规") {
		t.Errorf("200 含 error 应报错, 实际=%v", err)
	}
}

// TestZhipu_Submit_Success_NoID 200 但无 id 应报错。
func TestZhipu_Submit_Success_NoID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"task_status":"PROCESSING"}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil || !strings.Contains(err.Error(), "task ID") {
		t.Errorf("无 id 应报错, 实际=%v", err)
	}
}

// TestZhipu_Submit_MalformedJSON 200 但 body 非法 JSON 应报 decode 错误。
func TestZhipu_Submit_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{not json`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Errorf("非法 JSON 应报 decode 错误, 实际=%v", err)
	}
}

// TestZhipu_Submit_ContextCanceled 已取消的 ctx 应导致请求失败。
func TestZhipu_Submit_ContextCanceled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"t","task_status":"PROCESSING"}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 立即取消
	_, err := p.Submit(ctx, Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil {
		t.Error("已取消 ctx 应报错")
	}
}

// -------------------------- zhipu Poll --------------------------

// TestZhipu_Poll_NoAPIKey 无 key 应报错。
func TestZhipu_Poll_NoAPIKey(t *testing.T) {
	p := newZhipuWithServer("", "http://x.test")
	_, err := p.Poll(context.Background(), "task-1")
	if err == nil || !strings.Contains(err.Error(), "API Key") {
		t.Errorf("无 key 应报错, 实际=%v", err)
	}
}

// TestZhipu_Poll_StateMachine 状态映射表驱动：覆盖 SUCCESS/FAIL/FAILED/PROCESSING/未知。
func TestZhipu_Poll_StateMachine(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantStatus string
		wantDone   bool
		wantVideo  string
		wantCover  string
		wantErrMsg string
	}{
		{
			name:       "SUCCESS有结果",
			body:       `{"model":"cogvideox-2","task_status":"SUCCESS","video_result":[{"url":"http://v/x.mp4","cover_image_url":"http://c/x.jpg"}]}`,
			wantStatus: "success", wantDone: true, wantVideo: "http://v/x.mp4", wantCover: "http://c/x.jpg",
		},
		{
			name:       "SUCCESS空结果",
			body:       `{"task_status":"SUCCESS","video_result":[]}`,
			wantStatus: "success", wantDone: true,
		},
		{
			name:       "FAIL带error",
			body:       `{"task_status":"FAIL","error":{"code":"x","message":"生成失败原因"}}`,
			wantStatus: "failed", wantDone: true, wantErrMsg: "生成失败原因",
		},
		{
			name:       "FAILED无error默认文案",
			body:       `{"task_status":"FAILED"}`,
			wantStatus: "failed", wantDone: true, wantErrMsg: "任务失败",
		},
		{
			name:       "PROCESSING非终态",
			body:       `{"task_status":"PROCESSING"}`,
			wantStatus: "processing", wantDone: false,
		},
		{
			name:       "未知状态小写透传非终态",
			body:       `{"task_status":"WEIRD"}`,
			wantStatus: "weird", wantDone: false,
		},
		{
			name:       "空状态",
			body:       `{}`,
			wantStatus: "", wantDone: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()
			p := newZhipuWithServer("k", srv.URL)

			st, err := p.Poll(context.Background(), "task-1")
			if err != nil {
				t.Fatalf("未预期错误: %v", err)
			}
			if st.Status != tt.wantStatus {
				t.Errorf("Status 期望=%q 实际=%q", tt.wantStatus, st.Status)
			}
			if st.Done != tt.wantDone {
				t.Errorf("Done 期望=%v 实际=%v", tt.wantDone, st.Done)
			}
			if st.VideoURL != tt.wantVideo {
				t.Errorf("VideoURL 期望=%q 实际=%q", tt.wantVideo, st.VideoURL)
			}
			if st.CoverURL != tt.wantCover {
				t.Errorf("CoverURL 期望=%q 实际=%q", tt.wantCover, st.CoverURL)
			}
			if st.Error != tt.wantErrMsg {
				t.Errorf("Error 期望=%q 实际=%q", tt.wantErrMsg, st.Error)
			}
			if st.Provider != "zhipu-cogvideox" {
				t.Errorf("Provider 应填充, 实际=%q", st.Provider)
			}
		})
	}
}

// TestZhipu_Poll_RequestConstruction 验证 GET path + Authorization。
func TestZhipu_Poll_RequestConstruction(t *testing.T) {
	var gotMethod, gotPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"task_status":"PROCESSING"}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k2", srv.URL)

	_, err := p.Poll(context.Background(), "the-task-id")
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if gotMethod != http.MethodGet {
		t.Errorf("Poll 应为 GET, 实际=%s", gotMethod)
	}
	if !strings.HasSuffix(gotPath, "/async-result/the-task-id") {
		t.Errorf("Poll path 不符, 实际=%s", gotPath)
	}
	if gotAuth != "Bearer k2" {
		t.Errorf("Poll Authorization 不符, 实际=%s", gotAuth)
	}
}

// TestZhipu_Poll_HTTPError 非200错误路径。
func TestZhipu_Poll_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"任务不存在"}}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Poll(context.Background(), "task-x")
	if err == nil || !strings.Contains(err.Error(), "任务不存在") || !strings.Contains(err.Error(), "404") {
		t.Errorf("Poll 非200应报含 message+状态码, 实际=%v", err)
	}
}

// TestZhipu_Poll_MalformedJSON poll 200 非法 JSON 应报 decode 错误。
func TestZhipu_Poll_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`}{`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Poll(context.Background(), "task-x")
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Errorf("非法 JSON 应报 decode 错误, 实际=%v", err)
	}
}

// TestZhipu_Poll_MultipleVideoResults 只取首个结果（实现取 [0]）。
func TestZhipu_Poll_MultipleVideoResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"task_status":"SUCCESS","video_result":[{"url":"first"},{"url":"second"}]}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	st, err := p.Poll(context.Background(), "task-x")
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if st.VideoURL != "first" {
		t.Errorf("应取首个 video_result, 实际=%s", st.VideoURL)
	}
}

// ============================================================================
//                          SubmitAndWait 同步包装
// ============================================================================

// TestSubmitAndWait_SubmitError Submit 失败应直接返回错误，不轮询。
func TestSubmitAndWait_SubmitError(t *testing.T) {
	p := &mockVideoProvider{name: "x", models: []string{"m"}}
	svc := NewService(map[string]Provider{"x": p}, "x")

	// prompt+image 皆空 -> Submit 在校验层失败
	_, err := svc.SubmitAndWait(context.Background(), "", Request{}, time.Millisecond)
	if err == nil {
		t.Fatal("Submit 校验失败应返回错误")
	}
	if p.pollCall != 0 {
		t.Error("Submit 失败不应轮询")
	}
}

// TestSubmitAndWait_PollUntilDone 多次轮询直到终态。
func TestSubmitAndWait_PollUntilDone(t *testing.T) {
	p := &sequenceProvider{
		name:   "x",
		models: []string{"m"},
		statuses: []TaskStatus{
			{Status: "running", Done: false},
			{Status: "running", Done: false},
			{Status: "success", Done: true, VideoURL: "http://v"},
		},
	}
	svc := NewService(map[string]Provider{"x": p}, "x")

	st, err := svc.SubmitAndWait(context.Background(), "", Request{Prompt: "hi", Model: "m"}, time.Millisecond)
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if !st.Done || st.Status != "success" || st.VideoURL != "http://v" {
		t.Errorf("应返回终态 success, 实际=%+v", st)
	}
	if p.pollCall < 3 {
		t.Errorf("应至少轮询 3 次, 实际=%d", p.pollCall)
	}
}

// TestSubmitAndWait_PollError 轮询出错应透传，last 为上次成功状态（可能零值）。
func TestSubmitAndWait_PollError(t *testing.T) {
	p := &sequenceProvider{
		name:      "x",
		models:    []string{"m"},
		statuses:  []TaskStatus{{Status: "running", Done: false}},
		pollErrAt: 2, // 第 2 次轮询返回错误
		pollErr:   errPollBoom,
	}
	svc := NewService(map[string]Provider{"x": p}, "x")

	last, err := svc.SubmitAndWait(context.Background(), "", Request{Prompt: "hi", Model: "m"}, time.Millisecond)
	if err == nil {
		t.Fatal("轮询错误应透传")
	}
	// 文档承诺：出错时 last 为最后一次成功获取的状态（这里第 1 次成功拿到 running）
	if last.Status != "running" {
		t.Errorf("出错时 last 应为最后一次成功状态 running, 实际=%+v", last)
	}
}

// TestSubmitAndWait_ContextCanceledDuringPoll ctx 在轮询途中取消应返回 ctx.Err。
func TestSubmitAndWait_ContextCanceledDuringPoll(t *testing.T) {
	p := &mockVideoProvider{
		name:   "x",
		models: []string{"m"},
		status: TaskStatus{Status: "running", Done: false}, // 永不终态
	}
	svc := NewService(map[string]Provider{"x": p}, "x")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	_, err := svc.SubmitAndWait(ctx, "", Request{Prompt: "hi", Model: "m"}, 5*time.Millisecond)
	if err == nil {
		t.Fatal("永不终态 + ctx 超时应返回错误")
	}
}

// TestSubmitAndWait_ImmediateDone 首轮即终态应只轮询一次。
func TestSubmitAndWait_ImmediateDone(t *testing.T) {
	p := &mockVideoProvider{
		name:   "x",
		models: []string{"m"},
		status: TaskStatus{Status: "success", Done: true},
	}
	svc := NewService(map[string]Provider{"x": p}, "x")

	st, err := svc.SubmitAndWait(context.Background(), "", Request{Prompt: "hi", Model: "m"}, time.Second)
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if !st.Done {
		t.Error("应立即得到终态")
	}
	if p.pollCall != 1 {
		t.Errorf("首轮即终态应只轮询 1 次, 实际=%d", p.pollCall)
	}
}

// ============================================================================
//                          并发安全性
// ============================================================================

// TestService_ConcurrentReadOnly 并发只读访问（Submit/Poll/列表）应无 race。
// 用 -race 运行可检测数据竞争；Service 字段在构造后只读，预期安全。
//
// 注意：被测的是 Service 的并发安全性，Provider 用无状态实现（statelessProvider）
// 以免 mock 自身的计数器引入与被测代码无关的伪 race。
func TestService_ConcurrentReadOnly(t *testing.T) {
	p := &statelessProvider{name: "x", models: []string{"m"}}
	svc := NewService(map[string]Provider{"x": p}, "x")

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = svc.Submit(context.Background(), "", Request{Prompt: "hi", Model: "m"})
			_, _ = svc.Poll(context.Background(), "x::raw")
			_ = svc.Providers()
			_ = svc.Models()
			_ = svc.HasProvider()
		}()
	}
	wg.Wait()
}

// ============================================================================
//                          测试辅助类型
// ============================================================================

var errPollBoom = errPoll("poll boom")

type errPoll string

func (e errPoll) Error() string { return string(e) }

// statelessProvider 无任何可变状态，可安全跨 goroutine 并发使用，
// 专用于验证 Service 本身的并发只读安全性。
type statelessProvider struct {
	name   string
	models []string
}

func (s *statelessProvider) Name() string              { return s.name }
func (s *statelessProvider) SupportedModels() []string { return s.models }
func (s *statelessProvider) Submit(_ context.Context, _ Request) (string, error) {
	return "raw", nil
}
func (s *statelessProvider) Poll(_ context.Context, _ string) (TaskStatus, error) {
	return TaskStatus{Status: "success", Done: true}, nil
}

// capturingProvider 捕获最后一次 Submit 的 Request。
type capturingProvider struct {
	name     string
	models   []string
	submitID string
	capture  *Request
}

func (c *capturingProvider) Name() string              { return c.name }
func (c *capturingProvider) SupportedModels() []string { return c.models }
func (c *capturingProvider) Submit(_ context.Context, req Request) (string, error) {
	*c.capture = req
	return c.submitID, nil
}
func (c *capturingProvider) Poll(_ context.Context, _ string) (TaskStatus, error) {
	return TaskStatus{}, nil
}

// ============================================================================
//   追加审计：状态字段一致性 / 真实 HTTP 路径并发 / 错误路径补全
// ============================================================================

// TestZhipu_Poll_SuccessStatusNotLeftLowercased 验证 SUCCESS 分支最终 Status
// 被规范化为 "success"，而不是停留在初始 strings.ToLower 的中间值。
//
// 源码先 st.Status = strings.ToLower(parsed.TaskStatus)（"success"），
// 再在 switch case "SUCCESS" 里覆盖为 "success"。对 SUCCESS 二者恰好一致，
// 此测试钉死该不变量，防止未来上游返回 "Success"/"Succeed" 等变体时回归。
func TestZhipu_Poll_SuccessStatusNotLeftLowercased(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"task_status":"SUCCESS","video_result":[{"url":"http://v"}]}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	st, err := p.Poll(context.Background(), "t")
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if st.Status != "success" {
		t.Errorf("SUCCESS 应规范为 success, 实际=%q", st.Status)
	}
	if !st.Done {
		t.Error("SUCCESS 应为终态")
	}
}

// TestZhipu_Poll_MixedCaseStatusNotRecognized 暴露状态机大小写处理的一致性问题。
//
// switch 用 strings.ToUpper 比较，所以 "success"/"Success" 都能命中 SUCCESS 分支
// 并置 Done=true。此测试验证混合大小写终态仍被正确识别为终态（防止漏判导致
// 调用方无限轮询）。
func TestZhipu_Poll_MixedCaseTerminalRecognized(t *testing.T) {
	for _, raw := range []string{"success", "Success", "Fail", "failed"} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"task_status":"` + raw + `"}`))
		}))
		p := newZhipuWithServer("k", srv.URL)
		st, err := p.Poll(context.Background(), "t")
		srv.Close()
		if err != nil {
			t.Fatalf("raw=%q 未预期错误: %v", raw, err)
		}
		if !st.Done {
			t.Errorf("raw=%q 应被识别为终态 Done=true, 实际 Status=%q Done=%v", raw, st.Status, st.Done)
		}
	}
}

// TestZhipu_ConnectionRefused_Submit Submit 在底层 transport 失败时应包 "submit:" 前缀。
// 用一个已关闭的 httptest server 制造连接失败（不打真实外网）。
func TestZhipu_ConnectionRefused_Submit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close() // 立刻关闭 -> 后续请求连接被拒
	p := newZhipuWithServer("k", url)

	_, err := p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
	if err == nil || !strings.Contains(err.Error(), "submit") {
		t.Errorf("连接失败应报 submit 包装错误, 实际=%v", err)
	}
}

// TestZhipu_ConnectionRefused_Poll Poll transport 失败应包 "poll:" 前缀。
func TestZhipu_ConnectionRefused_Poll(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()
	p := newZhipuWithServer("k", url)

	_, err := p.Poll(context.Background(), "t")
	if err == nil || !strings.Contains(err.Error(), "poll") {
		t.Errorf("连接失败应报 poll 包装错误, 实际=%v", err)
	}
}

// TestZhipu_Poll_EmptyTaskIDStillBuildsPath taskID 为空时仍能构造请求（path 以 / 结尾）。
// 这是一个退化输入：Service 层 resolveByTaskID 对 "zhipu::" 会得到空 raw 并放行（见 findings），
// 空 raw 传到这里 path 变成 ".../async-result/"。验证至少不 panic、能拿到上游响应。
func TestZhipu_Poll_EmptyTaskID(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"task_status":"PROCESSING"}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	_, err := p.Poll(context.Background(), "")
	if err != nil {
		t.Fatalf("空 taskID 不应导致错误（退化但合法）, 实际=%v", err)
	}
	if !strings.HasSuffix(gotPath, "/async-result/") {
		t.Errorf("空 taskID path 应以 /async-result/ 结尾, 实际=%s", gotPath)
	}
}

// TestZhipu_RealHTTPConcurrency 通过 httptest 真实 HTTP 路径并发 Submit/Poll，
// 用 -race 验证 Provider 在并发下无数据竞争（httpc/baseURL/apiKey 构造后只读）。
func TestZhipu_RealHTTPConcurrency(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_, _ = w.Write([]byte(`{"id":"t1","task_status":"PROCESSING"}`))
			return
		}
		_, _ = w.Write([]byte(`{"task_status":"SUCCESS","video_result":[{"url":"http://v"}]}`))
	}))
	defer srv.Close()
	p := newZhipuWithServer("k", srv.URL)

	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = p.Submit(context.Background(), Request{Prompt: "x", Model: "cogvideox-2"})
			_, _ = p.Poll(context.Background(), "t1")
		}()
	}
	wg.Wait()
}

// TestService_SubmitAndWait_TaskIDRoundTrip 端到端：SubmitAndWait 内部 Submit 生成
// 带前缀 ID，Poll 时应能正确反查回原 Provider 的 raw ID。
//
// 用 capturingPollProvider 记录 Poll 收到的 raw taskID，验证前缀被正确剥离。
func TestService_SubmitAndWait_TaskIDRoundTrip(t *testing.T) {
	cp := &capturingPollProvider{name: "zhipu", models: []string{"m"}, submitID: "raw-77"}
	svc := NewService(map[string]Provider{"zhipu": cp}, "zhipu")

	st, err := svc.SubmitAndWait(context.Background(), "", Request{Prompt: "hi", Model: "m"}, time.Millisecond)
	if err != nil {
		t.Fatalf("未预期错误: %v", err)
	}
	if cp.lastPollRaw != "raw-77" {
		t.Errorf("Poll 应收到剥离前缀的 raw ID, 期望=raw-77 实际=%q", cp.lastPollRaw)
	}
	if st.TaskID != "zhipu::raw-77" {
		t.Errorf("终态 TaskID 应带前缀, 实际=%q", st.TaskID)
	}
}

// capturingPollProvider 记录 Poll 收到的 raw taskID，首轮即返回终态。
type capturingPollProvider struct {
	name        string
	models      []string
	submitID    string
	lastPollRaw string
}

func (c *capturingPollProvider) Name() string              { return c.name }
func (c *capturingPollProvider) SupportedModels() []string { return c.models }
func (c *capturingPollProvider) Submit(_ context.Context, _ Request) (string, error) {
	return c.submitID, nil
}
func (c *capturingPollProvider) Poll(_ context.Context, raw string) (TaskStatus, error) {
	c.lastPollRaw = raw
	return TaskStatus{Status: "success", Done: true}, nil
}

// sequenceProvider 按序返回一组状态，可在第 N 次 Poll 注入错误。
type sequenceProvider struct {
	name      string
	models    []string
	statuses  []TaskStatus
	pollCall  int
	pollErrAt int // 第几次 Poll 返回错误（1-based, 0 表示不注入）
	pollErr   error
}

func (s *sequenceProvider) Name() string              { return s.name }
func (s *sequenceProvider) SupportedModels() []string { return s.models }
func (s *sequenceProvider) Submit(_ context.Context, _ Request) (string, error) {
	return "raw-seq", nil
}
func (s *sequenceProvider) Poll(_ context.Context, _ string) (TaskStatus, error) {
	s.pollCall++
	if s.pollErrAt != 0 && s.pollCall == s.pollErrAt {
		return TaskStatus{}, s.pollErr
	}
	idx := s.pollCall - 1
	if idx >= len(s.statuses) {
		idx = len(s.statuses) - 1
	}
	if idx < 0 {
		return TaskStatus{}, nil
	}
	return s.statuses[idx], nil
}
