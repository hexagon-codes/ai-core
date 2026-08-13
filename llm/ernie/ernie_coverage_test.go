package ernie

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

// 本文件把 ernie (文心一言) provider 的全场景单测补齐：配置/请求构造/
// 角色转换/鉴权 token 获取/HTTP 成功与错误路径/流式/响应与流块解析。
//
// 难点说明：ernie 的 ensureToken 直接请求硬编码常量 tokenURL，无法通过
// WithBaseURL 覆盖。因此采用两种 mock 手段：
//  1. 通过 WithHTTPClient 注入带自定义 RoundTripper 的 *http.Client，
//     按请求 Host/Path 分流 token 与 chat 两类请求；
//  2. 对只关心 chat 链路的用例，直接预置 p.accessToken/p.tokenExpiry，
//     令 ensureToken 命中缓存分支、跳过真实 token 请求。

// rtFunc 把函数适配为 http.RoundTripper
type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// jsonResp 构造一个携带 JSON body 的 *http.Response
func jsonResp(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

// seedToken 预置有效 token，使 ensureToken 走缓存分支
func seedToken(p *Provider) {
	p.accessToken = "cached-token"
	p.tokenExpiry = time.Now().Add(time.Hour)
}

// errReadCloser 是一个读取即报错的 ReadCloser，用于触发 io.ReadAll 失败路径
type errReadCloser struct{}

func (errReadCloser) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
func (errReadCloser) Close() error             { return nil }

func TestNew_DefaultsAndOptions(t *testing.T) {
	p := New("ak", "sk")
	if p.apiKey != "ak" || p.secretKey != "sk" {
		t.Fatalf("apiKey/secretKey 未写入: %+v", p)
	}
	if p.baseURL != defaultBaseURL || p.model != defaultModel {
		t.Fatalf("默认 baseURL/model 错误: %+v", p)
	}
	if p.httpClient == nil {
		t.Fatal("httpClient 应非 nil")
	}

	custom := &http.Client{}
	p2 := New("ak", "sk",
		WithBaseURL("http://x"),
		WithModel("ernie-x1"),
		WithHTTPClient(custom),
	)
	if p2.baseURL != "http://x" || p2.model != "ernie-x1" || p2.httpClient != custom {
		t.Fatalf("选项未生效: %+v", p2)
	}
}

func TestName_Coverage(t *testing.T) {
	if New("ak", "sk").Name() != "ernie" {
		t.Fatal("Name 应为 ernie")
	}
}

func TestModels_Coverage(t *testing.T) {
	models := New("ak", "sk").Models()
	if len(models) != 4 {
		t.Fatalf("应有 4 个模型, got %d", len(models))
	}
	if models[0].ID != "ernie-4.5-8k" {
		t.Fatalf("首个模型 ID 错误: %+v", models[0])
	}
}

func TestCountTokens_Coverage(t *testing.T) {
	// 中文每字 2 token，英文每字符也按 rune 计 2 token
	n, err := New("ak", "sk").CountTokens([]llm.Message{
		{Role: llm.RoleUser, Content: "你好"},      // 2 runes * 2 = 4
		{Role: llm.RoleAssistant, Content: "ab"}, // 2 runes * 2 = 4
	})
	if err != nil || n != 8 {
		t.Fatalf("CountTokens = %d,%v want 8,nil", n, err)
	}
	// 空消息列表
	if n, _ := New("ak", "sk").CountTokens(nil); n != 0 {
		t.Fatalf("空消息应为 0, got %d", n)
	}
}

func TestBuildRequest_AllBranches(t *testing.T) {
	p := New("ak", "sk")

	// 含 system + user + assistant + 非法角色(tool→user) + stream 标志
	body := p.buildRequest(llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "你是助手"},
			{Role: llm.RoleUser, Content: "你好"},
			{Role: llm.RoleAssistant, Content: "在的"},
			{Role: llm.RoleTool, Content: "工具结果"}, // 非 user/assistant → 归一为 user
		},
	}, true)

	var er ernieRequest
	if err := json.Unmarshal(body, &er); err != nil {
		t.Fatal(err)
	}
	if er.System != "你是助手" {
		t.Fatalf("system 未分离: %q", er.System)
	}
	if !er.Stream {
		t.Fatal("stream 标志应为 true")
	}
	if len(er.Messages) != 3 {
		t.Fatalf("应有 3 条非 system 消息, got %d", len(er.Messages))
	}
	if er.Messages[0].Role != "user" {
		t.Fatalf("首条应为 user, got %q", er.Messages[0].Role)
	}
	if er.Messages[2].Role != "user" || er.Messages[2].Content != "工具结果" {
		t.Fatalf("tool 角色应归一为 user: %+v", er.Messages[2])
	}
}

func TestBuildRequest_PrependUserWhenAssistantFirst(t *testing.T) {
	p := New("ak", "sk")
	// 首条为 assistant，ERNIE 要求首条 user → 自动前插 "请继续"
	body := p.buildRequest(llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, Content: "hi"},
			{Role: llm.RoleUser, Content: "hello"},
		},
	}, false)
	var er ernieRequest
	_ = json.Unmarshal(body, &er)
	if er.Stream {
		t.Fatal("非流式 stream 应为 false")
	}
	if len(er.Messages) != 3 || er.Messages[0].Role != "user" || er.Messages[0].Content != "请继续" {
		t.Fatalf("应前插 user 占位消息: %+v", er.Messages)
	}
}

func TestBuildRequest_Empty(t *testing.T) {
	// 无消息：不触发前插分支，messages 为空
	body := New("ak", "sk").buildRequest(llm.CompletionRequest{}, false)
	var er ernieRequest
	_ = json.Unmarshal(body, &er)
	if len(er.Messages) != 0 {
		t.Fatalf("空请求 messages 应为空, got %d", len(er.Messages))
	}
}

func TestParseResponse_Success_Coverage(t *testing.T) {
	resp, err := New("ak", "sk").parseResponse([]byte(
		`{"id":"as-1","result":"你好世界","usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "as-1" || resp.Content != "你好世界" {
		t.Fatalf("内容解析错误: %+v", resp)
	}
	if resp.Usage.TotalTokens != 10 || resp.Usage.PromptTokens != 7 || resp.Usage.CompletionTokens != 3 {
		t.Fatalf("usage 解析错误: %+v", resp.Usage)
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("FinishReason 应为 stop, got %q", resp.FinishReason)
	}
}

func TestParseResponse_ErrorCode(t *testing.T) {
	_, err := New("ak", "sk").parseResponse([]byte(
		`{"error_code":110,"error_msg":"Access token invalid or no longer valid"}`))
	if err == nil || !strings.Contains(err.Error(), "110") ||
		!strings.Contains(err.Error(), "Access token invalid") {
		t.Fatalf("应返回含错误码与错误信息的错误, got %v", err)
	}
}

func TestParseResponse_Malformed(t *testing.T) {
	_, err := New("ak", "sk").parseResponse([]byte(`{not json`))
	if err == nil || !strings.Contains(err.Error(), "parse failed") {
		t.Fatalf("畸形 JSON 应返回解析错误, got %v", err)
	}
}

// --- ensureToken 各分支 ---

func TestEnsureToken_CachedHit(t *testing.T) {
	p := New("ak", "sk")
	seedToken(p)
	tok, err := p.ensureToken(context.Background())
	if err != nil || tok != "cached-token" {
		t.Fatalf("应命中缓存返回 cached-token, got %q,%v", tok, err)
	}
}

func TestEnsureToken_FetchSuccess(t *testing.T) {
	var hitURL string
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			hitURL = r.URL.String()
			return jsonResp(http.StatusOK, `{"access_token":"new-token","expires_in":2592000}`), nil
		}),
	}))
	tok, err := p.ensureToken(context.Background())
	if err != nil || tok != "new-token" {
		t.Fatalf("应获取 new-token, got %q,%v", tok, err)
	}
	// token 请求 URL 含凭据与 grant_type
	if !strings.Contains(hitURL, "grant_type=client_credentials") ||
		!strings.Contains(hitURL, "client_id=ak") ||
		!strings.Contains(hitURL, "client_secret=sk") {
		t.Fatalf("token 请求 URL 错误: %s", hitURL)
	}
	// 过期时间被设置到未来
	if !p.tokenExpiry.After(time.Now()) {
		t.Fatal("tokenExpiry 应设置到未来")
	}
	// 再次调用命中缓存（不应改变 token）
	tok2, _ := p.ensureToken(context.Background())
	if tok2 != "new-token" {
		t.Fatalf("二次调用应命中缓存, got %q", tok2)
	}
}

func TestEnsureToken_ExpiredRefetch(t *testing.T) {
	calls := 0
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			calls++
			return jsonResp(http.StatusOK, `{"access_token":"refreshed","expires_in":1}`), nil
		}),
	}))
	// 预置一个即将过期的 token（5 分钟刷新窗口内）→ 触发重新获取
	p.accessToken = "old"
	p.tokenExpiry = time.Now().Add(time.Minute)
	tok, err := p.ensureToken(context.Background())
	if err != nil || tok != "refreshed" || calls != 1 {
		t.Fatalf("过期窗口内应重新获取, got %q,%v calls=%d", tok, err, calls)
	}
}

func TestEnsureToken_RequestError(t *testing.T) {
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			return nil, io.ErrUnexpectedEOF
		}),
	}))
	_, err := p.ensureToken(context.Background())
	if err == nil || !strings.Contains(err.Error(), "token request failed") {
		t.Fatalf("token 请求失败应包装错误, got %v", err)
	}
}

func TestEnsureToken_DecodeError(t *testing.T) {
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			return jsonResp(http.StatusOK, `{bad json`), nil
		}),
	}))
	_, err := p.ensureToken(context.Background())
	if err == nil || !strings.Contains(err.Error(), "token decode failed") {
		t.Fatalf("token 解码失败应包装错误, got %v", err)
	}
}

func TestEnsureToken_APIError(t *testing.T) {
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			return jsonResp(http.StatusOK,
				`{"error":"invalid_client","error_description":"unknown client id"}`), nil
		}),
	}))
	_, err := p.ensureToken(context.Background())
	if err == nil || !strings.Contains(err.Error(), "token error") ||
		!strings.Contains(err.Error(), "invalid_client") {
		t.Fatalf("token 业务错误应返回, got %v", err)
	}
}

// --- Complete 各路径 ---

// completeReq 是一个最小可用的补全请求
func completeReq() llm.CompletionRequest {
	return llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "你好"}},
	}
}

func TestComplete_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("缺 Content-Type")
		}
		// access_token 应出现在 query
		if r.URL.Query().Get("access_token") != "cached-token" {
			t.Errorf("缺 access_token query: %s", r.URL.RawQuery)
		}
		// 模型应编码进 path
		if !strings.Contains(r.URL.Path, "/chat/ernie-4.0-8k") {
			t.Errorf("模型未编码进 path: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"id":"as-9","result":"嗨","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
	}))
	defer srv.Close()

	p := New("ak", "sk", WithBaseURL(srv.URL))
	seedToken(p) // 跳过真实 token 请求
	resp, err := p.Complete(context.Background(), completeReq())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "嗨" || resp.Usage.TotalTokens != 3 {
		t.Fatalf("响应解析错误: %+v", resp)
	}
}

func TestComplete_UsesRequestModelOverride(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/chat/ernie-x1") {
			t.Errorf("请求级 Model 未覆盖默认: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"id":"x","result":"ok"}`))
	}))
	defer srv.Close()

	p := New("ak", "sk", WithBaseURL(srv.URL))
	seedToken(p)
	req := completeReq()
	req.Model = "ernie-x1"
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}

func TestComplete_AuthFailure(t *testing.T) {
	// token 请求失败 → Complete 提前返回 auth failed
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			return nil, io.ErrUnexpectedEOF
		}),
	}))
	_, err := p.Complete(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "ernie auth failed") {
		t.Fatalf("token 失败应返回 auth failed, got %v", err)
	}
}

func TestComplete_ErrorBody(t *testing.T) {
	// 上游返回业务错误码（HTTP 200 + error_code）
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"error_code":336003,"error_msg":"messages role must be user"}`))
	}))
	defer srv.Close()
	p := New("ak", "sk", WithBaseURL(srv.URL))
	seedToken(p)
	_, err := p.Complete(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "336003") {
		t.Fatalf("业务错误码应返回错误, got %v", err)
	}
}

func TestComplete_MalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{not json`))
	}))
	defer srv.Close()
	p := New("ak", "sk", WithBaseURL(srv.URL))
	seedToken(p)
	_, err := p.Complete(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "parse failed") {
		t.Fatalf("畸形 JSON 应返回解析错误, got %v", err)
	}
}

func TestComplete_ReadBodyError(t *testing.T) {
	// 响应体读取中途失败 → io.ReadAll 返回错误
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       errReadCloser{},
				Header:     make(http.Header),
			}, nil
		}),
	}))
	seedToken(p)
	_, err := p.Complete(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "read response failed") {
		t.Fatalf("读取响应体失败应返回错误, got %v", err)
	}
}

func TestComplete_BadBaseURL(t *testing.T) {
	// baseURL 缺协议头 → NewRequestWithContext 解析失败
	p := New("ak", "sk", WithBaseURL("://invalid"))
	seedToken(p)
	if _, err := p.Complete(context.Background(), completeReq()); err == nil {
		t.Fatal("非法 URL 应报错")
	}
}

func TestComplete_Unreachable(t *testing.T) {
	// 不可达上游 → httpClient.Do 失败
	p := New("ak", "sk", WithBaseURL("http://127.0.0.1:1"))
	seedToken(p)
	_, err := p.Complete(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "ernie request failed") {
		t.Fatalf("不可达上游应返回 request failed, got %v", err)
	}
}

// --- Stream 各路径 ---

func TestStream_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("流式应带 Accept text/event-stream")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"result\":\"你\",\"is_end\":false}\n\n"))
		_, _ = w.Write([]byte("data: {\"result\":\"好\",\"is_end\":true,\"usage\":{\"total_tokens\":5}}\n\n"))
	}))
	defer srv.Close()

	p := New("ak", "sk", WithBaseURL(srv.URL))
	seedToken(p)
	s, err := p.Stream(context.Background(), completeReq())
	if err != nil || s == nil {
		t.Fatalf("流式成功应返回非 nil stream: %v", err)
	}
	result, err := s.Collect()
	if err != nil {
		t.Fatalf("Collect 出错: %v", err)
	}
	if !strings.Contains(result.Content, "你") || !strings.Contains(result.Content, "好") {
		t.Fatalf("流式内容聚合错误: %q", result.Content)
	}
}

func TestStream_AuthFailure(t *testing.T) {
	p := New("ak", "sk", WithHTTPClient(&http.Client{
		Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
			return nil, io.ErrUnexpectedEOF
		}),
	}))
	_, err := p.Stream(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "ernie auth failed") {
		t.Fatalf("token 失败应返回 auth failed, got %v", err)
	}
}

func TestStream_BadBaseURL(t *testing.T) {
	p := New("ak", "sk", WithBaseURL("://invalid"))
	seedToken(p)
	if _, err := p.Stream(context.Background(), completeReq()); err == nil {
		t.Fatal("非法 URL 应报错")
	}
}

func TestStream_Unreachable(t *testing.T) {
	p := New("ak", "sk", WithBaseURL("http://127.0.0.1:1"))
	seedToken(p)
	_, err := p.Stream(context.Background(), completeReq())
	if err == nil || !strings.Contains(err.Error(), "ernie stream failed") {
		t.Fatalf("不可达上游应返回 stream failed, got %v", err)
	}
}

func TestStream_UsesRequestModelOverride(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/chat/ernie-3.5-8k") {
			t.Errorf("Stream 请求级 Model 未覆盖: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"result\":\"x\",\"is_end\":true}\n\n"))
	}))
	defer srv.Close()
	p := New("ak", "sk", WithBaseURL(srv.URL))
	seedToken(p)
	req := completeReq()
	req.Model = "ernie-3.5-8k"
	if _, err := p.Stream(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}

// --- ernieStreamParser 各分支 ---

func TestStreamParser_Parse(t *testing.T) {
	sp := &ernieStreamParser{}

	// 普通增量块（未结束）
	c, err := sp.Parse([]byte(`{"result":"片段","is_end":false}`))
	if err != nil || c.Content != "片段" || c.FinishReason != "" {
		t.Fatalf("增量块解析错误: %+v,%v", c, err)
	}

	// 结束块
	c, err = sp.Parse([]byte(`{"result":"末尾","is_end":true,"usage":{"total_tokens":9}}`))
	if err != nil || c.Content != "末尾" || c.FinishReason != "stop" {
		t.Fatalf("结束块解析错误: %+v,%v", c, err)
	}

	// 畸形 JSON → 返回错误
	if _, err := sp.Parse([]byte(`{bad`)); err == nil {
		t.Fatal("畸形 JSON 应返回错误")
	}
}

func TestStreamParser_IsDone(t *testing.T) {
	sp := &ernieStreamParser{}
	if !sp.IsDone([]byte(`{"is_end":true}`)) {
		t.Fatal("is_end=true 应判为 done")
	}
	if sp.IsDone([]byte(`{"is_end":false}`)) {
		t.Fatal("is_end=false 不应判为 done")
	}
	// 畸形 JSON：json.Unmarshal 失败但被忽略，IsEnd 保持零值 false
	if sp.IsDone([]byte(`{bad`)) {
		t.Fatal("畸形 JSON 应判为非 done")
	}
}
