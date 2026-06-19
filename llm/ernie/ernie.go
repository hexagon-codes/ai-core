// Package ernie provides Baidu ERNIE (文心一言) LLM provider.
//
// Differences from OpenAI:
//   - Auth: OAuth access_token (not API key in header)
//   - Model: encoded in URL path (not request body)
//   - Response: "result" field (not "choices[0].message.content")
//   - System: separate "system" field (not in messages array)
package ernie

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const (
	defaultBaseURL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
	tokenURL       = "https://aip.baidubce.com/oauth/2.0/token"
	defaultModel   = "ernie-4.0-8k"
)

// 编译期接口断言：*Provider 满足 llm.Provider 契约。
var _ llm.Provider = (*Provider)(nil)

// Provider 百度文心一言 Provider
type Provider struct {
	apiKey      string
	secretKey   string
	baseURL     string
	model       string
	httpClient  *http.Client
	mu          sync.Mutex
	accessToken string
	tokenExpiry time.Time
}

// Option 配置选项
type Option func(*Provider)

// WithBaseURL 设置 API 基础 URL
func WithBaseURL(url string) Option {
	return func(p *Provider) { p.baseURL = url }
}

// WithModel 设置默认模型
func WithModel(model string) Option {
	return func(p *Provider) { p.model = model }
}

// WithHTTPClient 设置 HTTP 客户端
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) { p.httpClient = client }
}

// New 创建 ERNIE Provider
func New(apiKey, secretKey string, opts ...Option) *Provider {
	p := &Provider{
		apiKey:     apiKey,
		secretKey:  secretKey,
		baseURL:    defaultBaseURL,
		model:      defaultModel,
		httpClient: httpx.RawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Name 返回 provider 名称
func (p *Provider) Name() string { return "ernie" }

// Complete 非流式补全
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	token, err := p.ensureToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("ernie auth failed: %w", err)
	}

	model := req.Model
	if model == "" {
		model = p.model
	}

	body := p.buildRequest(req, false)
	url := fmt.Sprintf("%s/chat/%s?access_token=%s", p.baseURL, model, token)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ernie request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ernie read response failed: %w", err)
	}

	return p.parseResponse(data)
}

// Stream 流式补全
func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	token, err := p.ensureToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("ernie auth failed: %w", err)
	}

	model := req.Model
	if model == "" {
		model = p.model
	}

	body := p.buildRequest(req, true)
	url := fmt.Sprintf("%s/chat/%s?access_token=%s", p.baseURL, model, token)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ernie stream failed: %w", err)
	}

	return streamx.NewStreamWithParser(resp.Body, &ernieStreamParser{}), nil
}

// Models 返回支持的模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{ID: "ernie-4.5-8k", Name: "ERNIE 4.5", Description: "百度最新旗舰模型"},
		{ID: "ernie-4.0-8k", Name: "ERNIE 4.0", Description: "强大的中文理解能力"},
		{ID: "ernie-3.5-8k", Name: "ERNIE 3.5", Description: "性价比高"},
		{ID: "ernie-x1", Name: "ERNIE X1", Description: "深度推理模型"},
	}
}

// CountTokens 估算 token 数量 (粗略: 1 中文字 ≈ 2 tokens)
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	total := 0
	for _, m := range messages {
		total += len([]rune(m.Content)) * 2
	}
	return total, nil
}

// ============== Internal ==============

// ensureToken 确保 access_token 有效，过期前 5 分钟刷新
func (p *Provider) ensureToken(ctx context.Context) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.accessToken != "" && time.Now().Before(p.tokenExpiry.Add(-5*time.Minute)) {
		return p.accessToken, nil
	}

	url := fmt.Sprintf("%s?grant_type=client_credentials&client_id=%s&client_secret=%s",
		tokenURL, p.apiKey, p.secretKey)

	resp, err := p.httpClient.Get(url)
	if err != nil {
		return "", fmt.Errorf("token request failed: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
		Error       string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("token decode failed: %w", err)
	}
	if result.Error != "" {
		return "", fmt.Errorf("token error: %s", result.Error)
	}

	p.accessToken = result.AccessToken
	p.tokenExpiry = time.Now().Add(time.Duration(result.ExpiresIn) * time.Second)
	return p.accessToken, nil
}

type ernieMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ernieRequest struct {
	Messages []ernieMessage `json:"messages"`
	System   string         `json:"system,omitempty"`
	Stream   bool           `json:"stream,omitempty"`
}

func (p *Provider) buildRequest(req llm.CompletionRequest, stream bool) []byte {
	er := ernieRequest{Stream: stream}

	for _, m := range req.Messages {
		role := string(m.Role)
		if role == "system" {
			er.System = m.Content
			continue
		}
		if role != "user" && role != "assistant" {
			role = "user"
		}
		er.Messages = append(er.Messages, ernieMessage{Role: role, Content: m.Content})
	}

	// ERNIE requires messages to start with user
	if len(er.Messages) > 0 && er.Messages[0].Role != "user" {
		er.Messages = append([]ernieMessage{{Role: "user", Content: "请继续"}}, er.Messages...)
	}

	data, _ := json.Marshal(er)
	return data
}

type ernieResponse struct {
	ID     string `json:"id"`
	Result string `json:"result"`
	Usage  struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	ErrorCode int    `json:"error_code"`
	ErrorMsg  string `json:"error_msg"`
}

func (p *Provider) parseResponse(data []byte) (*llm.CompletionResponse, error) {
	var er ernieResponse
	if err := json.Unmarshal(data, &er); err != nil {
		return nil, fmt.Errorf("ernie parse failed: %w", err)
	}
	if er.ErrorCode != 0 {
		return nil, fmt.Errorf("ernie error %d: %s", er.ErrorCode, er.ErrorMsg)
	}

	return &llm.CompletionResponse{
		ID:      er.ID,
		Content: er.Result,
		Usage: llm.Usage{
			PromptTokens:     er.Usage.PromptTokens,
			CompletionTokens: er.Usage.CompletionTokens,
			TotalTokens:      er.Usage.TotalTokens,
		},
		FinishReason: "stop",
	}, nil
}

// ernieStreamParser 解析 ERNIE SSE 流
type ernieStreamParser struct{}

func (p *ernieStreamParser) Parse(data []byte) (*streamx.Chunk, error) {
	var er struct {
		Result string `json:"result"`
		IsEnd  bool   `json:"is_end"`
		Usage  struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &er); err != nil {
		return nil, err
	}

	chunk := &streamx.Chunk{
		Content: er.Result,
	}
	if er.IsEnd {
		chunk.FinishReason = "stop"
	}
	return chunk, nil
}

func (p *ernieStreamParser) IsDone(data []byte) bool {
	var r struct {
		IsEnd bool `json:"is_end"`
	}
	json.Unmarshal(data, &r)
	return r.IsEnd
}
