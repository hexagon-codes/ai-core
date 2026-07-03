package video

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
)

// 智谱视频生成 endpoint：
//   - POST {base}/videos/generations           — 异步提交，返回 task ID
//   - GET  {base}/async-result/{task_id}       — 查询任务状态
//
// 文档：https://bigmodel.cn/dev/api/videomodel/cogvideox
const (
	defaultZhipuVideoBase = "https://open.bigmodel.cn/api/paas/v4"
)

type zhipuCogVideoX struct {
	apiKey  string
	baseURL string
	models  []string
	httpc   *http.Client
	timeout time.Duration
	policy  *llm.NetworkPolicy
	tx      *transport.Transport
}

// ZhipuOption 配置 zhipu CogVideoX Provider 的可选项。
//
// 采用函数式选项（functional option）模式，对既有 NewZhipuCogVideoX(apiKey,
// baseURL) 调用完全向后兼容：不传任何 Option 时行为不变。
type ZhipuOption func(*zhipuCogVideoX)

// WithHTTPClient 注入自定义 *http.Client。
//
// 默认 Provider 共享 http.DefaultClient（全局单例，超时不可控）。生产环境
// 应注入独立 client 以隔离连接池、设置超时与 transport。传入 nil 时忽略，
// 保留默认 client，避免误注入导致后续请求 panic。
func WithHTTPClient(c *http.Client) ZhipuOption {
	return func(z *zhipuCogVideoX) {
		if c != nil {
			z.httpc = c
		}
	}
}

// WithNetworkPolicy 设置上游网络出口约束。
func WithNetworkPolicy(policy llm.NetworkPolicy) ZhipuOption {
	return func(z *zhipuCogVideoX) {
		z.policy = transport.CloneNetworkPolicy(&policy)
	}
}

// NewZhipuCogVideoX 创建智谱 CogVideoX Provider。
//
// opts 为可选的额外配置（如 WithHTTPClient 注入自定义 HTTP 客户端），
// 不传时使用默认值，保持与旧调用方的二进制/源码兼容。
func NewZhipuCogVideoX(apiKey, baseURL string, opts ...ZhipuOption) Provider {
	if baseURL == "" {
		baseURL = defaultZhipuVideoBase
	}
	z := &zhipuCogVideoX{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		// CogVideoX 在售模型；后续新模型可追加
		models: []string{"cogvideox-2", "cogvideox-flash", "cogvideox"},
		// 默认使用独立 client（含超时），避免与全局 http.DefaultClient 共享；
		// 调用方可通过 WithHTTPClient 进一步覆盖。
		httpc:   &http.Client{Timeout: 60 * time.Second},
		timeout: 60 * time.Second, // 提交快，轮询另算
	}
	for _, opt := range opts {
		if opt != nil {
			opt(z)
		}
	}
	z.tx = transport.NewTransport(z.httpc, z.policy)
	return z
}

func (z *zhipuCogVideoX) Name() string              { return "zhipu-cogvideox" }
func (z *zhipuCogVideoX) SupportedModels() []string { return z.models }

// 提交请求体匹配智谱 /videos/generations 协议。
type zhipuSubmitReq struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt,omitempty"`
	ImageURL  string `json:"image_url,omitempty"`
	Quality   string `json:"quality,omitempty"`
	WithAudio bool   `json:"with_audio,omitempty"`
	Size      string `json:"size,omitempty"`
	FPS       int    `json:"fps,omitempty"`
	Duration  int    `json:"duration,omitempty"`
	UserID    string `json:"user_id,omitempty"`
}

type zhipuSubmitResp struct {
	ID         string `json:"id"`
	Model      string `json:"model"`
	RequestID  string `json:"request_id"`
	TaskStatus string `json:"task_status"`
	Error      *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type zhipuPollResp struct {
	ID          string `json:"id"`
	Model       string `json:"model"`
	TaskStatus  string `json:"task_status"` // PROCESSING / SUCCESS / FAIL
	VideoResult []struct {
		URL      string `json:"url"`
		CoverURL string `json:"cover_image_url"`
	} `json:"video_result,omitempty"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (z *zhipuCogVideoX) Submit(ctx context.Context, req Request) (string, error) {
	if z.apiKey == "" {
		return "", fmt.Errorf("zhipu-cogvideox: 未配置 API Key")
	}
	body := zhipuSubmitReq{
		Model:     req.Model,
		Prompt:    req.Prompt,
		ImageURL:  req.ImageURL,
		Quality:   req.Quality,
		WithAudio: req.WithAudio,
		Size:      req.Size,
		FPS:       req.FPS,
		Duration:  req.Duration,
		UserID:    req.UserID,
	}
	if body.Model == "" && len(z.models) > 0 {
		body.Model = z.models[0]
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	resp, err := transport.Do(ctx, transport.Request{
		Provider:  z.Name(),
		Action:    "video.submit",
		Method:    http.MethodPost,
		URL:       z.baseURL + "/videos/generations",
		Body:      payload,
		Client:    z.httpc,
		Transport: z.tx,
		Timeout:   z.timeout,
		// 本 Provider 不向上游透传幂等键，任务创建 POST 一律不重试。
		Retry: media.SubmitRetryPolicy(""),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Authorization", "Bearer "+z.apiKey)
		},
	})
	if err != nil {
		return "", fmt.Errorf("submit: %w", err)
	}
	defer resp.Body.Close()

	var parsed zhipuSubmitResp
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("decode submit response: %w", err)
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("zhipu: %s", parsed.Error.Message)
	}
	if parsed.ID == "" {
		return "", fmt.Errorf("zhipu: 提交成功但未返回 task ID")
	}
	return parsed.ID, nil
}

func (z *zhipuCogVideoX) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	if z.apiKey == "" {
		return TaskStatus{}, fmt.Errorf("zhipu-cogvideox: 未配置 API Key")
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  z.Name(),
		Action:    "video.poll",
		Method:    http.MethodGet,
		URL:       z.baseURL + "/async-result/" + taskID,
		Client:    z.httpc,
		Transport: z.tx,
		Timeout:   z.timeout,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Authorization", "Bearer "+z.apiKey)
		},
	})
	if err != nil {
		return TaskStatus{}, fmt.Errorf("poll: %w", err)
	}
	defer resp.Body.Close()

	var parsed zhipuPollResp
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("decode poll response: %w", err)
	}

	st := TaskStatus{
		Provider: z.Name(),
		Model:    parsed.Model,
		Status:   strings.ToLower(parsed.TaskStatus),
	}
	switch strings.ToUpper(parsed.TaskStatus) {
	case "SUCCESS":
		st.Done = true
		st.Status = "success"
		if len(parsed.VideoResult) > 0 {
			st.VideoURL = parsed.VideoResult[0].URL
			st.CoverURL = parsed.VideoResult[0].CoverURL
		}
	case "FAIL", "FAILED":
		st.Done = true
		st.Status = "failed"
		if parsed.Error != nil {
			st.Error = parsed.Error.Message
		} else {
			st.Error = "任务失败"
		}
	}
	return st, nil
}
