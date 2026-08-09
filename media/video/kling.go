package video

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const defaultKlingBase = "https://api-singapore.klingai.com/v1"

type klingProvider struct {
	accessKey string
	secretKey string
	bearer    string
	baseURL   string
	models    []string
	httpc     *http.Client
	timeout   time.Duration
	headers   map[string]string
	policy    *llm.NetworkPolicy
	transport *transport.Transport
	now       func() time.Time
}

// KlingOption configures Kling Provider.
type KlingOption func(*klingProvider)

// NewKling creates a Kling video Provider.
//
// If secretKey is empty, accessKey is treated as a ready-to-use Bearer token.
// If both are present, the Provider signs official Kling HS256 JWT tokens.
func NewKling(accessKey, secretKey string, opts ...KlingOption) Provider {
	p := &klingProvider{
		accessKey: accessKey,
		secretKey: secretKey,
		baseURL:   defaultKlingBase,
		models:    []string{"kling-v3", "kling-v2-1-master", "kling-v1-6", "kling-v1"},
		httpc:     httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
		timeout:   60 * time.Second,
		now:       time.Now,
	}
	if secretKey == "" {
		p.bearer = accessKey
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	p.baseURL = strings.TrimRight(p.baseURL, "/")
	p.transport = transport.NewTransport(p.httpc, p.policy)
	return p
}

func KlingWithBaseURL(baseURL string) KlingOption {
	return func(p *klingProvider) {
		if strings.TrimSpace(baseURL) != "" {
			p.baseURL = baseURL
		}
	}
}

func KlingWithModels(models []string) KlingOption {
	return func(p *klingProvider) {
		if len(models) > 0 {
			p.models = append([]string(nil), models...)
		}
	}
}

func KlingWithHTTPClient(client *http.Client) KlingOption {
	return func(p *klingProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

func KlingWithHeaders(headers map[string]string) KlingOption {
	return func(p *klingProvider) {
		p.headers = cloneStringMap(headers)
	}
}

func KlingWithNetworkPolicy(policy llm.NetworkPolicy) KlingOption {
	return func(p *klingProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *klingProvider) Name() string { return "kling" }

func (p *klingProvider) SupportedModels() []string {
	return append([]string(nil), p.models...)
}

func (p *klingProvider) Capabilities() []catalog.Capability {
	rows := make([]catalog.Capability, 0, len(p.models))
	for _, model := range p.models {
		rows = append(rows, catalog.Capability{
			Provider:    p.Name(),
			Model:       model,
			Name:        model,
			Modality:    catalog.ModalityVideo,
			Features:    []string{catalog.FeatureAsyncTask, catalog.FeatureImageInput, catalog.FeatureNegativePrompt, catalog.FeatureCallback},
			Async:       true,
			Stable:      true,
			OfficialAPI: true,
			Source:      "https://kling.ai/document-api",
		})
	}
	return rows
}

func (p *klingProvider) Submit(ctx context.Context, req Request) (string, error) {
	if p.accessKey == "" && p.bearer == "" {
		return "", fmt.Errorf("kling: 未配置 access key 或 bearer token")
	}
	model := req.Model
	if model == "" && len(p.models) > 0 {
		model = p.models[0]
	}
	kind := "text2video"
	body := map[string]any{
		"model_name": model,
		"prompt":     req.Prompt,
	}
	if req.ImageURL != "" {
		kind = "image2video"
		body["image"] = req.ImageURL
		body["image_url"] = req.ImageURL
	}
	if req.EndImageURL != "" {
		body["image_tail"] = req.EndImageURL
	}
	if req.Negative != "" {
		body["negative_prompt"] = req.Negative
	}
	if req.Ratio != "" {
		body["aspect_ratio"] = req.Ratio
	}
	if req.Duration > 0 {
		body["duration"] = req.Duration
	}
	if req.Quality != "" {
		body["mode"] = req.Quality
	}
	if req.CallbackURL != "" {
		body["callback_url"] = req.CallbackURL
	}
	if req.IdempotencyKey != "" {
		body["external_task_id"] = req.IdempotencyKey
	}
	mergeExtra(body, req.Extra)

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("kling: marshal submit request: %w", err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "video.submit",
		Method:    http.MethodPost,
		URL:       p.baseURL + "/videos/" + kind,
		Body:      payload,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		Retry:     media.SubmitRetryPolicy(req.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Authorization", "Bearer "+p.token())
		},
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("kling: decode submit response: %w", err)
	}
	taskID := firstJSONPath(parsed, "data.task_id", "task_id", "id")
	if taskID == "" {
		return "", fmt.Errorf("kling: submit response missing task id")
	}
	return kind + ":" + taskID, nil
}

func (p *klingProvider) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	kind, rawID, ok := strings.Cut(taskID, ":")
	if !ok {
		kind = "text2video"
		rawID = taskID
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "video.poll",
		Method:    http.MethodGet,
		URL:       p.baseURL + "/videos/" + kind + "/" + rawID,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Authorization", "Bearer "+p.token())
		},
	})
	if err != nil {
		return TaskStatus{}, err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("kling: decode poll response: %w", err)
	}
	rawStatus := firstJSONPath(parsed, "data.task_status", "data.status", "task_status", "status")
	state := media.NormalizeTaskState(rawStatus)
	return TaskStatus{
		TaskID:    taskID,
		Provider:  p.Name(),
		Model:     firstJSONPath(parsed, "data.model_name", "data.model", "model_name", "model"),
		Status:    statusString(state, rawStatus),
		State:     state,
		Done:      state.Terminal(),
		VideoURL:  firstJSONPath(parsed, "data.task_result.videos.0.url", "data.task_result.video.url", "data.video_url", "url"),
		CoverURL:  firstJSONPath(parsed, "data.task_result.videos.0.cover_image_url", "data.cover_url", "cover_url"),
		Error:     firstJSONPath(parsed, "data.task_status_msg", "data.error.message", "error.message", "message"),
		RequestID: firstJSONPath(parsed, "request_id", "data.request_id", "data.task_id"),
		RawStatus: rawStatus,
	}, nil
}

func (p *klingProvider) token() string {
	if p.bearer != "" || p.secretKey == "" {
		return p.bearer
	}
	now := p.now()
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	claims, _ := json.Marshal(map[string]any{
		"iss": p.accessKey,
		"exp": now.Add(30 * time.Minute).Unix(),
		"nbf": now.Add(-5 * time.Second).Unix(),
	})
	payload := base64.RawURLEncoding.EncodeToString(claims)
	mac := hmac.New(sha256.New, []byte(p.secretKey))
	_, _ = mac.Write([]byte(header + "." + payload))
	signature := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	return header + "." + payload + "." + signature
}

var _ catalog.Provider = (*klingProvider)(nil)
