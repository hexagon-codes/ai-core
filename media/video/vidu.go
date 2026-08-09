package video

import (
	"context"
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

const defaultViduBase = "https://api.vidu.com"

type viduProvider struct {
	apiKey    string
	baseURL   string
	models    []string
	httpc     *http.Client
	timeout   time.Duration
	headers   map[string]string
	policy    *llm.NetworkPolicy
	transport *transport.Transport
}

type ViduOption func(*viduProvider)

func NewVidu(apiKey string, opts ...ViduOption) Provider {
	p := &viduProvider{
		apiKey:  apiKey,
		baseURL: defaultViduBase,
		models:  []string{"viduq3-pro-fast", "viduq3-turbo", "viduq3-pro", "viduq2-pro-fast", "viduq2-pro", "viduq2-turbo", "viduq2", "viduq1", "vidu2.0"},
		httpc:   httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
		timeout: 60 * time.Second,
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

func ViduWithBaseURL(baseURL string) ViduOption {
	return func(p *viduProvider) {
		if strings.TrimSpace(baseURL) != "" {
			p.baseURL = baseURL
		}
	}
}

func ViduWithModels(models []string) ViduOption {
	return func(p *viduProvider) {
		if len(models) > 0 {
			p.models = append([]string(nil), models...)
		}
	}
}

func ViduWithHTTPClient(client *http.Client) ViduOption {
	return func(p *viduProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

func ViduWithHeaders(headers map[string]string) ViduOption {
	return func(p *viduProvider) {
		p.headers = cloneStringMap(headers)
	}
}

func ViduWithNetworkPolicy(policy llm.NetworkPolicy) ViduOption {
	return func(p *viduProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *viduProvider) Name() string { return "vidu" }

func (p *viduProvider) SupportedModels() []string {
	return append([]string(nil), p.models...)
}

func (p *viduProvider) Capabilities() []catalog.Capability {
	rows := make([]catalog.Capability, 0, len(p.models))
	for _, model := range p.models {
		rows = append(rows, catalog.Capability{
			Provider:    p.Name(),
			Model:       model,
			Name:        model,
			Modality:    catalog.ModalityVideo,
			Features:    []string{catalog.FeatureAsyncTask, catalog.FeatureImageInput, catalog.FeatureCallback, catalog.FeatureSeed},
			Async:       true,
			Stable:      true,
			OfficialAPI: true,
			Source:      "https://platform.vidu.com/docs/text-to-video",
		})
	}
	return rows
}

func (p *viduProvider) Submit(ctx context.Context, req Request) (string, error) {
	if p.apiKey == "" {
		return "", fmt.Errorf("vidu: 未配置 API Key")
	}
	model := req.Model
	if model == "" && len(p.models) > 0 {
		model = p.models[0]
	}
	path := "/ent/v2/text2video"
	body := map[string]any{
		"model":  model,
		"prompt": req.Prompt,
	}
	if req.ImageURL != "" {
		path = "/ent/v2/img2video"
		body["images"] = []string{req.ImageURL}
		body["image"] = req.ImageURL
	}
	if req.EndImageURL != "" {
		if images, ok := body["images"].([]string); ok {
			body["images"] = append(images, req.EndImageURL)
		} else {
			body["images"] = []string{req.ImageURL, req.EndImageURL}
		}
	}
	if req.StyleOrQuality() != "" {
		body["style"] = req.StyleOrQuality()
	}
	if req.Duration > 0 {
		body["duration"] = req.Duration
	}
	if req.Size != "" {
		body["resolution"] = req.Size
	}
	if req.Ratio != "" {
		body["aspect_ratio"] = req.Ratio
	}
	if req.Seed != 0 {
		body["seed"] = req.Seed
	}
	if req.CallbackURL != "" {
		body["callback_url"] = req.CallbackURL
	}
	mergeExtra(body, req.Extra)

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("vidu: marshal submit request: %w", err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "video.submit",
		Method:    http.MethodPost,
		URL:       p.baseURL + path,
		Body:      payload,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		Retry:     media.SubmitRetryPolicy(req.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Authorization", "Token "+p.apiKey)
			if req.IdempotencyKey != "" {
				r.Header.Set("Idempotency-Key", req.IdempotencyKey)
			}
		},
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("vidu: decode submit response: %w", err)
	}
	taskID := firstJSONPath(parsed, "task_id", "id", "data.task_id", "data.id")
	if taskID == "" {
		return "", fmt.Errorf("vidu: submit response missing task id")
	}
	return taskID, nil
}

func (p *viduProvider) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "video.poll",
		Method:    http.MethodGet,
		URL:       p.baseURL + "/ent/v2/tasks/" + taskID + "/creations",
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Authorization", "Token "+p.apiKey)
		},
	})
	if err != nil {
		return TaskStatus{}, err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("vidu: decode poll response: %w", err)
	}
	rawStatus := firstJSONPath(parsed, "status", "state", "data.status", "creations.0.status", "data.creations.0.status")
	state := media.NormalizeTaskState(rawStatus)
	return TaskStatus{
		TaskID:    taskID,
		Provider:  p.Name(),
		Model:     firstJSONPath(parsed, "model", "data.model"),
		Status:    statusString(state, rawStatus),
		State:     state,
		Done:      state.Terminal(),
		VideoURL:  firstJSONPath(parsed, "creations.0.url", "creations.0.video_url", "data.creations.0.url", "data.video_url", "video_url"),
		CoverURL:  firstJSONPath(parsed, "creations.0.cover_url", "data.creations.0.cover_url", "cover_url"),
		Error:     firstJSONPath(parsed, "error", "error.message", "data.error.message", "message"),
		RequestID: firstJSONPath(parsed, "request_id", "id", "task_id"),
		RawStatus: rawStatus,
	}, nil
}

func (r Request) StyleOrQuality() string {
	if r.Quality != "" {
		return r.Quality
	}
	return ""
}

var _ catalog.Provider = (*viduProvider)(nil)
