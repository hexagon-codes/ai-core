package image

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

// CompatibleConfig describes an async image API exposed by a compliant gateway.
type CompatibleConfig struct {
	Name             string
	BaseURL          string
	SubmitPath       string
	PollPathTemplate string
	AuthHeader       string
	AuthPrefix       string
	Models           []string
	Source           string
	OfficialAPI      bool

	SubmitIDPaths []string
	StatusPaths   []string
	ImageURLPaths []string
	ImageB64Paths []string
	ErrorPaths    []string
}

type compatibleProvider struct {
	cfg       CompatibleConfig
	apiKey    string
	httpc     *http.Client
	timeout   time.Duration
	headers   map[string]string
	policy    *llm.NetworkPolicy
	transport *transport.Transport
}

type CompatibleOption func(*compatibleProvider)

func NewAsyncCompatible(apiKey string, cfg CompatibleConfig, opts ...CompatibleOption) interface {
	AsyncImageProvider
	catalog.Provider
} {
	cfg = normalizeCompatibleConfig(cfg)
	p := &compatibleProvider{
		cfg:     cfg,
		apiKey:  apiKey,
		httpc:   httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
		timeout: 60 * time.Second,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	p.transport = transport.NewTransport(p.httpc, p.policy)
	return p
}

func NewMidjourneyCompatible(apiKey, baseURL string, opts ...CompatibleOption) interface {
	AsyncImageProvider
	catalog.Provider
} {
	return NewAsyncCompatible(apiKey, CompatibleConfig{
		Name:             "midjourney-compatible",
		BaseURL:          baseURL,
		SubmitPath:       "/v1/images/generations",
		PollPathTemplate: "/v1/images/generations/{task_id}",
		Models:           []string{"midjourney-v7", "midjourney-v6.1", "niji-v6"},
		Source:           "compatible-gateway",
		OfficialAPI:      false,
	}, opts...)
}

func CompatibleWithHTTPClient(client *http.Client) CompatibleOption {
	return func(p *compatibleProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

func CompatibleWithHeaders(headers map[string]string) CompatibleOption {
	return func(p *compatibleProvider) {
		p.headers = cloneStringMap(headers)
	}
}

func CompatibleWithNetworkPolicy(policy llm.NetworkPolicy) CompatibleOption {
	return func(p *compatibleProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *compatibleProvider) Name() string { return p.cfg.Name }

func (p *compatibleProvider) SupportedModels() []string {
	return append([]string(nil), p.cfg.Models...)
}

func (p *compatibleProvider) Capabilities() []catalog.Capability {
	rows := make([]catalog.Capability, 0, len(p.cfg.Models))
	for _, model := range p.cfg.Models {
		rows = append(rows, catalog.Capability{
			Provider:    p.cfg.Name,
			Model:       model,
			Name:        model,
			Modality:    catalog.ModalityImage,
			Features:    []string{catalog.FeatureAsyncTask, catalog.FeatureImageInput},
			Async:       true,
			Stable:      p.cfg.OfficialAPI,
			OfficialAPI: p.cfg.OfficialAPI,
			Source:      p.cfg.Source,
		})
	}
	return rows
}

func (p *compatibleProvider) Submit(ctx context.Context, req Request) (string, error) {
	model := req.Model
	if model == "" && len(p.cfg.Models) > 0 {
		model = p.cfg.Models[0]
	}
	body := map[string]any{
		"model":  model,
		"prompt": req.Prompt,
	}
	if req.Negative != "" {
		body["negative_prompt"] = req.Negative
	}
	if req.ImageURL != "" {
		body["image_url"] = req.ImageURL
	}
	if req.Size != "" {
		body["size"] = req.Size
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
	if req.IdempotencyKey != "" {
		body["idempotency_key"] = req.IdempotencyKey
	}
	mergeExtra(body, req.Extra)
	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("%s: marshal submit request: %w", p.cfg.Name, err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.cfg.Name,
		Action:    "image.submit",
		Method:    http.MethodPost,
		URL:       p.cfg.BaseURL + p.cfg.SubmitPath,
		Body:      payload,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		Retry:     media.SubmitRetryPolicy(req.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			p.setAuth(r)
		},
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("%s: decode submit response: %w", p.cfg.Name, err)
	}
	taskID := firstJSONPath(parsed, p.cfg.SubmitIDPaths...)
	if taskID == "" {
		return "", fmt.Errorf("%s: submit response missing task id", p.cfg.Name)
	}
	return taskID, nil
}

func (p *compatibleProvider) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.cfg.Name,
		Action:    "image.poll",
		Method:    http.MethodGet,
		URL:       p.cfg.BaseURL + strings.ReplaceAll(p.cfg.PollPathTemplate, "{task_id}", taskID),
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		SetHeaders: func(r *http.Request) {
			p.setAuth(r)
		},
	})
	if err != nil {
		return TaskStatus{}, err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("%s: decode poll response: %w", p.cfg.Name, err)
	}
	rawStatus := firstJSONPath(parsed, p.cfg.StatusPaths...)
	state := media.NormalizeTaskState(rawStatus)
	st := TaskStatus{
		TaskID:    taskID,
		State:     state,
		Error:     firstJSONPath(parsed, p.cfg.ErrorPaths...),
		RequestID: firstJSONPath(parsed, "request_id", "id", "task_id", "data.id", "data.task_id"),
		RawStatus: rawStatus,
	}
	imageURL := firstJSONPath(parsed, p.cfg.ImageURLPaths...)
	imageB64 := firstJSONPath(parsed, p.cfg.ImageB64Paths...)
	if state == media.TaskSucceeded && (imageURL != "" || imageB64 != "") {
		st.Result = &Result{
			Provider:  p.cfg.Name,
			Model:     firstJSONPath(parsed, "model", "data.model"),
			Created:   time.Now().Unix(),
			Images:    []Image{{URL: imageURL, B64JSON: imageB64}},
			RequestID: st.RequestID,
		}
	}
	return st, nil
}

func (p *compatibleProvider) setAuth(r *http.Request) {
	if p.apiKey == "" || p.cfg.AuthHeader == "" {
		return
	}
	r.Header.Set(p.cfg.AuthHeader, p.cfg.AuthPrefix+p.apiKey)
}

func normalizeCompatibleConfig(cfg CompatibleConfig) CompatibleConfig {
	cfg.Name = strings.TrimSpace(cfg.Name)
	if cfg.Name == "" {
		cfg.Name = "image-compatible"
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	cfg.SubmitPath = slashPath(cfg.SubmitPath)
	cfg.PollPathTemplate = slashPath(cfg.PollPathTemplate)
	if cfg.AuthHeader == "" {
		cfg.AuthHeader = "Authorization"
	}
	if cfg.AuthPrefix == "" {
		cfg.AuthPrefix = "Bearer "
	}
	if len(cfg.SubmitIDPaths) == 0 {
		cfg.SubmitIDPaths = []string{"task_id", "id", "data.task_id", "data.id"}
	}
	if len(cfg.StatusPaths) == 0 {
		cfg.StatusPaths = []string{"status", "state", "data.status", "data.state", "task_status"}
	}
	if len(cfg.ImageURLPaths) == 0 {
		cfg.ImageURLPaths = []string{"image_url", "url", "data.image_url", "data.url", "data.output.image_url", "output.image_url"}
	}
	if len(cfg.ImageB64Paths) == 0 {
		cfg.ImageB64Paths = []string{"b64_json", "data.b64_json", "data.image_b64", "output.b64_json"}
	}
	if len(cfg.ErrorPaths) == 0 {
		cfg.ErrorPaths = []string{"error.message", "data.error.message", "message"}
	}
	return cfg
}

func slashPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return path
	}
	if strings.HasPrefix(path, "/") {
		return path
	}
	return "/" + path
}

var _ catalog.Provider = (*compatibleProvider)(nil)
