package image

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const defaultBFLBase = "https://api.bfl.ai/v1"

type fluxProvider struct {
	apiKey       string
	baseURL      string
	models       []string
	httpc        *http.Client
	timeout      time.Duration
	pollInterval time.Duration
	headers      map[string]string
	policy       *llm.NetworkPolicy
	transport    *transport.Transport
}

// FluxOption configures Black Forest Labs FLUX Provider.
type FluxOption func(*fluxProvider)

// NewFlux creates a Black Forest Labs FLUX Provider.
func NewFlux(apiKey string, opts ...FluxOption) interface {
	Provider
	AsyncImageProvider
	catalog.Provider
} {
	p := &fluxProvider{
		apiKey:       apiKey,
		baseURL:      defaultBFLBase,
		models:       []string{"flux-pro-1.1", "flux-pro-1.1-ultra", "flux-kontext-pro", "flux-kontext-max", "flux-pro-1.0", "flux-dev"},
		httpc:        httpx.RawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
		timeout:      60 * time.Second,
		pollInterval: 2 * time.Second,
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

func FluxWithBaseURL(baseURL string) FluxOption {
	return func(p *fluxProvider) {
		if strings.TrimSpace(baseURL) != "" {
			p.baseURL = baseURL
		}
	}
}

func FluxWithModels(models []string) FluxOption {
	return func(p *fluxProvider) {
		if len(models) > 0 {
			p.models = append([]string(nil), models...)
		}
	}
}

func FluxWithHTTPClient(client *http.Client) FluxOption {
	return func(p *fluxProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

func FluxWithNetworkPolicy(policy llm.NetworkPolicy) FluxOption {
	return func(p *fluxProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func FluxWithHeaders(headers map[string]string) FluxOption {
	return func(p *fluxProvider) {
		p.headers = cloneStringMap(headers)
	}
}

func (p *fluxProvider) Name() string { return "bfl-flux" }

func (p *fluxProvider) SupportedModels() []string {
	return append([]string(nil), p.models...)
}

func (p *fluxProvider) Capabilities() []catalog.Capability {
	rows := make([]catalog.Capability, 0, len(p.models))
	for _, model := range p.models {
		rows = append(rows, catalog.Capability{
			Provider:    p.Name(),
			Model:       model,
			Name:        model,
			Modality:    catalog.ModalityImage,
			Features:    []string{catalog.FeatureAsyncTask, catalog.FeatureSeed, catalog.FeatureImageInput},
			Async:       true,
			Stable:      true,
			OfficialAPI: true,
			Source:      "https://docs.bfl.ml/quick_start/generating_images",
		})
	}
	return rows
}

func (p *fluxProvider) Generate(ctx context.Context, req Request) (*Result, error) {
	st, err := SubmitAndWait(ctx, p, req, p.pollInterval)
	if err != nil {
		return nil, err
	}
	if st.Result == nil {
		return nil, fmt.Errorf("bfl-flux: task ended without result state=%s error=%s", st.State, st.Error)
	}
	return st.Result, nil
}

func (p *fluxProvider) Submit(ctx context.Context, req Request) (string, error) {
	if p.apiKey == "" {
		return "", fmt.Errorf("bfl-flux: 未配置 API Key")
	}
	model := req.Model
	if model == "" && len(p.models) > 0 {
		model = p.models[0]
	}
	body := map[string]any{
		"prompt": req.Prompt,
	}
	if req.Negative != "" {
		body["negative_prompt"] = req.Negative
	}
	if req.Seed != 0 {
		body["seed"] = req.Seed
	}
	if req.Ratio != "" {
		body["aspect_ratio"] = req.Ratio
	}
	if req.ImageURL != "" {
		body["input_image"] = req.ImageURL
	}
	if req.Size != "" {
		width, height := parseSize(req.Size)
		if width > 0 && height > 0 {
			body["width"] = width
			body["height"] = height
		}
	}
	mergeExtra(body, req.Extra)
	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("bfl-flux: marshal submit request: %w", err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "image.submit",
		Method:    http.MethodPost,
		URL:       p.baseURL + "/" + model,
		Body:      payload,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		Retry:     media.SubmitRetryPolicy(req.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("x-key", p.apiKey)
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
		return "", fmt.Errorf("bfl-flux: decode submit response: %w", err)
	}
	id := firstJSONPath(parsed, "id", "request_id", "data.id")
	if id == "" {
		return "", fmt.Errorf("bfl-flux: submit response missing id")
	}
	return model + ":" + id, nil
}

func (p *fluxProvider) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	model, rawID, ok := strings.Cut(taskID, ":")
	if !ok {
		rawID = taskID
	}
	resultURL := p.baseURL + "/get_result?id=" + url.QueryEscape(rawID)
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "image.poll",
		Method:    http.MethodGet,
		URL:       resultURL,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("x-key", p.apiKey)
		},
	})
	if err != nil {
		return TaskStatus{}, err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("bfl-flux: decode poll response: %w", err)
	}
	rawStatus := firstJSONPath(parsed, "status", "state")
	state := media.NormalizeTaskState(rawStatus)
	st := TaskStatus{
		TaskID:    taskID,
		State:     state,
		Error:     firstJSONPath(parsed, "error", "error.message"),
		RequestID: firstJSONPath(parsed, "id", "request_id"),
		RawStatus: rawStatus,
	}
	imageURL := firstJSONPath(parsed, "result.sample", "result.image", "result.url", "image_url", "url")
	if state == media.TaskSucceeded && imageURL != "" {
		st.Result = &Result{
			Provider:  p.Name(),
			Model:     model,
			Created:   time.Now().Unix(),
			Images:    []Image{{URL: imageURL}},
			RequestID: st.RequestID,
		}
	}
	return st, nil
}

func parseSize(size string) (int, int) {
	left, right, ok := strings.Cut(strings.ToLower(strings.TrimSpace(size)), "x")
	if !ok {
		return 0, 0
	}
	width, err1 := strconv.Atoi(strings.TrimSpace(left))
	height, err2 := strconv.Atoi(strings.TrimSpace(right))
	if err1 != nil || err2 != nil {
		return 0, 0
	}
	return width, height
}
