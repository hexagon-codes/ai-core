package video

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const defaultVeoBase = "https://generativelanguage.googleapis.com/v1beta"

type veoProvider struct {
	apiKey    string
	baseURL   string
	models    []string
	httpc     *http.Client
	timeout   time.Duration
	headers   map[string]string
	policy    *llm.NetworkPolicy
	transport *transport.Transport
}

type VeoOption func(*veoProvider)

func NewVeo(apiKey string, opts ...VeoOption) Provider {
	p := &veoProvider{
		apiKey:  apiKey,
		baseURL: defaultVeoBase,
		models:  []string{"veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.0-generate-001", "veo-3.0-fast-generate-001"},
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

func VeoWithBaseURL(baseURL string) VeoOption {
	return func(p *veoProvider) {
		if strings.TrimSpace(baseURL) != "" {
			p.baseURL = baseURL
		}
	}
}

func VeoWithModels(models []string) VeoOption {
	return func(p *veoProvider) {
		if len(models) > 0 {
			p.models = append([]string(nil), models...)
		}
	}
}

func VeoWithHTTPClient(client *http.Client) VeoOption {
	return func(p *veoProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

func VeoWithHeaders(headers map[string]string) VeoOption {
	return func(p *veoProvider) {
		p.headers = cloneStringMap(headers)
	}
}

func VeoWithNetworkPolicy(policy llm.NetworkPolicy) VeoOption {
	return func(p *veoProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *veoProvider) Name() string { return "google-veo" }

func (p *veoProvider) SupportedModels() []string {
	return append([]string(nil), p.models...)
}

func (p *veoProvider) Capabilities() []catalog.Capability {
	rows := make([]catalog.Capability, 0, len(p.models))
	for _, model := range p.models {
		rows = append(rows, catalog.Capability{
			Provider:    p.Name(),
			Model:       model,
			Name:        model,
			Modality:    catalog.ModalityVideo,
			Features:    []string{catalog.FeatureAsyncTask, catalog.FeatureImageInput, catalog.FeatureAudioOutput},
			Async:       true,
			Stable:      true,
			OfficialAPI: true,
			Source:      "https://ai.google.dev/gemini-api/docs/veo",
		})
	}
	return rows
}

func (p *veoProvider) Submit(ctx context.Context, req Request) (string, error) {
	if p.apiKey == "" {
		return "", fmt.Errorf("google-veo: 未配置 API Key")
	}
	model := req.Model
	if model == "" && len(p.models) > 0 {
		model = p.models[0]
	}
	instance := map[string]any{"prompt": req.Prompt}
	if req.ImageURL != "" {
		instance["image"] = map[string]any{"uri": req.ImageURL}
	}
	parameters := map[string]any{}
	if req.Ratio != "" {
		parameters["aspectRatio"] = req.Ratio
	}
	if req.Duration > 0 {
		parameters["durationSeconds"] = req.Duration
	}
	if req.WithAudio {
		parameters["generateAudio"] = true
	}
	if req.EndImageURL != "" {
		parameters["lastFrame"] = map[string]any{"uri": req.EndImageURL}
	}
	mergeExtra(parameters, req.Extra)
	body := map[string]any{"instances": []any{instance}}
	if len(parameters) > 0 {
		body["parameters"] = parameters
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("google-veo: marshal submit request: %w", err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "video.submit",
		Method:    http.MethodPost,
		URL:       p.baseURL + "/models/" + url.PathEscape(model) + ":predictLongRunning?key=" + url.QueryEscape(p.apiKey),
		Body:      payload,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		Retry:     media.SubmitRetryPolicy(req.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("x-goog-api-key", p.apiKey)
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
		return "", fmt.Errorf("google-veo: decode submit response: %w", err)
	}
	name := firstJSONPath(parsed, "name")
	if name == "" {
		return "", fmt.Errorf("google-veo: submit response missing operation name")
	}
	return name, nil
}

func (p *veoProvider) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	if p.apiKey == "" {
		return TaskStatus{}, fmt.Errorf("google-veo: 未配置 API Key")
	}
	path := strings.TrimPrefix(taskID, "/")
	if !strings.HasPrefix(path, "operations/") && !strings.HasPrefix(path, "models/") {
		path = "operations/" + path
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.Name(),
		Action:    "video.poll",
		Method:    http.MethodGet,
		URL:       p.baseURL + "/" + path + "?key=" + url.QueryEscape(p.apiKey),
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("x-goog-api-key", p.apiKey)
		},
	})
	if err != nil {
		return TaskStatus{}, err
	}
	defer resp.Body.Close()
	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("google-veo: decode poll response: %w", err)
	}
	done := firstJSONBool(parsed, "done")
	rawStatus := "processing"
	if done {
		rawStatus = "succeeded"
		if firstJSONPath(parsed, "error.message") != "" {
			rawStatus = "failed"
		}
	}
	state := media.NormalizeTaskState(rawStatus)
	return TaskStatus{
		TaskID:    taskID,
		Provider:  p.Name(),
		Model:     firstJSONPath(parsed, "metadata.model"),
		Status:    statusString(state, rawStatus),
		State:     state,
		Done:      done,
		VideoURL:  firstJSONPath(parsed, "response.generateVideoResponse.generatedSamples.0.video.uri", "response.generatedVideos.0.video.uri", "response.video.uri"),
		Error:     firstJSONPath(parsed, "error.message"),
		RequestID: firstJSONPath(parsed, "name"),
		RawStatus: rawStatus,
	}, nil
}

func firstJSONBool(root map[string]any, paths ...string) bool {
	for _, path := range paths {
		var current any = root
		ok := true
		for _, part := range strings.Split(path, ".") {
			m, isMap := current.(map[string]any)
			if !isMap {
				ok = false
				break
			}
			current, ok = m[part]
			if !ok {
				break
			}
		}
		if ok {
			if b, isBool := current.(bool); isBool {
				return b
			}
		}
	}
	return false
}

var _ catalog.Provider = (*veoProvider)(nil)
