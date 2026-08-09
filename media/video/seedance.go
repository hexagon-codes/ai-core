package video

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const (
	defaultSeedanceCNBase     = "https://ark.cn-beijing.volces.com/api/v3"
	defaultSeedanceGlobalBase = "https://ark.ap-southeast.bytepluses.com/api/v3"
)

// SeedanceRegion identifies the ModelArk route. The model IDs and billing
// route must not be mixed across regions.
type SeedanceRegion string

const (
	SeedanceRegionCN     SeedanceRegion = "cn"
	SeedanceRegionGlobal SeedanceRegion = "global"
)

// SeedanceOption configures Seedance/Doubao video Provider.
type SeedanceOption func(*seedanceProvider)

type seedanceProvider struct {
	name      string
	region    SeedanceRegion
	apiKey    string
	baseURL   string
	models    []string
	httpc     *http.Client
	timeout   time.Duration
	headers   map[string]string
	policy    *llm.NetworkPolicy
	transport *transport.Transport
}

// NewSeedanceCN creates a Volcengine Ark China-region Seedance Provider.
func NewSeedanceCN(apiKey string, opts ...SeedanceOption) Provider {
	return newSeedance("seedance-cn", SeedanceRegionCN, apiKey, opts...)
}

// NewSeedanceGlobal creates a BytePlus ModelArk global Seedance Provider.
func NewSeedanceGlobal(apiKey string, opts ...SeedanceOption) Provider {
	return newSeedance("seedance-global", SeedanceRegionGlobal, apiKey, opts...)
}

// NewDoubaoSeedanceCN is an alias for the China-region Doubao/Seedance route.
func NewDoubaoSeedanceCN(apiKey string, opts ...SeedanceOption) Provider {
	return newSeedance("doubao-seedance-cn", SeedanceRegionCN, apiKey, opts...)
}

// NewDoubaoSeedanceGlobal is an alias for the global BytePlus route.
func NewDoubaoSeedanceGlobal(apiKey string, opts ...SeedanceOption) Provider {
	return newSeedance("doubao-seedance-global", SeedanceRegionGlobal, apiKey, opts...)
}

func newSeedance(name string, region SeedanceRegion, apiKey string, opts ...SeedanceOption) Provider {
	baseURL := defaultSeedanceCNBase
	models := []string{
		"doubao-seedance-2-0-260128",
		"doubao-seedance-2-0-fast-260128",
		"doubao-seedance-1-0-pro-250528",
		"doubao-seedance-1-0-lite-t2v-250428",
		"doubao-seedance-1-0-lite-i2v-250428",
	}
	if region == SeedanceRegionGlobal {
		baseURL = defaultSeedanceGlobalBase
		models = []string{
			"dreamina-seedance-2-0-260128",
			"dreamina-seedance-2-0-fast-260128",
			"seedance-1-0-pro-250528",
			"seedance-1-0-lite-t2v-250428",
			"seedance-1-0-lite-i2v-250428",
		}
	}
	p := &seedanceProvider{
		name:    name,
		region:  region,
		apiKey:  apiKey,
		baseURL: baseURL,
		models:  models,
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

// SeedanceWithBaseURL overrides the ModelArk base URL.
func SeedanceWithBaseURL(baseURL string) SeedanceOption {
	return func(p *seedanceProvider) {
		if strings.TrimSpace(baseURL) != "" {
			p.baseURL = baseURL
		}
	}
}

// SeedanceWithModels overrides supported model IDs.
func SeedanceWithModels(models []string) SeedanceOption {
	return func(p *seedanceProvider) {
		if len(models) > 0 {
			p.models = append([]string(nil), models...)
		}
	}
}

// SeedanceWithHTTPClient injects a custom HTTP client.
func SeedanceWithHTTPClient(client *http.Client) SeedanceOption {
	return func(p *seedanceProvider) {
		if client != nil {
			p.httpc = client
		}
	}
}

// SeedanceWithRequestTimeout sets request timeout for submit/poll calls.
func SeedanceWithRequestTimeout(timeout time.Duration) SeedanceOption {
	return func(p *seedanceProvider) {
		p.timeout = timeout
	}
}

// SeedanceWithHeaders sets safe extra headers.
func SeedanceWithHeaders(headers map[string]string) SeedanceOption {
	return func(p *seedanceProvider) {
		p.headers = cloneStringMap(headers)
	}
}

// SeedanceWithNetworkPolicy enables SSRF/network policy enforcement.
func SeedanceWithNetworkPolicy(policy llm.NetworkPolicy) SeedanceOption {
	return func(p *seedanceProvider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

func (p *seedanceProvider) Name() string { return p.name }

func (p *seedanceProvider) SupportedModels() []string {
	return append([]string(nil), p.models...)
}

func (p *seedanceProvider) Capabilities() []catalog.Capability {
	rows := make([]catalog.Capability, 0, len(p.models))
	for _, model := range p.models {
		rows = append(rows, catalog.Capability{
			Provider:    p.name,
			Model:       model,
			Name:        model,
			Modality:    catalog.ModalityVideo,
			Features:    []string{catalog.FeatureAsyncTask, catalog.FeatureImageInput, catalog.FeatureAudioOutput, catalog.FeatureCallback, "region:" + string(p.region)},
			Async:       true,
			Stable:      true,
			OfficialAPI: true,
			Source:      "https://docs.byteplus.com/en/docs/ModelArk/1520757",
		})
	}
	return rows
}

func (p *seedanceProvider) Submit(ctx context.Context, req Request) (string, error) {
	if p.apiKey == "" {
		return "", fmt.Errorf("%s: 未配置 API Key", p.name)
	}
	model := req.Model
	if model == "" && len(p.models) > 0 {
		model = p.models[0]
	}
	body := map[string]any{
		"model":   model,
		"content": p.content(req),
	}
	mergeExtra(body, req.Extra)
	if req.CallbackURL != "" {
		body["callback_url"] = req.CallbackURL
	}
	if req.WithAudio {
		body["generate_audio"] = true
	}
	output := map[string]any{}
	if req.Size != "" {
		output["resolution"] = req.Size
	}
	if req.Ratio != "" {
		output["aspect_ratio"] = req.Ratio
	}
	if req.Duration > 0 {
		output["duration"] = req.Duration
		output["duration_s"] = req.Duration
	}
	if req.FPS > 0 {
		output["fps"] = req.FPS
	}
	if len(output) > 0 {
		body["output"] = output
	}
	if req.Seed != 0 {
		body["seed"] = req.Seed
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("%s: marshal submit request: %w", p.name, err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.name,
		Action:    "video.submit",
		Method:    http.MethodPost,
		URL:       p.baseURL + "/contents/generations/tasks",
		Body:      payload,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		Retry:     media.SubmitRetryPolicy(req.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("Authorization", "Bearer "+p.apiKey)
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
		return "", fmt.Errorf("%s: decode submit response: %w", p.name, err)
	}
	taskID := firstJSONPath(parsed, "id", "task_id", "data.id", "data.task_id")
	if taskID == "" {
		return "", fmt.Errorf("%s: submit response missing task id", p.name)
	}
	return taskID, nil
}

func (p *seedanceProvider) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	if p.apiKey == "" {
		return TaskStatus{}, fmt.Errorf("%s: 未配置 API Key", p.name)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  p.name,
		Action:    "video.poll",
		Method:    http.MethodGet,
		URL:       p.baseURL + "/contents/generations/tasks/" + taskID,
		Client:    p.httpc,
		Transport: p.transport,
		Headers:   p.headers,
		Timeout:   p.timeout,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Authorization", "Bearer "+p.apiKey)
		},
	})
	if err != nil {
		return TaskStatus{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return TaskStatus{}, fmt.Errorf("%s: read poll response: %w", p.name, err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return TaskStatus{}, fmt.Errorf("%s: decode poll response: %w", p.name, err)
	}

	rawStatus := firstJSONPath(parsed, "status", "task_status", "data.status", "data.task_status")
	state := media.NormalizeTaskState(rawStatus)
	videoURL := firstJSONPath(parsed,
		"content.video_url",
		"content.url",
		"data.content.video_url",
		"data.content.url",
		"data.video_url",
		"result.video_url",
		"video_url",
	)
	coverURL := firstJSONPath(parsed, "content.cover_url", "data.content.cover_url", "cover_url")
	errorMessage := firstJSONPath(parsed, "error.message", "data.error.message", "message")
	model := firstJSONPath(parsed, "model", "data.model")
	requestID := firstJSONPath(parsed, "request_id", "data.request_id", "trace_id", "id")
	return TaskStatus{
		TaskID:    taskID,
		Provider:  p.name,
		Model:     model,
		Status:    statusString(state, rawStatus),
		State:     state,
		Done:      state.Terminal(),
		VideoURL:  videoURL,
		CoverURL:  coverURL,
		Error:     errorMessage,
		RequestID: requestID,
		RawStatus: rawStatus,
	}, nil
}

func (p *seedanceProvider) content(req Request) []map[string]any {
	content := make([]map[string]any, 0, 3)
	if strings.TrimSpace(req.Prompt) != "" {
		content = append(content, map[string]any{"type": "text", "text": req.Prompt})
	}
	if strings.TrimSpace(req.ImageURL) != "" {
		content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": req.ImageURL}})
	}
	if strings.TrimSpace(req.EndImageURL) != "" {
		content = append(content, map[string]any{"type": "image_url", "role": "last_frame", "image_url": map[string]any{"url": req.EndImageURL}})
	}
	return content
}

func statusString(state media.TaskState, raw string) string {
	switch state {
	case media.TaskSucceeded:
		return "success"
	case media.TaskFailed:
		return "failed"
	case media.TaskCancelled:
		return "cancelled"
	case media.TaskQueued:
		return "queueing"
	default:
		if strings.TrimSpace(raw) == "" {
			return "running"
		}
		return strings.ToLower(raw)
	}
}

func firstJSONPath(root map[string]any, paths ...string) string {
	for _, path := range paths {
		var current any = root
		ok := true
		for _, part := range strings.Split(path, ".") {
			switch node := current.(type) {
			case map[string]any:
				current, ok = node[part]
			case []any:
				index, err := strconv.Atoi(part)
				if err != nil || index < 0 || index >= len(node) {
					ok = false
					break
				}
				current = node[index]
			default:
				ok = false
			}
			if !ok {
				break
			}
		}
		if ok {
			if value := anyToString(current); value != "" {
				return value
			}
		}
	}
	return ""
}

func anyToString(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case json.Number:
		return v.String()
	case fmt.Stringer:
		return v.String()
	default:
		return ""
	}
}

func mergeExtra(dst map[string]any, extra map[string]any) {
	for k, v := range extra {
		if strings.TrimSpace(k) == "" {
			continue
		}
		dst[k] = v
	}
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

var _ catalog.Provider = (*seedanceProvider)(nil)
