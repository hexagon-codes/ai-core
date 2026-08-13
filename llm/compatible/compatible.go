// Package compatible provides a configurable OpenAI-compatible LLM adapter.
package compatible

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
	"github.com/hexagon-codes/ai-core/transport"
)

const defaultBaseURL = "https://api.openai.com/v1"

// ModelSpec describes one OpenAI-compatible model exposed by a provider.
type ModelSpec struct {
	ID          string
	Name        string
	Description string
	MaxTokens   int
	InputCost   float64
	OutputCost  float64
	Features    []string
}

// Profile captures a known OpenAI-compatible provider endpoint and model set.
type Profile struct {
	Name     string
	BaseURL  string
	EnvKey   string
	Model    string
	Models   []ModelSpec
	Modality catalog.Modality
	Source   string
	Official bool
	Stable   bool
	Region   string
}

// Provider wraps the OpenAI adapter while reporting provider-specific name,
// models, and capability metadata.
type Provider struct {
	name         string
	models       []ModelSpec
	capabilities []catalog.Capability
	inner        *openai.Provider
}

// Option configures Provider.
type Option func(*config)

type config struct {
	name          string
	apiKey        string
	baseURL       string
	defaultModel  string
	models        []ModelSpec
	httpClient    *http.Client
	headers       map[string]string
	requestTO     time.Duration
	streamIdleTO  time.Duration
	networkPolicy *llm.NetworkPolicy
	source        string
	official      bool
	stable        bool
	region        string
}

// WithBaseURL sets the OpenAI-compatible base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) {
		c.baseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithModel sets the default model.
func WithModel(model string) Option {
	return func(c *config) {
		c.defaultModel = model
	}
}

// WithModels sets provider model metadata.
func WithModels(models []ModelSpec) Option {
	return func(c *config) {
		c.models = append([]ModelSpec(nil), models...)
	}
}

// WithHTTPClient injects a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) {
		c.httpClient = client
	}
}

// WithHeaders sets safe extra headers passed through the shared transport.
func WithHeaders(headers map[string]string) Option {
	return func(c *config) {
		c.headers = transport.CloneHeaders(headers)
	}
}

// WithRequestTimeout sets a request-level timeout.
func WithRequestTimeout(timeout time.Duration) Option {
	return func(c *config) {
		c.requestTO = timeout
	}
}

// WithStreamIdleTimeout sets stream idle read timeout.
func WithStreamIdleTimeout(timeout time.Duration) Option {
	return func(c *config) {
		c.streamIdleTO = timeout
	}
}

// WithNetworkPolicy enables SSRF/network policy enforcement.
func WithNetworkPolicy(policy llm.NetworkPolicy) Option {
	return func(c *config) {
		cloned := policy
		cloned.AllowedHosts = append([]string(nil), policy.AllowedHosts...)
		c.networkPolicy = &cloned
	}
}

// WithRegion annotates capability rows with provider deployment region.
func WithRegion(region string) Option {
	return func(c *config) {
		c.region = strings.TrimSpace(region)
	}
}

// New creates a named OpenAI-compatible Provider.
func New(name, apiKey string, opts ...Option) *Provider {
	cfg := &config{
		name:     strings.TrimSpace(name),
		apiKey:   apiKey,
		baseURL:  defaultBaseURL,
		official: true,
		stable:   true,
	}
	if cfg.name == "" {
		cfg.name = "openai-compatible"
	}
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
	if cfg.defaultModel == "" && len(cfg.models) > 0 {
		cfg.defaultModel = cfg.models[0].ID
	}
	openAIOpts := []openai.Option{
		openai.WithBaseURL(cfg.baseURL),
		openai.WithModel(cfg.defaultModel),
	}
	if cfg.httpClient != nil {
		openAIOpts = append(openAIOpts, openai.WithHTTPClient(cfg.httpClient))
	}
	if len(cfg.headers) > 0 {
		openAIOpts = append(openAIOpts, openai.WithHeaders(cfg.headers))
	}
	if cfg.requestTO > 0 {
		openAIOpts = append(openAIOpts, openai.WithRequestTimeout(cfg.requestTO))
	}
	if cfg.streamIdleTO > 0 {
		openAIOpts = append(openAIOpts, openai.WithStreamIdleTimeout(cfg.streamIdleTO))
	}
	if cfg.networkPolicy != nil {
		openAIOpts = append(openAIOpts, openai.WithNetworkPolicy(*cfg.networkPolicy))
	}

	return &Provider{
		name:         cfg.name,
		models:       append([]ModelSpec(nil), cfg.models...),
		capabilities: capabilitiesFromSpecs(cfg),
		inner:        openai.New(cfg.apiKey, openAIOpts...),
	}
}

// NewProfile creates a Provider from a known profile.
func NewProfile(profileName, apiKey string, opts ...Option) (*Provider, error) {
	profile, ok := Profiles()[profileName]
	if !ok {
		return nil, fmt.Errorf("unknown compatible profile %q", profileName)
	}
	if apiKey == "" && profile.EnvKey != "" {
		apiKey = os.Getenv(profile.EnvKey)
	}
	profileOpts := []Option{
		WithBaseURL(profile.BaseURL),
		WithModel(profile.Model),
		WithModels(profile.Models),
		func(c *config) {
			c.source = profile.Source
			c.official = profile.Official
			c.stable = profile.Stable
			c.region = profile.Region
		},
	}
	profileOpts = append(profileOpts, opts...)
	return New(profile.Name, apiKey, profileOpts...), nil
}

func (p *Provider) Name() string { return p.name }

func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return p.inner.Complete(ctx, req)
}

func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*llm.Stream, error) {
	return p.inner.Stream(ctx, req)
}

func (p *Provider) Models() []llm.ModelInfo {
	if len(p.models) == 0 {
		return p.inner.Models()
	}
	out := make([]llm.ModelInfo, 0, len(p.models))
	for _, m := range p.models {
		out = append(out, llm.ModelInfo{
			ID:          m.ID,
			Name:        firstNonEmpty(m.Name, m.ID),
			Description: m.Description,
			MaxTokens:   m.MaxTokens,
			InputCost:   m.InputCost,
			OutputCost:  m.OutputCost,
			Features:    append([]string(nil), m.Features...),
		})
	}
	return out
}

func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	return p.inner.CountTokens(messages)
}

// CountTokensContext 将可取消的 Token 计数透传给兼容 Provider。
func (p *Provider) CountTokensContext(ctx context.Context, messages []llm.Message) (int, error) {
	return p.inner.CountTokensContext(ctx, messages)
}

func (p *Provider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return p.inner.Embed(ctx, texts)
}

func (p *Provider) EmbedWithModel(ctx context.Context, model string, texts []string) ([][]float32, error) {
	return p.inner.EmbedWithModel(ctx, model, texts)
}

// Capabilities reports provider/model capability rows.
func (p *Provider) Capabilities() []catalog.Capability {
	return append([]catalog.Capability(nil), p.capabilities...)
}

func capabilitiesFromSpecs(c *config) []catalog.Capability {
	if len(c.models) == 0 {
		return nil
	}
	rows := make([]catalog.Capability, 0, len(c.models))
	for _, m := range c.models {
		if strings.TrimSpace(m.ID) == "" {
			continue
		}
		features := append([]string(nil), m.Features...)
		if c.region != "" {
			features = append(features, "region:"+c.region)
		}
		rows = append(rows, catalog.Capability{
			Provider:      c.name,
			Model:         m.ID,
			Name:          firstNonEmpty(m.Name, m.ID),
			Modality:      catalog.ModalityText,
			Features:      features,
			ContextWindow: m.MaxTokens,
			Stable:        c.stable,
			OfficialAPI:   c.official,
			Source:        c.source,
			Cost: catalog.Cost{
				InputPerMillionTokens:  m.InputCost,
				OutputPerMillionTokens: m.OutputCost,
				Currency:               "USD",
			},
		})
	}
	return rows
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

var _ llm.Provider = (*Provider)(nil)
var _ llm.ContextTokenCounter = (*Provider)(nil)
var _ llm.EmbeddingProvider = (*Provider)(nil)
var _ catalog.Provider = (*Provider)(nil)
