// Package deepseek provides DeepSeek LLM provider implementation.
package deepseek

import (
	"net/http"
	"os"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
)

const (
	defaultBaseURL = "https://api.deepseek.com/v1"
	defaultModel   = "deepseek-chat"
)

// Provider 实现 DeepSeek LLM 提供者
// DeepSeek 使用 OpenAI 兼容的 API，所以复用 OpenAI Provider
type Provider struct {
	*openai.Provider
}

// config 收集 Option 设定的配置，在 New 中据此构造内嵌的 openai.Provider。
//
// 由于 openai.Provider 的 model/baseURL/httpClient 字段都是私有的，
// 无法在 Provider 构造完成后再修改，因此先把 Option 的意图收集到 config，
// 再一次性透传给 openai.New。
type config struct {
	// model 默认模型；为空时使用 defaultModel
	model string
	// baseURL API 基础 URL；为空时使用 defaultBaseURL
	baseURL string
	// httpClient 自定义 HTTP 客户端；为 nil 时使用 openai 包的默认客户端
	httpClient *http.Client
}

// Option 是 Provider 的配置选项
type Option func(*config)

// WithModel 设置默认模型
//
// 传入的 model 会在 New 中透传给底层 openai.Provider 的 WithModel，
// 从而真正改变请求中省略 Model 字段时使用的默认模型。
func WithModel(model string) Option {
	return func(c *config) {
		c.model = model
	}
}

// WithBaseURL 设置 API 基础 URL
//
// 用于将请求指向自建网关或 OpenAI 兼容代理（如 one-api / new-api），
// 透传给底层 openai.Provider 的 WithBaseURL。
func WithBaseURL(url string) Option {
	return func(c *config) {
		c.baseURL = url
	}
}

// WithHTTPClient 设置自定义 HTTP 客户端
//
// 用于注入自定义传输层（如带超时、代理或 httptest 测试服务器的客户端），
// 透传给底层 openai.Provider 的 WithHTTPClient。
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) {
		c.httpClient = client
	}
}

// New 创建 DeepSeek Provider
// apiKey 可以为空，会从环境变量 DEEPSEEK_API_KEY 读取
func New(apiKey string, opts ...Option) *Provider {
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}

	// 先收集 Option 意图，再据此构造内嵌的 openai.Provider。
	cfg := &config{
		model:   defaultModel,
		baseURL: defaultBaseURL,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	// 将 deepseek 的配置透传给底层 openai.New。
	openaiOpts := []openai.Option{
		openai.WithBaseURL(cfg.baseURL),
		openai.WithModel(cfg.model),
	}
	if cfg.httpClient != nil {
		openaiOpts = append(openaiOpts, openai.WithHTTPClient(cfg.httpClient))
	}

	return &Provider{
		Provider: openai.New(apiKey, openaiOpts...),
	}
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "deepseek"
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	return []llm.ModelInfo{
		{
			ID:          "deepseek-chat",
			Name:        "DeepSeek Chat",
			Description: "General purpose chat model, good balance of capability and cost",
			MaxTokens:   64000,
			InputCost:   0.14, // per million tokens
			OutputCost:  0.28, // per million tokens
			Features:    []string{llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureStreaming},
		},
		{
			ID:          "deepseek-reasoner",
			Name:        "DeepSeek Reasoner",
			Description: "Advanced reasoning model for complex tasks",
			MaxTokens:   64000,
			InputCost:   0.55,
			OutputCost:  2.19,
			Features:    []string{llm.FeatureStreaming},
		},
	}
}

// 确保实现了 Provider 和 EmbeddingProvider 接口
// Embed/EmbedWithModel 方法继承自 openai.Provider
var _ llm.Provider = (*Provider)(nil)
var _ llm.EmbeddingProvider = (*Provider)(nil)
