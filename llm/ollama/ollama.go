// Package ollama provides Ollama local LLM provider implementation.
package ollama

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptrace"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/ai-core/tokenizer"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const (
	defaultBaseURL = "http://localhost:11434"
	defaultModel   = "llama3.2"

	defaultOllamaNumCtx   = 32768
	maxAutomaticNumCtx    = 32768
	defaultModelMaxTokens = 32768
	// Complete and Stream share a transport safety ceiling long enough for a
	// cold local prefill. Product/model contexts remain the authoritative,
	// shorter SLO (for example HexClaw qwen3.5:9b at 360 seconds).
	ollamaDefaultResponseHeaderWait = 10 * time.Minute
	ollamaStreamResponseHeaderWait  = 10 * time.Minute

	// defaultKeepAlive 每次请求随发的模型驻留时长（BUG-20260710）。
	// Ollama 服务端默认仅 5 分钟：空闲即卸载模型、KV 前缀缓存全丢——纯 CPU 机器上
	// 大 system prompt（真机取证 ~7.9k token @ 23 tok/s ≈ 344s）每次都要冷 prefill。
	// 请求级 keep_alive 每次刷新驻留窗口，让模型与 KV 缓存跨对话间隙存活。
	defaultKeepAlive = "30m"
)

// Provider 实现 Ollama LLM 提供者
type Provider struct {
	baseURL   string
	model     string
	keepAlive string // 请求级模型驻留时长（空串=不下发，回落 Ollama 服务端默认）
	// numCtx 粘性水位按模型记录成功请求，自动分档只升不降；有界 LRU 防止
	// 任意模型名耗尽内存。显式 metadata 与失败请求均不进入水位。
	stickyNumCtxMu    sync.Mutex
	stickyNumCtx      map[string]int
	stickyNumCtxOrder []string
	pendingNumCtx     map[string]map[int]int
	numCtxWriteOnce   [64]sync.Once
	numCtxWriteGate   [64]chan struct{}
	httpClient        *http.Client
	streamHTTPClient  *http.Client
	transport         *transport.Transport
	streamTransport   *transport.Transport
	policy            *llm.NetworkPolicy
	// modelsMu 保护 models:Models() 惰性写入与 numCtxForRequest 的读取可能并发
	// (并发 Complete + Models 曾构成 data race,BUG-20260710 F5)。
	modelsMu sync.RWMutex
	models   []llm.ModelInfo // 缓存的模型列表
}

// Option 是 Provider 的配置选项
type Option func(*Provider)

// WithBaseURL 设置 API 基础 URL
func WithBaseURL(url string) Option {
	return func(p *Provider) {
		p.baseURL = url
	}
}

// WithModel 设置默认模型
func WithModel(model string) Option {
	return func(p *Provider) {
		p.model = model
	}
}

// WithKeepAlive 设置请求级模型驻留时长（如 "30m"/"2h"；BUG-20260710，见 defaultKeepAlive）。
func WithKeepAlive(d string) Option {
	return func(p *Provider) {
		p.keepAlive = d
	}
}

// WithHTTPClient 设置 HTTP 客户端
func WithHTTPClient(client *http.Client) Option {
	return func(p *Provider) {
		p.httpClient = client
		p.streamHTTPClient = client
	}
}

// WithNetworkPolicy 设置上游网络出口约束。
//
// Ollama 默认允许本地 HTTP/私网访问；传入该选项可收紧到调用方指定策略。
func WithNetworkPolicy(policy llm.NetworkPolicy) Option {
	return func(p *Provider) {
		p.policy = transport.CloneNetworkPolicy(&policy)
	}
}

const trustedReasoningDisclosureEvidenceMetadataKey = "ollama_trusted_reasoning_disclosure_evidence_v1"

type trustedReasoningDisclosureEvidence struct {
	evidence streamx.ReasoningDisclosureEvidence
}

// InjectTrustedReasoningDisclosureEvidence 将宿主冻结的精确路由交给原生流解析器。
// 网络响应和普通请求元数据无法构造此私有包装，因此缺少宿主注入时保持失败关闭。
func InjectTrustedReasoningDisclosureEvidence(req *llm.CompletionRequest, provider, model string) {
	if req == nil {
		return
	}
	provider = strings.TrimSpace(provider)
	model = strings.TrimSpace(model)
	if provider == "" || model == "" {
		return
	}
	if req.Metadata == nil {
		req.Metadata = make(map[string]any, 1)
	}
	req.Metadata[trustedReasoningDisclosureEvidenceMetadataKey] = trustedReasoningDisclosureEvidence{
		evidence: streamx.ReasoningDisclosureEvidence{
			ExplicitlyPublic: true,
			Provider:         provider,
			Model:            model,
		},
	}
}

func trustedReasoningDisclosureEvidenceFromMetadata(metadata map[string]any) streamx.ReasoningDisclosureEvidence {
	raw, ok := metadata[trustedReasoningDisclosureEvidenceMetadataKey]
	if !ok {
		return streamx.ReasoningDisclosureEvidence{}
	}
	evidence, ok := raw.(trustedReasoningDisclosureEvidence)
	if !ok {
		return streamx.ReasoningDisclosureEvidence{}
	}
	return evidence.evidence
}

// New 创建 Ollama Provider
func New(opts ...Option) *Provider {
	defaultPolicy := llm.NetworkPolicy{AllowHTTP: true, AllowPrivate: true}
	p := &Provider{
		baseURL:      defaultBaseURL,
		model:        defaultModel,
		keepAlive:    defaultKeepAlive,
		stickyNumCtx: map[string]int{},
		// 不设全局 Timeout — 流式请求的超时由调用方 context 控制
		// http.Client.Timeout 对流式响应会在整个读取期间生效，
		// 本地模型推理可能需要数分钟
		httpClient:       httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(ollamaDefaultResponseHeaderWait)),
		streamHTTPClient: httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(ollamaStreamResponseHeaderWait)),
		policy:           &defaultPolicy,
	}

	// 从环境变量读取
	if url := os.Getenv("OLLAMA_HOST"); url != "" {
		p.baseURL = url
	}
	if model := os.Getenv("OLLAMA_MODEL"); model != "" {
		p.model = model
	}

	for _, opt := range opts {
		opt(p)
	}
	p.transport = transport.NewTransport(p.httpClient, p.policy)
	p.streamTransport = transport.NewTransport(p.streamHTTPClient, p.policy)

	return p
}

// Name 返回提供者名称
func (p *Provider) Name() string {
	return "ollama"
}

// Complete 执行非流式补全请求
func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if req.Model == "" {
		req.Model = p.model
	}
	reasoningPlan, explicitReasoning, err := ollamaReasoningPlan(req, p.baseURL)
	if err != nil {
		if explicitReasoning {
			llm.PublishReasoningReceipt(req.Metadata, reasoningPlan.Receipt)
		}
		return nil, err
	}
	ctx, releaseWrite, err := p.lockNumCtxWrite(ctx, req.Model)
	if err != nil {
		return nil, err
	}
	defer releaseWrite()

	body, numCtx, automaticNumCtx, err := p.buildRequestBodyForSend(req, false)
	if err != nil {
		return nil, err
	}
	var reservation *numCtxReservation
	if automaticNumCtx {
		reservation = p.reserveNumCtx(req.Model, numCtx)
	}

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/chat", body, false)
	releaseWrite()
	if err != nil {
		reservation.finish(false)
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	var result ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		reservation.finish(false)
		return nil, err
	}
	if result.Error != "" {
		reservation.finish(false)
		return nil, fmt.Errorf("ollama response error: %s", result.Error)
	}
	reservation.finish(true)

	parsed := p.parseResponse(&result, req.Model)
	if explicitReasoning {
		receipt := reasoningPlan.Receipt.WithEvidence(true, result.Message.Thinking != "", 0)
		parsed.ReasoningReceipt = &receipt
		llm.PublishReasoningReceipt(req.Metadata, receipt)
	}
	return parsed, nil
}

// Stream 执行流式补全请求
func (p *Provider) Stream(ctx context.Context, req llm.CompletionRequest) (*streamx.Stream, error) {
	if req.Model == "" {
		req.Model = p.model
	}
	reasoningPlan, explicitReasoning, err := ollamaReasoningPlan(req, p.baseURL)
	if err != nil {
		if explicitReasoning {
			llm.PublishReasoningReceipt(req.Metadata, reasoningPlan.Receipt)
		}
		return nil, err
	}
	ctx, releaseWrite, err := p.lockNumCtxWrite(ctx, req.Model)
	if err != nil {
		return nil, err
	}
	defer releaseWrite()

	body, numCtx, automaticNumCtx, err := p.buildRequestBodyForSend(req, true)
	if err != nil {
		return nil, err
	}
	var reservation *numCtxReservation
	if automaticNumCtx {
		reservation = p.reserveNumCtx(req.Model, numCtx)
	}

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/chat", body, true)
	releaseWrite()
	if err != nil {
		reservation.finish(false)
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	parser := &ollamaStreamParser{
		metadata:          req.Metadata,
		reasoningEvidence: trustedReasoningDisclosureEvidenceFromMetadata(req.Metadata),
	}
	if explicitReasoning {
		receipt := reasoningPlan.Receipt.WithEvidence(true, false, 0)
		parser.receipt = receipt
		llm.PublishReasoningReceipt(req.Metadata, receipt)
	}
	if reservation != nil {
		reservation.bindContext(ctx)
		parser.onSuccess = func() { reservation.finish(true) }
		parser.onFailure = func() { reservation.finish(false) }
		resp.Body = &numCtxReservationReadCloser{ReadCloser: resp.Body, reservation: reservation}
	}

	return streamx.NewStreamWithContext(ctx, resp.Body, streamx.CustomFormat).SetParser(parser), nil
}

// Models 返回可用模型列表
func (p *Provider) Models() []llm.ModelInfo {
	// 如果已缓存，直接返回
	p.modelsMu.RLock()
	cached := cloneModelInfos(p.models)
	p.modelsMu.RUnlock()
	if len(cached) > 0 {
		return cached
	}

	// 尝试从 Ollama 获取本地模型列表（网络请求在锁外，避免持锁阻塞）
	models, err := p.fetchLocalModels()
	if err == nil && len(models) > 0 {
		p.modelsMu.Lock()
		p.models = models
		p.modelsMu.Unlock()
		return cloneModelInfos(models)
	}

	// 返回常见模型的默认列表
	defaults := []llm.ModelInfo{
		{
			ID:          "llama3.2",
			Name:        "Llama 3.2",
			Description: "Meta's Llama 3.2 model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "llama3.2:1b",
			Name:        "Llama 3.2 1B",
			Description: "Meta's Llama 3.2 1B parameter model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "llama3.1",
			Name:        "Llama 3.1",
			Description: "Meta's Llama 3.1 model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "qwen2.5",
			Name:        "Qwen 2.5",
			Description: "Alibaba's Qwen 2.5 model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "qwen2.5:7b",
			Name:        "Qwen 2.5 7B",
			Description: "Alibaba's Qwen 2.5 7B model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "mistral",
			Name:        "Mistral",
			Description: "Mistral AI's model",
			MaxTokens:   32768,
			Features:    []string{llm.FeatureFunctions, llm.FeatureStreaming},
		},
		{
			ID:          "codellama",
			Name:        "Code Llama",
			Description: "Meta's Code Llama model for coding tasks",
			MaxTokens:   16384,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "deepseek-coder-v2",
			Name:        "DeepSeek Coder V2",
			Description: "DeepSeek's coding model",
			MaxTokens:   128000,
			Features:    []string{llm.FeatureStreaming},
		},
		{
			ID:          "llava",
			Name:        "LLaVA",
			Description: "Vision-language model",
			MaxTokens:   4096,
			Features:    []string{llm.FeatureVision, llm.FeatureStreaming},
		},
	}
	for i := range defaults {
		defaults[i].ReasoningSupport = llm.ReasoningUnknown
	}
	return defaults
}

func cloneModelInfos(models []llm.ModelInfo) []llm.ModelInfo {
	cloned := make([]llm.ModelInfo, len(models))
	for i := range models {
		cloned[i] = models[i]
		cloned[i].Features = append([]string(nil), models[i].Features...)
	}
	return cloned
}

// fetchLocalModels 从 Ollama 获取本地模型列表
func (p *Provider) fetchLocalModels() ([]llm.ModelInfo, error) {
	resp, err := p.doRequest(context.Background(), http.MethodGet, p.baseURL+"/api/tags", nil, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			Model        string   `json:"model"`
			Name         string   `json:"name"`
			Size         int64    `json:"size"`
			ModifiedAt   string   `json:"modified_at"`
			Capabilities []string `json:"capabilities"`
			Details      struct {
				Format            string `json:"format"`
				Family            string `json:"family"`
				ParameterSize     string `json:"parameter_size"`
				QuantizationLevel string `json:"quantization_level"`
				ContextLength     int    `json:"context_length"`
			} `json:"details"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	models := make([]llm.ModelInfo, 0, len(result.Models))
	for i := range result.Models {
		m := &result.Models[i]
		modelID := strings.TrimSpace(m.Name)
		if modelID == "" {
			modelID = strings.TrimSpace(m.Model)
		}
		if modelID == "" {
			continue
		}
		maxTokens := m.Details.ContextLength
		capabilities := m.Capabilities
		if len(capabilities) == 0 || maxTokens <= 0 {
			if showDetails, err := p.fetchModelDetails(context.Background(), modelID); err == nil {
				if len(capabilities) == 0 {
					capabilities = showDetails.capabilities
				}
				if maxTokens <= 0 {
					maxTokens = showDetails.contextLength
				}
			}
		}
		if maxTokens <= 0 {
			maxTokens = defaultModelMaxTokens
		}
		models = append(models, llm.ModelInfo{
			ID:               modelID,
			Name:             modelID,
			Description:      fmt.Sprintf("%s model (%s)", m.Details.Family, m.Details.ParameterSize),
			MaxTokens:        maxTokens,
			Features:         ollamaCapabilitiesToFeatures(capabilities),
			ReasoningSupport: ollamaReasoningSupport(capabilities),
		})
	}

	return models, nil
}

type ollamaModelDetails struct {
	capabilities  []string
	contextLength int
}

func (p *Provider) fetchModelDetails(ctx context.Context, model string) (ollamaModelDetails, error) {
	body, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return ollamaModelDetails{}, err
	}
	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/show", body, false)
	if err != nil {
		return ollamaModelDetails{}, err
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			return
		}
	}()

	var result struct {
		Capabilities []string       `json:"capabilities"`
		ModelInfo    map[string]any `json:"model_info"`
	}
	decoder := json.NewDecoder(resp.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&result); err != nil {
		return ollamaModelDetails{}, err
	}
	return ollamaModelDetails{
		capabilities:  result.Capabilities,
		contextLength: ollamaModelContextLength(result.ModelInfo),
	}, nil
}

func ollamaModelContextLength(modelInfo map[string]any) int {
	if architecture, ok := modelInfo["general.architecture"].(string); ok {
		if contextLength, ok := positiveInt(modelInfo[strings.TrimSpace(architecture)+".context_length"]); ok {
			return contextLength
		}
	}
	if contextLength, ok := positiveInt(modelInfo["context_length"]); ok {
		return contextLength
	}

	contextLength := 0
	for key, value := range modelInfo {
		if !strings.HasSuffix(key, ".context_length") {
			continue
		}
		if candidate, ok := positiveInt(value); ok && (contextLength == 0 || candidate < contextLength) {
			contextLength = candidate
		}
	}
	return contextLength
}

func ollamaCapabilitiesToFeatures(capabilities []string) []string {
	features := []string{llm.FeatureStreaming}
	seen := map[string]bool{llm.FeatureStreaming: true}
	add := func(feature string) {
		if feature == "" || seen[feature] {
			return
		}
		features = append(features, feature)
		seen[feature] = true
	}

	for _, capability := range capabilities {
		switch strings.ToLower(strings.TrimSpace(capability)) {
		case "vision":
			add(llm.FeatureVision)
		case "tools", "tool", "function", "functions":
			add(llm.FeatureFunctions)
		case "embedding", "embeddings", "embed":
			add(llm.FeatureEmbedding)
		}
	}
	return features
}

// CountTokens 计算消息的 Token 数量（简化实现）
func (p *Provider) CountTokens(messages []llm.Message) (int, error) {
	return p.countTokens(context.Background(), messages)
}

// CountTokensContext 计算消息的 Token 数量，并在遍历消息时响应 context 取消。
func (p *Provider) CountTokensContext(ctx context.Context, messages []llm.Message) (int, error) {
	return p.countTokens(ctx, messages)
}

func (p *Provider) countTokens(ctx context.Context, messages []llm.Message) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	// 简化估算：约 4 个字符一个 token
	var total int
	for _, msg := range messages {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		total += len(msg.Content) / 4
	}
	return total, nil
}

// buildRequestBody 构建请求体
func (p *Provider) buildRequestBody(req llm.CompletionRequest, stream bool) ([]byte, error) {
	body, _, _, err := p.buildRequestBodyForSend(req, stream)
	return body, err
}

func (p *Provider) buildRequestBodyForSend(req llm.CompletionRequest, stream bool) (body []byte, numCtx int, automaticNumCtx bool, err error) {
	messages, err := convertMessagesForOllama(req.Messages)
	if err != nil {
		return nil, 0, false, err
	}

	payload := map[string]any{
		"model":    req.Model,
		"messages": messages,
		"stream":   stream,
	}
	if plan, explicit, planErr := ollamaReasoningPlan(req, p.baseURL); planErr != nil {
		return nil, 0, false, planErr
	} else if explicit {
		for field, value := range plan.Wire {
			payload[field] = value
		}
	}
	// 模型驻留（BUG-20260710）：metadata 按请求覆盖 > 选项/默认；空串=不下发
	keepAlive := p.keepAlive
	if v, ok := req.Metadata["keep_alive"].(string); ok && strings.TrimSpace(v) != "" {
		keepAlive = strings.TrimSpace(v)
	}
	if keepAlive != "" {
		payload["keep_alive"] = keepAlive
	}

	// Ollama 使用 options 嵌套参数
	options := make(map[string]any)
	if req.Temperature != nil {
		options["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		options["top_p"] = *req.TopP
	}
	options["num_predict"] = outputBudget(req)
	if len(req.Stop) > 0 {
		options["stop"] = req.Stop
	}
	numCtx, explicitNumCtx := ollamaNumCtxFromMetadata(req.Metadata)
	if !explicitNumCtx {
		numCtx = p.numCtxForRequest(req)
		needed, exceedsAutomaticMaximum := automaticNumCtxNeeded(req)
		if exceedsAutomaticMaximum {
			return nil, 0, false, fmt.Errorf("ollama request exceeds automatic num_ctx maximum %d", maxAutomaticNumCtx)
		}
		if needed > numCtx {
			return nil, 0, false, fmt.Errorf("ollama request needs an estimated %d context tokens, automatic num_ctx is %d", needed, numCtx)
		}
	}
	if numCtx > 0 {
		options["num_ctx"] = numCtx
	}

	if len(options) > 0 {
		payload["options"] = options
	}

	// 工具支持（部分模型支持）
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}

	// ResponseFormat 支持
	// Ollama 使用 format 参数指定输出格式
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object", "json_schema":
			payload["format"] = "json"
		}
	}

	body, err = json.Marshal(payload)
	return body, numCtx, !explicitNumCtx, err
}

func (p *Provider) numCtxForRequest(req llm.CompletionRequest) int {
	// 显式 metadata 最高优先且原样生效(可低于自动档),不影响粘性水位——显式即契约。
	if numCtx, ok := ollamaNumCtxFromMetadata(req.Metadata); ok {
		return numCtx
	}

	// BUG-20260710 P0:恒 32768 让 KV/buffer 多预分配 ~1GB 纯脏页且小请求首推理 3×慢
	// (真机实测 230s vs 74s)。自动档按「prompt 估算 + 输出预算 + 余量」取最小可容纳档。
	needed, _ := automaticNumCtxNeeded(req)
	tier := automaticNumCtxTier(needed)

	// 模型自身上下文上限仍是硬顶(8k 模型不发 16k)。
	capLimit := 0
	p.modelsMu.RLock()
	for _, model := range p.models {
		if sameOllamaModel(model.ID, req.Model) || sameOllamaModel(model.Name, req.Model) {
			capLimit = clampAutomaticNumCtx(model.MaxTokens)
			break
		}
	}
	p.modelsMu.RUnlock()
	if capLimit > 0 && capLimit < tier {
		tier = capLimit
	}

	// 同模型已成功请求的粘性水位只升不降；这里只读候选值，只有上游成功后
	// commitNumCtx 才更新 LRU。硬顶在合并后再钳一次，避免历史水位绕过模型 cap。
	p.stickyNumCtxMu.Lock()
	defer p.stickyNumCtxMu.Unlock()
	key := stickyNumCtxKey(req.Model)
	if cur := p.stickyNumCtx[key]; cur > tier {
		tier = cur
	}
	if pending := p.pendingNumCtxMaxLocked(key); pending > tier {
		tier = pending
	}
	if capLimit > 0 && tier > capLimit {
		tier = capLimit
	}
	return tier
}

func (p *Provider) commitNumCtx(model string, numCtx int) {
	capLimit := p.numCtxCap(model)
	if capLimit > 0 && numCtx > capLimit {
		numCtx = capLimit
	}

	p.stickyNumCtxMu.Lock()
	defer p.stickyNumCtxMu.Unlock()
	key := stickyNumCtxKey(model)
	if cur := p.stickyNumCtx[key]; cur > numCtx {
		numCtx = cur
	}
	if capLimit > 0 && numCtx > capLimit {
		numCtx = capLimit
	}
	p.storeStickyNumCtxLocked(key, numCtx)
}

func (p *Provider) numCtxCap(model string) int {
	p.modelsMu.RLock()
	defer p.modelsMu.RUnlock()
	for _, modelInfo := range p.models {
		if sameOllamaModel(modelInfo.ID, model) || sameOllamaModel(modelInfo.Name, model) {
			return clampAutomaticNumCtx(modelInfo.MaxTokens)
		}
	}
	return 0
}

type numCtxReservation struct {
	provider *Provider
	model    string
	key      string
	numCtx   int
	once     sync.Once
	stopMu   sync.Mutex
	finished bool
	stopCtx  func() bool
}

func (p *Provider) reserveNumCtx(model string, numCtx int) *numCtxReservation {
	key := stickyNumCtxKey(model)
	p.stickyNumCtxMu.Lock()
	if p.pendingNumCtx == nil {
		p.pendingNumCtx = make(map[string]map[int]int)
	}
	counts := p.pendingNumCtx[key]
	if counts == nil {
		counts = make(map[int]int)
		p.pendingNumCtx[key] = counts
	}
	counts[numCtx]++
	p.stickyNumCtxMu.Unlock()
	return &numCtxReservation{provider: p, model: model, key: key, numCtx: numCtx}
}

func (r *numCtxReservation) finish(success bool) {
	if r == nil {
		return
	}
	r.once.Do(func() {
		r.stopMu.Lock()
		r.finished = true
		stopCtx := r.stopCtx
		r.stopCtx = nil
		r.stopMu.Unlock()
		if stopCtx != nil {
			stopCtx()
		}
		r.provider.finishNumCtxReservation(r, success)
	})
}

func (r *numCtxReservation) bindContext(ctx context.Context) {
	stopCtx := context.AfterFunc(ctx, func() { r.finish(false) })
	r.stopMu.Lock()
	if r.finished {
		r.stopMu.Unlock()
		stopCtx()
		return
	}
	r.stopCtx = stopCtx
	r.stopMu.Unlock()
}

func (p *Provider) finishNumCtxReservation(reservation *numCtxReservation, success bool) {
	capLimit := p.numCtxCap(reservation.model)
	p.stickyNumCtxMu.Lock()
	defer p.stickyNumCtxMu.Unlock()

	if success {
		numCtx := reservation.numCtx
		if cur := p.stickyNumCtx[reservation.key]; cur > numCtx {
			numCtx = cur
		}
		if capLimit > 0 && numCtx > capLimit {
			numCtx = capLimit
		}
		p.storeStickyNumCtxLocked(reservation.key, numCtx)
	}

	counts := p.pendingNumCtx[reservation.key]
	if counts != nil {
		counts[reservation.numCtx]--
		if counts[reservation.numCtx] <= 0 {
			delete(counts, reservation.numCtx)
		}
		if len(counts) == 0 {
			delete(p.pendingNumCtx, reservation.key)
		}
	}
}

func (p *Provider) pendingNumCtxMaxLocked(key string) int {
	maxNumCtx := 0
	for numCtx, count := range p.pendingNumCtx[key] {
		if count > 0 && numCtx > maxNumCtx {
			maxNumCtx = numCtx
		}
	}
	return maxNumCtx
}

func (p *Provider) lockNumCtxWrite(ctx context.Context, model string) (context.Context, func(), error) {
	if ctx == nil {
		ctx = context.Background()
	}
	digest := sha256.Sum256([]byte(canonicalOllamaModel(model)))
	index := int(digest[0]) % len(p.numCtxWriteGate)
	p.numCtxWriteOnce[index].Do(func() {
		p.numCtxWriteGate[index] = make(chan struct{}, 1)
		p.numCtxWriteGate[index] <- struct{}{}
	})
	gate := p.numCtxWriteGate[index]
	select {
	case <-ctx.Done():
		return ctx, func() {}, ctx.Err()
	case <-gate:
	}
	var once sync.Once
	release := func() {
		once.Do(func() { gate <- struct{}{} })
	}
	trace := &httptrace.ClientTrace{
		WroteRequest: func(httptrace.WroteRequestInfo) { release() },
	}
	return httptrace.WithClientTrace(ctx, trace), release, nil
}

type numCtxReservationReadCloser struct {
	io.ReadCloser
	reservation *numCtxReservation
}

func (r *numCtxReservationReadCloser) Read(p []byte) (int, error) {
	n, err := r.ReadCloser.Read(p)
	if err != nil && (n == 0 || !bytes.ContainsRune(p[:n], '\n') || len(bytes.TrimSpace(p[:n])) == 0) {
		r.reservation.finish(false)
	}
	return n, err
}

func (r *numCtxReservationReadCloser) Close() error {
	r.reservation.finish(false)
	return r.ReadCloser.Close()
}

const maxStickyNumCtxModels = 64
const maxStickyNumCtxKeyBytes = 256

func stickyNumCtxKey(model string) string {
	model = canonicalOllamaModel(model)
	if len(model) <= maxStickyNumCtxKeyBytes {
		return "raw:" + model
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(model)))
}

func canonicalOllamaModel(model string) string {
	model = strings.TrimSpace(model)
	if model == "" {
		return model
	}
	digest := ""
	if index := strings.Index(model, "@"); index >= 0 {
		digest = model[index:]
		model = model[:index]
	}
	model = strings.TrimPrefix(model, "registry.ollama.ai/")
	model = strings.TrimPrefix(model, "library/")
	if digest != "" {
		return model + digest
	}
	lastSlash := strings.LastIndex(model, "/")
	if strings.LastIndex(model, ":") <= lastSlash {
		return model + ":latest"
	}
	return model
}

func sameOllamaModel(left, right string) bool {
	return canonicalOllamaModel(left) == canonicalOllamaModel(right)
}

func (p *Provider) storeStickyNumCtxLocked(model string, numCtx int) {
	if p.stickyNumCtx == nil {
		p.stickyNumCtx = make(map[string]int)
	}
	_, exists := p.stickyNumCtx[model]
	for i, cachedModel := range p.stickyNumCtxOrder {
		if cachedModel != model {
			continue
		}
		copy(p.stickyNumCtxOrder[i:], p.stickyNumCtxOrder[i+1:])
		p.stickyNumCtxOrder[len(p.stickyNumCtxOrder)-1] = ""
		p.stickyNumCtxOrder = p.stickyNumCtxOrder[:len(p.stickyNumCtxOrder)-1]
		break
	}
	if !exists && len(p.stickyNumCtx) >= maxStickyNumCtxModels {
		delete(p.stickyNumCtx, p.stickyNumCtxOrder[0])
		copy(p.stickyNumCtxOrder, p.stickyNumCtxOrder[1:])
		p.stickyNumCtxOrder[len(p.stickyNumCtxOrder)-1] = ""
		p.stickyNumCtxOrder = p.stickyNumCtxOrder[:len(p.stickyNumCtxOrder)-1]
	}
	p.stickyNumCtx[model] = numCtx
	p.stickyNumCtxOrder = append(p.stickyNumCtxOrder, model)
}

// numCtxTiers 自动分档表:少档 + 只升不降,平衡「内存/首推理成本」与「重载抖动」。
var numCtxTiers = []int{4096, 8192, 16384, 32768}

func automaticNumCtxTier(needed int) int {
	for _, tier := range numCtxTiers {
		if needed <= tier {
			return tier
		}
	}
	return numCtxTiers[len(numCtxTiers)-1]
}

// numCtxHeadroom 覆盖模板与特殊 token 等固定估算开销。
const numCtxHeadroom = 512

// numCtxImageBudget 按已观测视觉编码上界留出整档余量；base64 长度与
// vision token 数无关，不能按文本口径估算。
const numCtxImageBudget = 6144

// numCtxEstimateMarginDenominator 给快速 tokenizer 的 ±10% 误差留 12.5% 余量。
const numCtxEstimateMarginDenominator = 8

const numCtxToolsPromptOverhead = 64
const numCtxToolDefinitionOverhead = 16
const numCtxToolCallOverhead = 8

// estimateRequestTokens 估算请求侧 token:复用 tokenizer 的中英混合口径(L0 复用,不手搓)。
// tool Parameters(完整 JSON schema 随 payload["tools"] 发给 Ollama)与图片必须计入,
// 否则档位边界处 num_ctx 偏小,Ollama 会静默截断 prompt(BUG-20260710 F2)。
func estimateRequestTokens(req llm.CompletionRequest) int {
	const estimateLimit = maxAutomaticNumCtx + 1
	var b strings.Builder
	images := 0
	framingTokens := 0
	for _, m := range req.Messages {
		if len(m.MultiContent) > 0 {
			for _, part := range m.MultiContent {
				switch part.Type {
				case "image_url":
					if part.ImageURL != nil {
						images++
					}
				default:
					b.WriteString(part.Text)
				}
			}
		} else {
			b.WriteString(m.Content)
		}
		if len(m.ToolCalls) > 0 {
			if raw, err := json.Marshal(m.ToolCalls); err == nil {
				b.Write(raw)
			}
			framingTokens = saturatingAdd(framingTokens, len(m.ToolCalls)*numCtxToolCallOverhead, estimateLimit)
		}
	}
	if len(req.Tools) > 0 {
		if raw, err := json.Marshal(req.Tools); err == nil {
			b.Write(raw)
		}
		framingTokens = saturatingAdd(framingTokens, numCtxToolsPromptOverhead, estimateLimit)
		framingTokens = saturatingAdd(framingTokens, len(req.Tools)*numCtxToolDefinitionOverhead, estimateLimit)
	}
	// 每条消息的模板包装开销粗记 8 token，并给快速 tokenizer 的误差留余量。
	textTokens := saturatingAdd(tokenizer.CountGPT4(b.String()), len(req.Messages)*8, estimateLimit)
	textTokens = saturatingAdd(textTokens, framingTokens, estimateLimit)
	margin := (textTokens + numCtxEstimateMarginDenominator - 1) / numCtxEstimateMarginDenominator
	estimated := saturatingAdd(textTokens, margin, estimateLimit)
	imageTokens := estimateLimit
	if images <= estimateLimit/numCtxImageBudget {
		imageTokens = images * numCtxImageBudget
	}
	return saturatingAdd(estimated, imageTokens, estimateLimit)
}

func automaticNumCtxNeeded(req llm.CompletionRequest) (int, bool) {
	const estimateLimit = maxAutomaticNumCtx + 1
	needed := estimateRequestTokens(req)
	needed = saturatingAdd(needed, outputBudget(req), estimateLimit)
	needed = saturatingAdd(needed, numCtxHeadroom, estimateLimit)
	if needed > maxAutomaticNumCtx {
		return maxAutomaticNumCtx, true
	}
	return needed, false
}

func saturatingAdd(total, delta, limit int) int {
	if total >= limit || delta >= limit-total {
		return limit
	}
	return total + delta
}

// outputBudget 输出预算:显式 MaxTokens 优先,未设时固定为 2048；请求构造会把
// 同一数值下发为 num_predict，保证估算预算与 Ollama 实际生成上限一致。
func outputBudget(req llm.CompletionRequest) int {
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}
	return 2048
}

func clampAutomaticNumCtx(contextLength int) int {
	if contextLength <= 0 {
		return defaultOllamaNumCtx
	}
	if contextLength > maxAutomaticNumCtx {
		return maxAutomaticNumCtx
	}
	return contextLength
}

func ollamaNumCtxFromMetadata(metadata map[string]any) (int, bool) {
	if len(metadata) == 0 {
		return 0, false
	}
	for _, key := range []string{"num_ctx", "ollama_num_ctx", "context_length", "max_context_tokens"} {
		if value, exists := metadata[key]; exists {
			if numCtx, ok := positiveInt(value); ok {
				return numCtx, true
			}
		}
	}
	return 0, false
}

func positiveInt(value any) (int, bool) {
	switch v := value.(type) {
	case int:
		return v, v > 0
	case int64:
		n := int(v)
		return n, n > 0 && int64(n) == v
	case int32:
		return int(v), v > 0
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 || v != math.Trunc(v) {
			return 0, false
		}
		n := int(v)
		return n, n > 0 && float64(n) == v
	case float32:
		if v <= 0 || v != float32(math.Trunc(float64(v))) {
			return 0, false
		}
		n := int(v)
		return n, n > 0 && float32(n) == v
	case json.Number:
		n, err := strconv.Atoi(v.String())
		return n, err == nil && n > 0
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		return n, err == nil && n > 0
	default:
		return 0, false
	}
}

func convertMessagesForOllama(messages []llm.Message) ([]map[string]any, error) {
	converted := openai.ConvertMessages(messages)
	for _, msg := range converted {
		// tool_calls[].function.arguments 归一化：OpenAI 协议把参数序列化为 JSON
		// 字符串，但 Ollama 原生 /api/chat 要求它是 JSON 对象，收到字符串会返回
		// 400 "Value looks like object, but can't find closing '}' symbol"。
		// 必须在 content 处理前跑——assistant+tool_calls 消息的 content 为 nil，
		// 会走下面的 continue 分支跳过。
		normalizeOllamaToolCallArguments(msg)

		contentParts, ok := contentPartsFromOpenAIMessage(msg["content"])
		if !ok {
			continue
		}

		texts := make([]string, 0, len(contentParts))
		images := make([]string, 0)
		for _, part := range contentParts {
			partType, ok := part["type"].(string)
			if !ok {
				continue
			}
			switch partType {
			case "image_url":
				imageURL, ok := part["image_url"].(map[string]any)
				if !ok {
					return nil, fmt.Errorf("ollama image_url part missing image_url")
				}
				rawURL, ok := imageURL["url"].(string)
				if !ok {
					return nil, fmt.Errorf("ollama image_url part missing url")
				}
				payload, err := ollamaImagePayload(rawURL)
				if err != nil {
					return nil, err
				}
				images = append(images, payload)
			default:
				text, ok := part["text"].(string)
				if ok && text != "" {
					texts = append(texts, text)
				}
			}
		}

		msg["content"] = strings.Join(texts, "\n")
		if len(images) > 0 {
			msg["images"] = images
		}
	}
	return converted, nil
}

// normalizeOllamaToolCallArguments 把消息里 tool_calls[].function.arguments 从
// OpenAI 风格的 JSON 字符串转成 Ollama 原生要求的 JSON 对象。
//
// openai.ConvertMessages 遵循 OpenAI /v1/chat/completions 协议，arguments 恒为字符串；
// Ollama /api/chat 却要求对象，收到字符串会返回 400。这里就地改写 map 值：
//   - 合法 JSON 对象字符串 → 解析为 map[string]any
//   - 空串（无参工具）或无法解析为对象 → 归一化为空对象 {}（避免透传字符串触发 400）
func normalizeOllamaToolCallArguments(msg map[string]any) {
	calls, ok := msg["tool_calls"].([]map[string]any)
	if !ok {
		return
	}
	for _, call := range calls {
		fn, ok := call["function"].(map[string]any)
		if !ok {
			continue
		}
		argStr, ok := fn["arguments"].(string)
		if !ok {
			// 已是对象或缺省，无需处理
			continue
		}
		obj := map[string]any{}
		if trimmed := strings.TrimSpace(argStr); trimmed != "" {
			if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
				obj = map[string]any{}
			}
		}
		fn["arguments"] = obj
	}
}

func contentPartsFromOpenAIMessage(content any) ([]map[string]any, bool) {
	switch parts := content.(type) {
	case []map[string]any:
		return parts, true
	case []any:
		result := make([]map[string]any, 0, len(parts))
		for _, part := range parts {
			m, ok := part.(map[string]any)
			if !ok {
				return nil, false
			}
			result = append(result, m)
		}
		return result, true
	default:
		return nil, false
	}
}

func ollamaImagePayload(raw string) (string, error) {
	payload := strings.TrimSpace(raw)
	if payload == "" {
		return "", fmt.Errorf("ollama image content is empty")
	}
	if strings.HasPrefix(strings.ToLower(payload), "data:") {
		comma := strings.Index(payload, ",")
		if comma < 0 {
			return "", fmt.Errorf("ollama image data URI missing base64 payload")
		}
		payload = strings.TrimSpace(payload[comma+1:])
	}
	if _, err := base64.StdEncoding.DecodeString(payload); err != nil {
		return "", fmt.Errorf("ollama image content must be base64 or data URI: %w", err)
	}
	return payload, nil
}

func ollamaThinkFromMetadata(metadata map[string]any) (any, bool) {
	if len(metadata) == 0 {
		return false, false
	}
	for _, key := range []string{"thinking", "think"} {
		value, exists := metadata[key]
		if !exists {
			continue
		}
		switch v := value.(type) {
		case bool:
			return v, true
		case string:
			switch strings.ToLower(strings.TrimSpace(v)) {
			case "on", "true", "1", "yes", "enabled":
				return true, true
			case "off", "false", "0", "no", "disabled":
				return false, true
			case "low", "medium", "high":
				return strings.ToLower(strings.TrimSpace(v)), true
			}
		}
	}
	return false, false
}

func ollamaReasoningPlan(req llm.CompletionRequest, baseURL string) (llm.ReasoningPlan, bool, error) {
	if plan, explicit, err := llm.PlanReasoningFromMetadata(req.Metadata, req.Model, baseURL); explicit {
		if err != nil {
			return plan, true, err
		}
		if plan.Receipt.Support != llm.ReasoningUnknown {
			return plan, true, nil
		}
	}
	think, explicit := ollamaThinkFromMetadata(req.Metadata)
	if !explicit {
		return llm.ReasoningPlan{}, false, nil
	}
	enabled := true
	if value, ok := think.(bool); ok {
		enabled = value
	}
	return llm.ReasoningPlan{
		Wire: map[string]any{"think": think},
		Receipt: llm.ReasoningReceipt{
			Version:     1,
			Enabled:     enabled,
			Support:     llm.ReasoningUnknown,
			Dialect:     llm.ReasoningDialectThink,
			Sent:        true,
			Application: llm.ReasoningApplicationUnknown,
		},
	}, true, nil
}

func ollamaReasoningSupport(capabilities []string) llm.ReasoningSupport {
	if capabilities == nil {
		return llm.ReasoningUnknown
	}
	for _, capability := range capabilities {
		if strings.EqualFold(strings.TrimSpace(capability), "thinking") {
			return llm.ReasoningSupported
		}
	}
	return llm.ReasoningUnsupported
}

// Ollama API 响应结构
type ollamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		Thinking  string `json:"thinking,omitempty"`
		ToolCalls []struct {
			Function struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls,omitempty"`
	} `json:"message"`
	Error              string `json:"error,omitempty"`
	Done               bool   `json:"done"`
	DoneReason         string `json:"done_reason"`
	TotalDuration      int    `json:"total_duration"`
	LoadDuration       int    `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int    `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int    `json:"eval_duration"`
}

// parseResponse 解析响应
func (p *Provider) parseResponse(resp *ollamaResponse, model string) *llm.CompletionResponse {
	result := &llm.CompletionResponse{
		ID:      resp.CreatedAt,
		Model:   model,
		Content: resp.Message.Content,
		Usage: llm.Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
	}

	if resp.Done {
		result.FinishReason = "stop"
	}

	// 解析工具调用
	for i, tc := range resp.Message.ToolCalls {
		args, _ := json.Marshal(tc.Function.Arguments)
		result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
			ID:        fmt.Sprintf("call_%d", i),
			Type:      "function",
			Name:      tc.Function.Name,
			Arguments: string(args),
		})
	}

	return result
}

type ollamaStreamParser struct {
	onSuccess         func()
	onFailure         func()
	successOnce       sync.Once
	reasoningEvidence streamx.ReasoningDisclosureEvidence
	metadata          map[string]any
	receipt           llm.ReasoningReceipt
}

func (p *ollamaStreamParser) Parse(data []byte) (*streamx.Chunk, error) {
	var resp ollamaResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		if p.onFailure != nil {
			p.onFailure()
		}
		return nil, err
	}
	if resp.Error != "" {
		if p.onFailure != nil {
			p.onFailure()
		}
		return nil, fmt.Errorf("ollama stream error: %s", resp.Error)
	}
	if p.onSuccess != nil {
		p.successOnce.Do(p.onSuccess)
	}

	chunk := &streamx.Chunk{
		ID:        resp.CreatedAt,
		Model:     resp.Model,
		Role:      resp.Message.Role,
		Content:   resp.Message.Content,
		Reasoning: resp.Message.Thinking,
		Raw:       data,
	}
	if resp.Message.Thinking != "" {
		chunk.ReasoningDisclosure = streamx.NewReasoningDisclosure(
			"ollama",
			"message.thinking",
			p.reasoningEvidence,
		)
		if p.receipt.Version == 0 {
			p.receipt = llm.ReasoningReceipt{
				Version:     1,
				Enabled:     true,
				Support:     llm.ReasoningUnknown,
				Dialect:     llm.ReasoningDialectThink,
				Application: llm.ReasoningApplicationUnknown,
			}
		}
		p.receipt = p.receipt.WithEvidence(true, true, 0)
		llm.PublishReasoningReceipt(p.metadata, p.receipt)
	}
	if p.receipt.Version != 0 {
		chunk.ReasoningEvidence = llm.ReasoningEvidenceMap(p.receipt)
	}
	if resp.Done {
		chunk.FinishReason = resp.DoneReason
		if chunk.FinishReason == "" {
			chunk.FinishReason = "stop"
		}
	}
	for i, tc := range resp.Message.ToolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			return nil, err
		}
		chunk.ToolCalls = append(chunk.ToolCalls, streamx.ToolCall{
			Index:     i,
			ID:        fmt.Sprintf("call_%d", i),
			Type:      "function",
			Name:      tc.Function.Name,
			Arguments: string(args),
		})
	}
	return chunk, nil
}

func (p *ollamaStreamParser) IsDone(data []byte) bool {
	var resp struct {
		Done bool `json:"done"`
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return false
	}
	return resp.Done
}

// Ping 检查 Ollama 服务是否可用
func (p *Provider) Ping(ctx context.Context) error {
	resp, err := p.doRequest(ctx, http.MethodGet, p.baseURL+"/api/tags", nil, false)
	if err != nil {
		return fmt.Errorf("ollama service unavailable: %w", err)
	}
	defer resp.Body.Close()

	return nil
}

// PullModel 拉取模型
func (p *Provider) PullModel(ctx context.Context, model string) error {
	payload := map[string]any{
		"name":   model,
		"stream": false,
	}
	body, _ := json.Marshal(payload)

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/pull", body, false)
	if err != nil {
		return fmt.Errorf("pull model failed: %w", err)
	}
	defer resp.Body.Close()

	return nil
}

func (p *Provider) doRequest(ctx context.Context, method, url string, body []byte, stream bool) (*http.Response, error) {
	client := p.httpClient
	requestTransport := p.transport
	if stream {
		client = p.streamHTTPClient
		requestTransport = p.streamTransport
	}
	isChat := strings.HasSuffix(url, "/api/chat")
	if isChat {
		if requestTransport != nil {
			if err := transport.ValidateURL(ctx, url, requestTransport.Policy()); err != nil {
				return nil, err
			}
			guardedClient, err := requestTransport.Client()
			if err != nil {
				return nil, err
			}
			client = guardedClient
			requestTransport = nil
		}
		if client == nil {
			client = http.DefaultClient
		}
		noRedirectClient := *client
		noRedirectClient.CheckRedirect = func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}
		client = &noRedirectClient
	}
	cfg := transport.Request{
		Provider:  p.Name(),
		Action:    strings.TrimPrefix(strings.TrimPrefix(url, strings.TrimRight(p.baseURL, "/")), "/"),
		Method:    method,
		URL:       url,
		Body:      body,
		Client:    client,
		Transport: requestTransport,
	}
	// /api/chat POST 不是幂等操作；transport 层重试会重复生成，并可能让旧
	// num_ctx 请求在新水位之后再次写出。失败交由调用方用新请求显式重试。
	if isChat {
		cfg.Retry = transport.RetryPolicy{MaxAttempts: 1}
	}
	if body != nil {
		cfg.SetHeaders = func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
		}
	}
	if stream {
		cfg.StreamIdle = 10 * time.Minute
	}
	return transport.Do(ctx, cfg)
}

// 确保实现了 Provider 接口
// EmbeddingProvider 接口验证在 embedding.go 中
var _ llm.Provider = (*Provider)(nil)
