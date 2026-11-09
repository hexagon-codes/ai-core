package llm

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/transport"
)

// AttemptReceipt 表示一次进入受保护 RoundTripper 的 HTTP 尝试及其累计回执更新。
// 不等同于一次 Provider 方法调用；nil Usage 表示未知，UsageFields 保留部分实报边界。
type AttemptReceipt struct {
	AttemptID       string         `json:"attempt_id"`
	ParentAttemptID string         `json:"parent_attempt_id,omitempty"`
	ReceiptID       string         `json:"receipt_id"`
	Sequence        uint64         `json:"sequence"`
	Provider        string         `json:"provider"`
	RequestedModel  string         `json:"requested_model,omitempty"`
	Model           string         `json:"model,omitempty"`
	Protocol        string         `json:"protocol,omitempty"`
	Action          string         `json:"action"`
	StatusCode      int            `json:"status_code,omitempty"`
	RequestID       string         `json:"request_id,omitempty"`
	ResponseID      string         `json:"response_id,omitempty"`
	Usage           *Usage         `json:"usage,omitempty"`
	UsageFields     []string       `json:"usage_fields,omitempty"`
	UsageDetails    map[string]int `json:"usage_details,omitempty"`
	UsageMode       string         `json:"usage_mode"`
	StartedAt       time.Time      `json:"started_at"`
	FinishedAt      time.Time      `json:"finished_at,omitempty"`
	Complete        bool           `json:"complete"`
	Dispatched      bool           `json:"dispatched"`
	Error           error          `json:"-"`
}

type AttemptObserver interface {
	// OnAttempt 同步接收独立事实副本；不同 HTTP 尝试可能并发调用。
	// 回调不拥有响应、连接或执行状态，不应用它阻塞等待当前响应结束。
	OnAttempt(context.Context, AttemptReceipt)
}
type AttemptObserverFunc func(context.Context, AttemptReceipt)

func (f AttemptObserverFunc) OnAttempt(ctx context.Context, receipt AttemptReceipt) { f(ctx, receipt) }

type BeforeHTTPAttempt func(context.Context, AttemptReceipt) error
type attemptObserverKey struct{}
type parentAttemptKey struct{}
type beforeHTTPAttemptKey struct{}
type attemptProviderKey struct{}
type attemptModelKey struct{}

// AttemptObserverFromContext 只读取得当前逻辑边界的观察器，不自动继承或组合父账本。
func AttemptObserverFromContext(ctx context.Context) (AttemptObserver, bool) {
	if ctx == nil {
		return nil, false
	}
	observer, ok := ctx.Value(attemptObserverKey{}).(AttemptObserver)
	return observer, ok && observer != nil
}

func BeforeHTTPAttemptFromContext(ctx context.Context) (BeforeHTTPAttempt, bool) {
	if ctx == nil {
		return nil, false
	}
	before, ok := ctx.Value(beforeHTTPAttemptKey{}).(BeforeHTTPAttempt)
	return before, ok && before != nil
}

func WithParentAttemptID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, parentAttemptKey{}, id)
}
func WithAttemptProvider(ctx context.Context, provider string) context.Context {
	return context.WithValue(ctx, attemptProviderKey{}, provider)
}
func WithAttemptModel(ctx context.Context, model string) context.Context {
	return context.WithValue(ctx, attemptModelKey{}, model)
}

func WithAttemptObserver(ctx context.Context, observer AttemptObserver) context.Context {
	if observer == nil {
		return ctx
	}
	ctx = context.WithValue(ctx, attemptObserverKey{}, observer)
	return transport.WithClientDecorator(ctx, decorateAttemptClient)
}

// WithBeforeHTTPAttempt 只执行宿主明确注入的策略；不附加默认预算或限制。
func WithBeforeHTTPAttempt(ctx context.Context, before BeforeHTTPAttempt) context.Context {
	if before == nil {
		return ctx
	}
	ctx = context.WithValue(ctx, beforeHTTPAttemptKey{}, before)
	if _, ok := ctx.Value(attemptObserverKey{}).(AttemptObserver); ok {
		return ctx
	}
	return transport.WithClientDecorator(ctx, decorateAttemptClient)
}

type attemptRoundTripper struct {
	base                    http.RoundTripper
	provider, action, model string
}

func decorateAttemptClient(ctx context.Context, client *http.Client, request transport.Request) (*http.Client, error) {
	if request.Action == "auth" || request.Action == "api/show" || request.Action == "api/tags" {
		return client, nil
	}
	clone := *client
	base := clone.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	if _, already := base.(*attemptRoundTripper); already {
		return client, nil
	}
	model, _ := ctx.Value(attemptModelKey{}).(string)
	scanner := newReceiptScanner(func(path []string, value string, numeric bool) {
		if len(path) == 1 && path[0] == "model" && !numeric {
			model = value
		}
	})
	scanner.Feed(request.Body)
	provider := request.Provider
	if configured, _ := ctx.Value(attemptProviderKey{}).(string); configured != "" {
		provider = configured
	}
	clone.Transport = &attemptRoundTripper{base: base, provider: provider, action: request.Action, model: model}
	return &clone, nil
}

func (t *attemptRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	ctx := request.Context()
	parent, _ := ctx.Value(parentAttemptKey{}).(string)
	receipt := AttemptReceipt{AttemptID: rand.Text(), ParentAttemptID: parent, Provider: t.provider, RequestedModel: t.model, Action: t.action, UsageMode: "cumulative", StartedAt: time.Now()}
	observer, _ := ctx.Value(attemptObserverKey{}).(AttemptObserver)
	if before, _ := ctx.Value(beforeHTTPAttemptKey{}).(BeforeHTTPAttempt); before != nil {
		if err := before(ctx, copyAttemptReceipt(receipt)); err != nil {
			receipt.Sequence, receipt.ReceiptID, receipt.Complete, receipt.FinishedAt, receipt.Error = 1, receipt.AttemptID+"/1", true, time.Now(), err
			if observer != nil {
				observer.OnAttempt(ctx, copyAttemptReceipt(receipt))
			}
			return nil, &transport.AttemptRejectedError{Cause: err}
		}
	}
	tracker := &attemptTracker{ctx: ctx, observer: observer, receipt: receipt, known: make(map[string]bool)}
	tracker.receipt.Dispatched = true
	tracker.publish()
	response, err := t.base.RoundTrip(request)
	if err != nil {
		tracker.finish(err)
		return response, err
	}
	if response == nil {
		tracker.finish(errors.New("nil HTTP response"))
		return nil, nil
	}
	tracker.receipt.StatusCode = response.StatusCode
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		tracker.receipt.Error = fmt.Errorf("upstream HTTP status %d", response.StatusCode)
	}
	tracker.receipt.RequestID = RequestIDFromHeader(response.Header)
	tracker.receipt.Protocol = "json"
	if strings.Contains(response.Header.Get("Content-Type"), "event-stream") {
		tracker.receipt.Protocol = "sse"
	}
	tracker.scanner = newReceiptScanner(tracker.fact)
	if response.Body == nil {
		tracker.finish(nil)
		return response, nil
	}
	response.Body = &attemptBody{ReadCloser: response.Body, tracker: tracker}
	return response, nil
}

type attemptTracker struct {
	mu       sync.Mutex
	ctx      context.Context
	observer AttemptObserver
	receipt  AttemptReceipt
	known    map[string]bool
	scanner  *receiptScanner
	finished bool
}

func (t *attemptTracker) publish() {
	if t.observer == nil {
		return
	}
	t.receipt.Sequence++
	t.receipt.ReceiptID = fmt.Sprintf("%s/%d", t.receipt.AttemptID, t.receipt.Sequence)
	t.observer.OnAttempt(t.ctx, copyAttemptReceipt(t.receipt))
}
func (t *attemptTracker) fact(path []string, value string, numeric bool) {
	if len(path) == 0 {
		return
	}
	key := path[len(path)-1]
	prefix := strings.Join(path[:len(path)-1], ".")
	if !numeric {
		if prefix == "" || prefix == "response" || prefix == "message" {
			switch key {
			case "model", "modelVersion":
				t.receipt.Model = value
			case "id", "response_id", "responseId":
				t.receipt.ResponseID = value
			}
		}
		return
	}
	var count int
	if err := json.Unmarshal([]byte(value), &count); err != nil || count < 0 {
		return
	}
	field := ""
	switch prefix {
	case "usage", "response.usage", "message.usage", "usageMetadata":
		switch key {
		case "prompt_tokens", "input_tokens", "promptTokenCount":
			field = "prompt_tokens"
		case "completion_tokens", "output_tokens", "candidatesTokenCount":
			field = "completion_tokens"
		case "total_tokens", "totalTokenCount":
			field = "total_tokens"
		case "cachedContentTokenCount":
			field = "cached_tokens"
		case "thoughtsTokenCount":
			field = "reasoning_tokens"
		case "cache_creation_input_tokens":
			field = "cache_creation_tokens"
		case "cache_read_input_tokens":
			field = "cached_tokens"
		}
	case "usage.prompt_tokens_details", "usage.input_tokens_details", "response.usage.input_tokens_details":
		if key == "cached_tokens" {
			field = "cached_tokens"
		}
	case "usage.completion_tokens_details", "usage.output_tokens_details", "response.usage.output_tokens_details":
		if key == "reasoning_tokens" {
			field = "reasoning_tokens"
		}
	case "":
		switch key {
		case "prompt_eval_count":
			field = "prompt_tokens"
		case "eval_count":
			field = "completion_tokens"
		}
	}
	if field == "" {
		return
	}
	if t.receipt.Usage == nil {
		t.receipt.Usage = &Usage{}
	}
	u := t.receipt.Usage
	switch field {
	case "prompt_tokens":
		u.PromptTokens = max(u.PromptTokens, count)
	case "completion_tokens":
		u.CompletionTokens = max(u.CompletionTokens, count)
	case "total_tokens":
		u.TotalTokens = max(u.TotalTokens, count)
	case "cached_tokens":
		u.CacheReadTokens = max(u.CacheReadTokens, count)
	case "cache_creation_tokens":
		u.CacheCreationTokens = max(u.CacheCreationTokens, count)
	case "reasoning_tokens":
		if t.receipt.UsageDetails == nil {
			t.receipt.UsageDetails = make(map[string]int)
		}
		t.receipt.UsageDetails[field] = max(t.receipt.UsageDetails[field], count)
	}
	t.known[field] = true
	t.receipt.UsageFields = t.receipt.UsageFields[:0]
	for name := range t.known {
		t.receipt.UsageFields = append(t.receipt.UsageFields, name)
	}
	sort.Strings(t.receipt.UsageFields)
	t.publish()
}
func (t *attemptTracker) feed(data []byte) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.finished && t.scanner != nil {
		t.scanner.Feed(data)
	}
}
func (t *attemptTracker) finish(err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.finished {
		return
	}
	t.finished = true
	t.receipt.Complete = true
	t.receipt.FinishedAt = time.Now()
	if err != nil && !errors.Is(err, io.EOF) {
		t.receipt.Error = err
	} else if t.ctx.Err() != nil {
		t.receipt.Error = context.Cause(t.ctx)
	} else if t.receipt.Protocol == "json" && t.scanner != nil && !t.scanner.rootComplete {
		t.receipt.Error = errors.New("HTTP response JSON is incomplete")
	}
	t.publish()
}

type attemptBody struct {
	io.ReadCloser
	tracker *attemptTracker
	once    sync.Once
}

func (b *attemptBody) Read(data []byte) (int, error) {
	n, err := b.ReadCloser.Read(data)
	if n > 0 {
		b.tracker.feed(data[:n])
	}
	if err != nil {
		b.tracker.finish(err)
	}
	return n, err
}
func (b *attemptBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(func() { b.tracker.finish(err) })
	return err
}

func copyAttemptReceipt(receipt AttemptReceipt) AttemptReceipt {
	if receipt.Usage != nil {
		usage := *receipt.Usage
		receipt.Usage = &usage
	}
	receipt.UsageFields = append([]string(nil), receipt.UsageFields...)
	if receipt.UsageDetails != nil {
		details := make(map[string]int, len(receipt.UsageDetails))
		for key, value := range receipt.UsageDetails {
			details[key] = value
		}
		receipt.UsageDetails = details
	}
	return receipt
}

var ErrAttemptReceiptConflict = errors.New("LLM attempt receipt identity conflict")

// IsHTTPAttemptRejected 区分宿主主动拒绝与可重试网络失败。
func IsHTTPAttemptRejected(err error) bool {
	var rejected *transport.AttemptRejectedError
	return errors.As(err, &rejected)
}

type AttemptLedger struct {
	mu       sync.Mutex
	attempts map[string]AttemptReceipt
	seen     map[string]AttemptReceipt
}

func NewAttemptLedger() *AttemptLedger {
	return &AttemptLedger{attempts: make(map[string]AttemptReceipt), seen: make(map[string]AttemptReceipt)}
}

// Merge 按原 Attempt/Receipt 身份幂等归档累计值；冲突返回错误，不修正为零。
// 乱序旧更新不覆盖新事实，使用者必须处理返回错误。
func (l *AttemptLedger) Merge(receipt AttemptReceipt) error {
	if receipt.AttemptID == "" || receipt.ReceiptID == "" || receipt.Sequence == 0 || receipt.UsageMode != "cumulative" {
		return fmt.Errorf("LLM attempt receipt identity and cumulative mode are required")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.attempts == nil {
		l.attempts = make(map[string]AttemptReceipt)
	}
	if l.seen == nil {
		l.seen = make(map[string]AttemptReceipt)
	}
	if old, ok := l.seen[receipt.ReceiptID]; ok {
		if !equalAttemptReceipt(old, receipt) {
			return ErrAttemptReceiptConflict
		}
		return nil
	}
	if old, ok := l.attempts[receipt.AttemptID]; ok {
		if old.ParentAttemptID != receipt.ParentAttemptID || old.Provider != receipt.Provider || old.RequestedModel != receipt.RequestedModel || !old.StartedAt.Equal(receipt.StartedAt) {
			return ErrAttemptReceiptConflict
		}
		if receipt.Sequence == old.Sequence && !equalAttemptReceipt(old, receipt) {
			return ErrAttemptReceiptConflict
		}
		if receipt.Sequence <= old.Sequence {
			l.seen[receipt.ReceiptID] = copyAttemptReceipt(receipt)
			return nil
		}
		if old.Complete && !receipt.Complete {
			return ErrAttemptReceiptConflict
		}
		if old.Usage != nil && (receipt.Usage == nil || receipt.Usage.PromptTokens < old.Usage.PromptTokens || receipt.Usage.CompletionTokens < old.Usage.CompletionTokens || receipt.Usage.TotalTokens < old.Usage.TotalTokens || receipt.Usage.CacheReadTokens < old.Usage.CacheReadTokens || receipt.Usage.CacheCreationTokens < old.Usage.CacheCreationTokens) {
			return ErrAttemptReceiptConflict
		}
	}
	l.seen[receipt.ReceiptID] = copyAttemptReceipt(receipt)
	l.attempts[receipt.AttemptID] = copyAttemptReceipt(receipt)
	return nil
}
func equalAttemptReceipt(a, b AttemptReceipt) bool {
	aError, bError := "", ""
	if a.Error != nil {
		aError = a.Error.Error()
	}
	if b.Error != nil {
		bError = b.Error.Error()
	}
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y) && aError == bError
}
func (l *AttemptLedger) Snapshot() []AttemptReceipt {
	l.mu.Lock()
	defer l.mu.Unlock()
	result := make([]AttemptReceipt, 0, len(l.attempts))
	for _, receipt := range l.attempts {
		result = append(result, copyAttemptReceipt(receipt))
	}
	sort.Slice(result, func(i, j int) bool { return result[i].AttemptID < result[j].AttemptID })
	return result
}

type AttemptUsageSummary struct {
	Usage            Usage
	ReportedAttempts int
	UnknownAttempts  int
	PartialAttempts  int
}

// Summary 只累计已派发尝试的已知值，不是供应商账单或业务费用结算。
// UnknownAttempts/PartialAttempts 必须随 Usage 一并解释，不能把缺失维度当零消耗。
func (l *AttemptLedger) Summary() AttemptUsageSummary {
	var summary AttemptUsageSummary
	for _, receipt := range l.Snapshot() {
		if !receipt.Dispatched {
			continue
		}
		if receipt.Usage == nil {
			summary.UnknownAttempts++
			continue
		}
		summary.ReportedAttempts++
		u := receipt.Usage
		summary.Usage.PromptTokens += u.PromptTokens
		summary.Usage.CompletionTokens += u.CompletionTokens
		summary.Usage.TotalTokens += u.TotalTokens
		summary.Usage.CacheReadTokens += u.CacheReadTokens
		summary.Usage.CacheCreationTokens += u.CacheCreationTokens
		prompt, completion, total := false, false, false
		for _, field := range receipt.UsageFields {
			if field == "prompt_tokens" {
				prompt = true
			}
			if field == "completion_tokens" {
				completion = true
			}
			if field == "total_tokens" {
				total = true
			}
		}
		if !prompt || !completion || !total {
			summary.PartialAttempts++
		}
	}
	return summary
}

// RequestIDFromHeader 从上游响应头提取请求追踪 ID。
// 兼容 OpenAI/Anthropic 常用的 x-request-id 与 request-id 命名。
func RequestIDFromHeader(header http.Header) string {
	for _, key := range []string{"x-request-id", "request-id"} {
		if value := header.Get(key); value != "" {
			return value
		}
	}
	return ""
}
