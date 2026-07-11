package ollama

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestStickyNumCtxCacheIsBoundedAndEvictsLeastRecentlyUsedModel(t *testing.T) {
	p := New()

	// Give the first two entries visible watermarks, then keep model-0 hot.
	if got := numCtxForSuccessfulRequest(p, mkReq("model-0", 400, 30000)); got != 32768 {
		t.Fatalf("initial model-0 watermark = %d, want 32768", got)
	}
	if got := numCtxForSuccessfulRequest(p, mkReq("model-1", 400, 30000)); got != 32768 {
		t.Fatalf("initial model-1 watermark = %d, want 32768", got)
	}
	for i := 2; i < maxStickyNumCtxModels; i++ {
		numCtxForSuccessfulRequest(p, mkReq(fmt.Sprintf("model-%d", i), 400, 0))
	}
	numCtxForSuccessfulRequest(p, mkReq("model-0", 400, 0))

	// Adding one distinct model evicts cold model-1, not recently touched model-0.
	numCtxForSuccessfulRequest(p, mkReq("model-overflow", 400, 0))
	if got := numCtxForSuccessfulRequest(p, mkReq("model-0", 400, 0)); got != 32768 {
		t.Fatalf("recent model-0 watermark = %d, want retained 32768", got)
	}
	if got := numCtxForSuccessfulRequest(p, mkReq("model-1", 400, 0)); got != 4096 {
		t.Fatalf("evicted model-1 watermark = %d, want a fresh 4096 watermark", got)
	}
	if got := len(p.stickyNumCtx); got > maxStickyNumCtxModels {
		t.Fatalf("sticky num_ctx cache has %d entries, want at most %d", got, maxStickyNumCtxModels)
	}
	assertStickyNumCtxCacheConsistent(t, p)
}

func TestStickyNumCtxCacheRefreshAtCapacityDoesNotEvictAnotherModel(t *testing.T) {
	p := New()
	for i := 0; i < maxStickyNumCtxModels; i++ {
		maxTokens := 0
		if i == 1 {
			maxTokens = 30000
		}
		numCtxForSuccessfulRequest(p, mkReq(fmt.Sprintf("model-%d", i), 400, maxTokens))
	}

	numCtxForSuccessfulRequest(p, mkReq("model-0", 9000, 0))
	if got := len(p.stickyNumCtx); got != maxStickyNumCtxModels {
		t.Fatalf("refreshing an existing model changed cache size to %d, want %d", got, maxStickyNumCtxModels)
	}
	if got := p.numCtxForRequest(mkReq("model-1", 400, 0)); got != 32768 {
		t.Fatalf("refresh evicted another model: model-1 watermark = %d, want 32768", got)
	}
	assertStickyNumCtxCacheConsistent(t, p)
}

func TestStickyNumCtxCacheBoundsRetainedModelKeyBytes(t *testing.T) {
	p := New()
	for i := 0; i < maxStickyNumCtxModels; i++ {
		model := fmt.Sprintf("model-%d-%s", i, strings.Repeat("x", 4096))
		numCtxForSuccessfulRequest(p, mkReq(model, 400, 0))
	}

	retainedBytes := 0
	for model := range p.stickyNumCtx {
		retainedBytes += len(model)
	}
	for _, model := range p.stickyNumCtxOrder {
		retainedBytes += len(model)
	}
	maxRetainedBytes := maxStickyNumCtxModels * 2 * (maxStickyNumCtxKeyBytes + len("raw:"))
	if retainedBytes > maxRetainedBytes {
		t.Fatalf("sticky cache retained %d model-key bytes, want at most %d", retainedBytes, maxRetainedBytes)
	}
}

func TestStickyNumCtxLongKeyCannotAliasLiteralHashModelName(t *testing.T) {
	p := New()
	longModel := strings.Repeat("long-model-", 128)
	digest := sha256.Sum256([]byte(longModel))
	literalHashModel := fmt.Sprintf("sha256:%x", digest)

	numCtxForSuccessfulRequest(p, mkReq(longModel, 400, 30000))
	if got := p.numCtxForRequest(mkReq(literalHashModel, 400, 0)); got != 4096 {
		t.Fatalf("literal hash-like model inherited long-model watermark %d, want 4096", got)
	}
}

func TestModelLatestAliasSharesCapAndStickyWatermark(t *testing.T) {
	t.Run("model cap", func(t *testing.T) {
		p := New()
		p.models = []llm.ModelInfo{{ID: "registry.ollama.ai/library/alias-model:latest", MaxTokens: 8192}}
		if got := p.numCtxForRequest(mkReq("alias-model", 400, 30000)); got != 8192 {
			t.Fatalf("untagged alias num_ctx = %d, want model cap 8192", got)
		}
	})

	t.Run("sticky watermark", func(t *testing.T) {
		p := New()
		numCtxForSuccessfulRequest(p, mkReq("library/alias-model:latest", 400, 30000))
		if got := p.numCtxForRequest(mkReq("alias-model:latest", 400, 0)); got != 32768 {
			t.Fatalf("tagged alias watermark = %d, want shared 32768", got)
		}
	})
}

func TestZeroValueProviderCanCommitNumCtx(t *testing.T) {
	p := &Provider{}
	if got := numCtxForSuccessfulRequest(p, mkReq("zero-value", 400, 0)); got != 4096 {
		t.Fatalf("zero-value Provider num_ctx = %d, want 4096", got)
	}
}

func TestStickyNumCtxConcurrentUpdatesKeepMaximumWatermark(t *testing.T) {
	p := New()
	requests := []llm.CompletionRequest{
		mkReq("shared", 400, 0),
		mkReq("shared", 4000, 0),
		mkReq("shared", 9000, 0),
		mkReq("shared", 400, 30000),
	}

	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(req llm.CompletionRequest) {
			defer wg.Done()
			numCtxForSuccessfulRequest(p, req)
		}(requests[i%len(requests)])
	}
	wg.Wait()

	if got := numCtxForSuccessfulRequest(p, mkReq("shared", 400, 0)); got != 32768 {
		t.Fatalf("concurrent maximum watermark = %d, want 32768", got)
	}
}

func TestStickyNumCtxConcurrentHighCardinalityStaysBounded(t *testing.T) {
	p := New()
	var wg sync.WaitGroup
	for worker := 0; worker < 16; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < maxStickyNumCtxModels*4; i++ {
				model := fmt.Sprintf("worker-%d-model-%d", worker, i)
				numCtxForSuccessfulRequest(p, mkReq(model, 400, 0))
			}
		}(worker)
	}
	wg.Wait()

	p.stickyNumCtxMu.Lock()
	entries := len(p.stickyNumCtx)
	orderEntries := len(p.stickyNumCtxOrder)
	p.stickyNumCtxMu.Unlock()
	if entries != maxStickyNumCtxModels || orderEntries != maxStickyNumCtxModels {
		t.Fatalf("concurrent cache sizes map/order = %d/%d, want %d/%d",
			entries, orderEntries, maxStickyNumCtxModels, maxStickyNumCtxModels)
	}
	assertStickyNumCtxCacheConsistent(t, p)
}

func TestFailedRequestDoesNotPromoteStickyNumCtx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "model load failed", http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	req := mkReq("failed-model", 400, 30000)
	if _, err := p.Complete(context.Background(), req); err == nil {
		t.Fatal("Complete() error = nil, want upstream failure")
	}
	if got := p.numCtxForRequest(mkReq("failed-model", 400, 0)); got != 4096 {
		t.Fatalf("failed request promoted sticky num_ctx to %d, want fresh 4096", got)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestRequestEncodingFailureDoesNotPromoteStickyNumCtx(t *testing.T) {
	p := New()
	nan := math.NaN()
	req := mkReq("encoding-failed-model", 400, 30000)
	req.Temperature = &nan
	if _, err := p.Complete(context.Background(), req); err == nil {
		t.Fatal("Complete() error = nil, want JSON encoding failure")
	}
	if got := p.numCtxForRequest(mkReq("encoding-failed-model", 400, 0)); got != 4096 {
		t.Fatalf("encoding failure promoted sticky num_ctx to %d, want fresh 4096", got)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestMalformedSuccessResponseDoesNotPromoteStickyNumCtx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{not-json`)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	if _, err := p.Complete(context.Background(), mkReq("malformed-model", 400, 30000)); err == nil {
		t.Fatal("Complete() error = nil, want response decoding failure")
	}
	if got := p.numCtxForRequest(mkReq("malformed-model", 400, 0)); got != 4096 {
		t.Fatalf("malformed response promoted sticky num_ctx to %d, want fresh 4096", got)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestApplicationErrorResponseDoesNotPromoteStickyNumCtx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"error":"runner failed"}`)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	if _, err := p.Complete(context.Background(), mkReq("app-error-model", 400, 30000)); err == nil {
		t.Fatal("Complete() error = nil, want Ollama application error")
	}
	if got := p.numCtxForRequest(mkReq("app-error-model", 400, 0)); got != 4096 {
		t.Fatalf("application error promoted sticky num_ctx to %d, want fresh 4096", got)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestStreamParseErrorDoesNotPromoteStickyNumCtx(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "application error", body: `{"error":"runner failed"}` + "\n"},
		{name: "malformed JSON", body: `{not-json` + "\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/x-ndjson")
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			p := New(WithBaseURL(srv.URL))
			stream, err := p.Stream(context.Background(), mkReq("stream-error-model", 400, 30000))
			if err != nil {
				t.Fatalf("Stream() setup error = %v", err)
			}
			defer stream.Close()
			if _, err := stream.Collect(); err == nil {
				t.Fatal("Collect() error = nil, want stream parse error")
			}
			if got := p.numCtxForRequest(mkReq("stream-error-model", 400, 0)); got != 4096 {
				t.Fatalf("stream parse error promoted sticky num_ctx to %d, want fresh 4096", got)
			}
			assertStickyNumCtxCacheEmpty(t, p)
		})
	}
}

func TestEmptyStreamReleasesPendingNumCtxWithoutExplicitClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.Stream(context.Background(), mkReq("empty-stream-model", 400, 30000))
	if err != nil {
		t.Fatalf("Stream() setup error = %v", err)
	}
	if _, err := stream.Collect(); err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestWhitespaceOnlyStreamReleasesPendingNumCtxWithoutExplicitClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprint(w, "\n")
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.Stream(context.Background(), mkReq("whitespace-stream-model", 400, 30000))
	if err != nil {
		t.Fatalf("Stream() setup error = %v", err)
	}
	if _, err := stream.Collect(); err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestUnterminatedStreamChunkReleasesPendingNumCtxWithoutExplicitClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprint(w, `{"message":{"content":"unterminated"},"done":true}`)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.Stream(context.Background(), mkReq("unterminated-stream-model", 400, 30000))
	if err != nil {
		t.Fatalf("Stream() setup error = %v", err)
	}
	if _, err := stream.Collect(); err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestCanceledStreamReleasesPendingNumCtxWithoutExplicitClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	p := New(WithBaseURL(srv.URL))
	stream, err := p.Stream(ctx, mkReq("canceled-stream-model", 400, 30000))
	if err != nil {
		cancel()
		t.Fatalf("Stream() setup error = %v", err)
	}
	defer stream.Close()
	cancel()
	_, _ = stream.Collect()
	waitForStickyNumCtxCacheEmpty(t, p)
}

func waitForStickyNumCtxCacheEmpty(t *testing.T, p *Provider) {
	t.Helper()
	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		p.stickyNumCtxMu.Lock()
		empty := len(p.stickyNumCtx) == 0 && len(p.stickyNumCtxOrder) == 0 && len(p.pendingNumCtx) == 0
		p.stickyNumCtxMu.Unlock()
		if empty {
			return
		}
		select {
		case <-deadline.C:
			assertStickyNumCtxCacheEmpty(t, p)
			return
		case <-ticker.C:
		}
	}
}

func TestStreamFirstValidChunkCommitsEvenIfLaterChunkFails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"message":{"role":"assistant","content":"partial"},"done":false}`)
		fmt.Fprintln(w, `{"error":"late stream failure"}`)
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	stream, err := p.Stream(context.Background(), mkReq("partial-stream-model", 400, 30000))
	if err != nil {
		t.Fatalf("Stream() setup error = %v", err)
	}
	defer stream.Close()
	if _, err := stream.Collect(); err == nil {
		t.Fatal("Collect() error = nil, want late stream error")
	}
	// A valid chunk proves the runner loaded this num_ctx; retaining it prevents a reload.
	if got := p.numCtxForRequest(mkReq("partial-stream-model", 400, 0)); got != 32768 {
		t.Fatalf("watermark after first valid chunk = %d, want 32768", got)
	}
}

func assertStickyNumCtxCacheEmpty(t *testing.T, p *Provider) {
	t.Helper()
	p.stickyNumCtxMu.Lock()
	defer p.stickyNumCtxMu.Unlock()
	if len(p.stickyNumCtx) != 0 || len(p.stickyNumCtxOrder) != 0 || len(p.pendingNumCtx) != 0 {
		t.Fatalf("sticky cache map/order/pending = %d/%d/%d, want 0/0/0",
			len(p.stickyNumCtx), len(p.stickyNumCtxOrder), len(p.pendingNumCtx))
	}
}

func assertStickyNumCtxCacheConsistent(t *testing.T, p *Provider) {
	t.Helper()
	p.stickyNumCtxMu.Lock()
	defer p.stickyNumCtxMu.Unlock()
	if len(p.stickyNumCtx) != len(p.stickyNumCtxOrder) {
		t.Fatalf("sticky cache map/order size mismatch: %d/%d", len(p.stickyNumCtx), len(p.stickyNumCtxOrder))
	}
	seen := make(map[string]struct{}, len(p.stickyNumCtxOrder))
	for _, model := range p.stickyNumCtxOrder {
		if _, exists := seen[model]; exists {
			t.Fatalf("sticky cache order contains duplicate model %q", model)
		}
		seen[model] = struct{}{}
		if _, exists := p.stickyNumCtx[model]; !exists {
			t.Fatalf("sticky cache order contains model %q absent from map", model)
		}
	}
}

func TestSuccessfulRequestsPromoteStickyNumCtx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"message":{"role":"assistant","content":"ok"},"done":true}`)
	}))
	defer srv.Close()

	t.Run("complete", func(t *testing.T) {
		p := New(WithBaseURL(srv.URL))
		if _, err := p.Complete(context.Background(), mkReq("complete-model", 400, 30000)); err != nil {
			t.Fatalf("Complete() error = %v", err)
		}
		if got := p.numCtxForRequest(mkReq("complete-model", 400, 0)); got != 32768 {
			t.Fatalf("successful Complete watermark = %d, want 32768", got)
		}
	})

	t.Run("stream", func(t *testing.T) {
		p := New(WithBaseURL(srv.URL))
		stream, err := p.Stream(context.Background(), mkReq("stream-model", 400, 30000))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer stream.Close()
		if _, err := stream.Collect(); err != nil {
			t.Fatalf("Collect() error = %v", err)
		}
		if got := p.numCtxForRequest(mkReq("stream-model", 400, 0)); got != 32768 {
			t.Fatalf("successful Stream watermark = %d, want 32768", got)
		}
	})
}

func TestExplicitNumCtxNeverEntersStickyWatermarkThroughPublicAPIs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"message":{"role":"assistant","content":"ok"},"done":true}`)
	}))
	defer srv.Close()

	req := mkReq("explicit-model", 400, 100)
	req.Metadata = map[string]any{"num_ctx": 8192}

	t.Run("complete", func(t *testing.T) {
		p := New(WithBaseURL(srv.URL))
		if _, err := p.Complete(context.Background(), req); err != nil {
			t.Fatalf("Complete() error = %v", err)
		}
		assertStickyNumCtxCacheEmpty(t, p)
	})

	t.Run("stream", func(t *testing.T) {
		p := New(WithBaseURL(srv.URL))
		stream, err := p.Stream(context.Background(), req)
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		defer stream.Close()
		if _, err := stream.Collect(); err != nil {
			t.Fatalf("Collect() error = %v", err)
		}
		assertStickyNumCtxCacheEmpty(t, p)
	})
}

func TestNumCtxForRequestSaturatesOnOverflow(t *testing.T) {
	p := New()
	req := mkReq("overflow", 400, math.MaxInt)
	if got := p.numCtxForRequest(req); got != maxAutomaticNumCtx {
		t.Fatalf("num_ctx with MaxTokens=MaxInt = %d, want %d", got, maxAutomaticNumCtx)
	}
}

func TestBuildRequestBodyDefaultNumPredictMatchesAutomaticBudget(t *testing.T) {
	body, err := New().buildRequestBody(mkReq("default-output-budget", 400, 0), false)
	if err != nil {
		t.Fatalf("buildRequestBody() error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	options := payload["options"].(map[string]any)
	raw, ok := options["num_predict"].(float64)
	if !ok {
		t.Fatalf("default num_predict missing or non-numeric: %#v", options["num_predict"])
	}
	if got := int(raw); got != outputBudget(llm.CompletionRequest{}) {
		t.Fatalf("default num_predict = %d, want output budget %d", got, outputBudget(llm.CompletionRequest{}))
	}
}

func TestVisionRequestUsesConservativeImageBudget(t *testing.T) {
	p := New()
	req := llm.CompletionRequest{
		Model: "vision",
		Messages: []llm.Message{{
			Role: llm.RoleUser,
			MultiContent: []llm.ContentPart{{
				Type:     "image_url",
				ImageURL: &llm.ImageURL{URL: "data:image/png;base64,QUJDRA=="},
			}},
		}},
	}
	if got := p.numCtxForRequest(req); got < 16384 {
		t.Fatalf("single-image num_ctx = %d, want at least 16384", got)
	}
}

func TestBuildRequestBodyRejectsAutomaticContextThatCannotFitEstimate(t *testing.T) {
	t.Run("model cap", func(t *testing.T) {
		p := New()
		p.models = []llm.ModelInfo{{ID: "small8k", MaxTokens: 8192}}
		if _, err := p.buildRequestBody(mkReq("small8k", 20000, 0), false); err == nil {
			t.Fatal("buildRequestBody() error = nil, want estimated-context overflow error")
		}
	})

	t.Run("automatic maximum", func(t *testing.T) {
		if _, err := New().buildRequestBody(mkReq("overflow", 400, math.MaxInt), false); err == nil {
			t.Fatal("buildRequestBody() error = nil, want automatic-context overflow error")
		}
	})

	t.Run("image estimate", func(t *testing.T) {
		parts := make([]llm.ContentPart, 8)
		for i := range parts {
			parts[i] = llm.ContentPart{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,QUJDRA=="}}
		}
		req := llm.CompletionRequest{
			Model:    "many-images",
			Messages: []llm.Message{{Role: llm.RoleUser, MultiContent: parts}},
		}
		if needed, overflow := automaticNumCtxNeeded(req); !overflow || needed != maxAutomaticNumCtx {
			t.Fatalf("automaticNumCtxNeeded() = %d,%v, want %d,true", needed, overflow, maxAutomaticNumCtx)
		}
		if _, err := New().buildRequestBody(req, false); err == nil {
			t.Fatal("buildRequestBody() error = nil, want image context overflow error")
		}
	})
}

func TestToolFramingIsCountedInAutomaticContext(t *testing.T) {
	t.Run("definitions", func(t *testing.T) {
		req := mkReq("many-tools", 100, 0)
		for i := 0; i < 128; i++ {
			req.Tools = append(req.Tools, llm.NewToolDefinition("f", "d", &llm.Schema{Type: "object"}))
		}
		if got := New().numCtxForRequest(req); got < 8192 {
			t.Fatalf("128 tool definitions selected num_ctx=%d, want at least 8192", got)
		}
	})

	t.Run("historical calls", func(t *testing.T) {
		req := mkReq("many-tool-calls", 100, 0)
		for i := 0; i < 256; i++ {
			req.Messages[1].ToolCalls = append(req.Messages[1].ToolCalls, llm.ToolCallRef{
				ID: "call", Name: "f", Arguments: `{}`,
			})
		}
		if got := New().numCtxForRequest(req); got < 8192 {
			t.Fatalf("256 historical tool calls selected num_ctx=%d, want at least 8192", got)
		}
	})
}

type blockingNumCtxRoundTripper struct {
	smallEntered         chan struct{}
	smallWritten         chan struct{}
	largeEntered         chan struct{}
	releaseSmallWrite    chan struct{}
	releaseSmallResponse chan struct{}
	smallOnce            sync.Once
	smallWrittenOnce     sync.Once
	largeOnce            sync.Once
	mu                   sync.Mutex
	sent                 []int
}

func (rt *blockingNumCtxRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Options struct {
			NumCtx int `json:"num_ctx"`
		} `json:"options"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	switch payload.Options.NumCtx {
	case 4096:
		rt.smallOnce.Do(func() { close(rt.smallEntered) })
		<-rt.releaseSmallWrite
	case 32768:
		rt.largeOnce.Do(func() { close(rt.largeEntered) })
	}
	rt.mu.Lock()
	rt.sent = append(rt.sent, payload.Options.NumCtx)
	rt.mu.Unlock()
	if trace := httptrace.ContextClientTrace(req.Context()); trace != nil && trace.WroteRequest != nil {
		trace.WroteRequest(httptrace.WroteRequestInfo{})
	}
	if payload.Options.NumCtx == 4096 {
		rt.smallWrittenOnce.Do(func() { close(rt.smallWritten) })
		<-rt.releaseSmallResponse
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(`{"message":{"content":"ok"},"done":true}`)),
		Request:    req,
	}, nil
}

func TestConcurrentRequestsNeverSendLowerNumCtxAfterHigherNumCtx(t *testing.T) {
	rt := &blockingNumCtxRoundTripper{
		smallEntered:         make(chan struct{}),
		smallWritten:         make(chan struct{}),
		largeEntered:         make(chan struct{}),
		releaseSmallWrite:    make(chan struct{}),
		releaseSmallResponse: make(chan struct{}),
	}
	p := New(WithBaseURL("http://ollama.test"), WithHTTPClient(&http.Client{Transport: rt}))
	// The test transport is deterministic and does not dial; bypass the network-policy wrapper,
	// which intentionally only accepts *http.Transport.
	p.transport = nil
	p.streamTransport = nil

	smallDone := make(chan error, 1)
	go func() {
		_, err := p.Complete(context.Background(), mkReq("ordered-model", 400, 100))
		smallDone <- err
	}()
	<-rt.smallEntered

	largeReady := make(chan struct{})
	largeDone := make(chan error, 1)
	go func() {
		close(largeReady)
		_, err := p.Complete(context.Background(), mkReq("ordered-model", 400, 30000))
		largeDone <- err
	}()
	<-largeReady

	overtook := false
	select {
	case <-rt.largeEntered:
		overtook = true
	case <-time.After(250 * time.Millisecond):
	}
	close(rt.releaseSmallWrite)
	<-rt.smallWritten
	blockedAfterWrite := false
	select {
	case <-rt.largeEntered:
	case <-time.After(250 * time.Millisecond):
		blockedAfterWrite = true
	}
	close(rt.releaseSmallResponse)
	if err := <-smallDone; err != nil {
		t.Fatalf("small Complete() error = %v", err)
	}
	if err := <-largeDone; err != nil {
		t.Fatalf("large Complete() error = %v", err)
	}
	if overtook {
		t.Fatal("higher num_ctx request overtook an earlier lower request")
	}
	if blockedAfterWrite {
		t.Fatal("higher request stayed blocked after the lower request was fully written")
	}
	rt.mu.Lock()
	sent := append([]int(nil), rt.sent...)
	rt.mu.Unlock()
	if len(sent) != 2 || sent[0] != 4096 || sent[1] != 32768 {
		t.Fatalf("wire num_ctx order = %v, want [4096 32768]", sent)
	}
}

type statusCountingRoundTripper struct {
	calls atomic.Int32
}

func (rt *statusCountingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.calls.Add(1)
	return &http.Response{
		StatusCode: http.StatusInternalServerError,
		Status:     "500 Internal Server Error",
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(`{"error":"runner failed"}`)),
		Request:    req,
	}, nil
}

func TestChatRequestDoesNotRetrySameSerializedGeneration(t *testing.T) {
	rt := &statusCountingRoundTripper{}
	p := New(WithBaseURL("http://ollama.test"), WithHTTPClient(&http.Client{Transport: rt}))
	p.transport = nil
	p.streamTransport = nil

	if _, err := p.Complete(context.Background(), mkReq("no-duplicate", 400, 100)); err == nil {
		t.Fatal("Complete() error = nil, want upstream failure")
	}
	if got := rt.calls.Load(); got != 1 {
		t.Fatalf("serialized /api/chat attempts = %d, want exactly 1", got)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

func TestChatRequestDoesNotFollowRedirectAndReplayBody(t *testing.T) {
	var redirectedCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/chat":
			http.Redirect(w, r, "/redirected", http.StatusTemporaryRedirect)
		case "/redirected":
			redirectedCalls.Add(1)
			fmt.Fprint(w, `{"message":{"content":"unexpected replay"},"done":true}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	if _, err := p.Complete(context.Background(), mkReq("no-redirect", 400, 100)); err == nil {
		t.Fatal("Complete() error = nil, want redirect rejection")
	}
	if got := redirectedCalls.Load(); got != 0 {
		t.Fatalf("redirect target received %d replayed chat requests, want 0", got)
	}
	assertStickyNumCtxCacheEmpty(t, p)
}

type writeBlockingRoundTripper struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (rt *writeBlockingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.once.Do(func() { close(rt.entered) })
	<-rt.release
	if trace := httptrace.ContextClientTrace(req.Context()); trace != nil && trace.WroteRequest != nil {
		trace.WroteRequest(httptrace.WroteRequestInfo{})
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(`{"message":{"content":"ok"},"done":true}`)),
		Request:    req,
	}, nil
}

func TestCanceledRequestDoesNotWaitForNumCtxWriteGate(t *testing.T) {
	rt := &writeBlockingRoundTripper{entered: make(chan struct{}), release: make(chan struct{})}
	p := New(WithBaseURL("http://ollama.test"), WithHTTPClient(&http.Client{Transport: rt}))
	p.transport = nil
	p.streamTransport = nil

	firstDone := make(chan error, 1)
	go func() {
		_, err := p.Complete(context.Background(), mkReq("gate-model", 400, 100))
		firstDone <- err
	}()
	<-rt.entered

	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	secondDone := make(chan error, 1)
	go func() {
		_, err := p.Complete(canceledCtx, mkReq("gate-model", 400, 100))
		secondDone <- err
	}()

	returnedBeforeRelease := false
	var secondErr error
	select {
	case secondErr = <-secondDone:
		returnedBeforeRelease = true
	case <-time.After(250 * time.Millisecond):
	}
	close(rt.release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first Complete() error = %v", err)
	}
	if !returnedBeforeRelease {
		secondErr = <-secondDone
	}
	if !returnedBeforeRelease {
		t.Fatal("canceled request waited for another request's write gate")
	}
	if !errors.Is(secondErr, context.Canceled) {
		t.Fatalf("canceled Complete() error = %v, want context.Canceled", secondErr)
	}
}

func TestEstimateRequestTokensUsesSerializedMultiContent(t *testing.T) {
	p := New()
	req := llm.CompletionRequest{
		Model: "multi-content",
		Messages: []llm.Message{{
			Content: strings.Repeat("旧", 20000),
			MultiContent: []llm.ContentPart{
				{Type: "text", Text: "actual payload"},
				{Type: "text", Text: "not an image", ImageURL: &llm.ImageURL{URL: "ignored"}},
			},
		}},
	}
	if got := p.numCtxForRequest(req); got != 4096 {
		t.Fatalf("num_ctx counted content not serialized by MultiContent: got %d, want 4096", got)
	}
}

func TestNumCtxMetadataRejectsNonIntegerNumbers(t *testing.T) {
	invalid := []any{0.5, 3.7, math.NaN(), math.Inf(1), math.MaxFloat64, float32(0.5), float32(3.7)}
	for _, value := range invalid {
		t.Run(fmt.Sprintf("%T_%v", value, value), func(t *testing.T) {
			p := New()
			req := mkReq("metadata", 400, 0)
			req.Metadata = map[string]any{"num_ctx": value}
			if got := p.numCtxForRequest(req); got != 4096 {
				t.Fatalf("num_ctx metadata %v (%T) produced %d, want automatic 4096", value, value, got)
			}
		})
	}
}

func TestModelsReturnsDeepCopy(t *testing.T) {
	t.Run("cached", func(t *testing.T) {
		p := New()
		p.models = []llm.ModelInfo{{
			ID:       "cached-model",
			Name:     "cached-model",
			Features: []string{llm.FeatureStreaming},
		}}

		assertModelsResultCannotMutateCache(t, p, "cached-model")
	})

	t.Run("fresh fetch", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"models":[{"model":"fetched-model","name":"fetched-model","capabilities":["tools"],"details":{"context_length":8192}}]}`)
		}))
		defer srv.Close()

		p := New(WithBaseURL(srv.URL))
		assertModelsResultCannotMutateCache(t, p, "fetched-model")
	})
}

func assertModelsResultCannotMutateCache(t *testing.T, p *Provider, wantID string) {
	t.Helper()

	models := p.Models()
	if len(models) != 1 || len(models[0].Features) == 0 {
		t.Fatalf("Models() = %#v, want one model with features", models)
	}
	models[0].ID = "caller-mutated"
	models[0].Features[0] = "caller-mutated"

	again := p.Models()
	if again[0].ID != wantID {
		t.Fatalf("cached model ID = %q after caller mutation, want %q", again[0].ID, wantID)
	}
	if again[0].Features[0] == "caller-mutated" {
		t.Fatal("caller mutation leaked through nested Features slice")
	}
}

func TestModelsReturnedValueCanBeMutatedConcurrentlyWithoutRacingCache(t *testing.T) {
	p := New()
	p.models = []llm.ModelInfo{{
		ID:        "race-model",
		Name:      "race-model",
		MaxTokens: 8192,
		Features:  []string{llm.FeatureStreaming},
	}}
	external := p.Models()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			external[0].MaxTokens = 4096 + i
			external[0].Features[0] = fmt.Sprintf("caller-%d", i)
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 1000; i++ {
			p.numCtxForRequest(mkReq("race-model", 400, 0))
		}
	}()
	wg.Wait()
}
