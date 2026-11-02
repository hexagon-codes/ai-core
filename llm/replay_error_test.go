package llm_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
	"github.com/hexagon-codes/ai-core/llm/router"
)

type replayMarkerError struct {
	unsafe bool
	cause  error
}

func (e *replayMarkerError) Error() string        { return "synthetic replay marker" }
func (e *replayMarkerError) UnsafeToReplay() bool { return e.unsafe }
func (e *replayMarkerError) Unwrap() error        { return e.cause }

func TestIsUnsafeToReplayWalksEveryErrorBranch(t *testing.T) {
	unsafe := &replayMarkerError{unsafe: true}
	for _, test := range []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil"},
		{name: "ordinary", err: errors.New("ordinary")},
		{name: "false", err: &replayMarkerError{}},
		{name: "false wraps true", err: &replayMarkerError{cause: unsafe}, want: true},
		{name: "false first in join", err: errors.Join(&replayMarkerError{}, fmt.Errorf("wrapped: %w", unsafe)), want: true},
		{name: "true first in join", err: errors.Join(unsafe, &replayMarkerError{}), want: true},
		{name: "nested aggregate", err: &router.MultiProviderError{Errors: map[string]error{"first": &replayMarkerError{}, "second": errors.Join(errors.New("ordinary"), unsafe)}}, want: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := llm.IsUnsafeToReplay(test.err); got != test.want {
				t.Fatalf("unsafe=%v want=%v", got, test.want)
			}
		})
	}
}

func TestUnknownResponsesOutcomeIsNotRetriedOrFailedOverActualHTTP(t *testing.T) {
	for _, withRouter := range []bool{false, true} {
		t.Run(fmt.Sprint(withRouter), func(t *testing.T) {
			var primaryCalls, fallbackCalls atomic.Int32
			primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				primaryCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"original-response","model":"actual-model","status":"in_progress","output":[{"id":"message","type":"message","role":"assistant","content":[{"type":"output_text","text":"already generated"}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
			}))
			defer primary.Close()
			fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				fallbackCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"id":"must-not-send","model":"fallback","status":"completed","output":[]}`)
			}))
			defer fallback.Close()
			provider := openai.New("synthetic", openai.WithBaseURL(primary.URL))
			ctx := openai.WithResponsesProtocol(context.Background(), "synthetic-scope")
			request := llm.CompletionRequest{Model: "requested-model", Messages: []llm.Message{{Role: llm.RoleUser, Content: "synthetic request"}}}
			var response *llm.CompletionResponse
			var err error
			if withRouter {
				r := router.New(router.WithFallback("fallback"), router.WithModelMap(map[string]string{"requested-model": "primary"})).Register("primary", provider).Register("fallback", openai.New("synthetic", openai.WithBaseURL(fallback.URL)))
				response, err = router.ExecuteWithRetry(ctx, r, request, 3)
			} else {
				response, err = llm.Chain(provider, llm.WithRetry(2, time.Millisecond)).Complete(ctx, request)
			}
			if err == nil || !llm.IsUnsafeToReplay(err) || response == nil || response.Content != "already generated" || response.FinishReason != "in_progress" || response.UsageReceipt == nil || response.UsageReceipt.TotalTokens != 3 || primaryCalls.Load() != 1 || fallbackCalls.Load() != 0 {
				t.Fatalf("response=%+v error=%v primary=%d fallback=%d", response, err, primaryCalls.Load(), fallbackCalls.Load())
			}
		})
	}
}
