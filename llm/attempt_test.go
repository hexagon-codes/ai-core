package llm_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/openai"
)

func observeAttempts(ctx context.Context, ledger *llm.AttemptLedger, mergeErr *error) context.Context {
	ctx = llm.WithParentAttemptID(ctx, "logical-provider-call")
	return llm.WithAttemptObserver(ctx, llm.AttemptObserverFunc(func(_ context.Context, receipt llm.AttemptReceipt) {
		if err := ledger.Merge(receipt); err != nil {
			*mergeErr = err
		}
		if err := ledger.Merge(receipt); err != nil {
			*mergeErr = err
		}
	}))
}

func TestHTTPAttemptRetryKeepsFailedUsageAndDeduplicatesReceipts(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			w.Header().Set("x-request-id", "failed-request")
			w.WriteHeader(503)
			fmt.Fprint(w, `{"id":"failed-response","model":"actual-first","usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7},"error":{"message":"temporary","type":"server_error"}}`)
			return
		}
		w.Header().Set("x-request-id", "successful-request")
		fmt.Fprint(w, `{"id":"successful-response","model":"actual-final","choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}`)
	}))
	defer server.Close()
	provider := openai.New("synthetic", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
	ledger := llm.NewAttemptLedger()
	var mergeErr error
	response, err := provider.Complete(observeAttempts(t.Context(), ledger, &mergeErr), llm.CompletionRequest{Model: "requested-model"})
	if err != nil || mergeErr != nil || response == nil {
		t.Fatalf("response=%+v error=%v merge=%v", response, err, mergeErr)
	}
	summary := ledger.Summary()
	receipts := ledger.Snapshot()
	if calls.Load() != 2 || len(receipts) != 2 || summary.ReportedAttempts != 2 || summary.UnknownAttempts != 0 || summary.Usage.TotalTokens != 21 || summary.Usage.PromptTokens != 16 || summary.Usage.CompletionTokens != 5 {
		t.Fatalf("physical=%d summary=%+v receipts=%+v", calls.Load(), summary, receipts)
	}
	for _, receipt := range receipts {
		if receipt.ParentAttemptID != "logical-provider-call" || receipt.RequestedModel != "requested-model" || receipt.ResponseID == "" || receipt.RequestID == "" || !receipt.Complete || !receipt.Dispatched {
			t.Fatalf("physical receipt lost provenance: %+v", receipt)
		}
	}
}

func TestHTTPAttemptUnknownIsDistinctFromReportedZero(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			w.WriteHeader(503)
			fmt.Fprint(w, `{"error":{"message":"temporary"}}`)
			return
		}
		fmt.Fprint(w, `{"id":"zero","model":"actual","choices":[{"message":{"role":"assistant","content":"zero"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`)
	}))
	defer server.Close()
	ledger := llm.NewAttemptLedger()
	var mergeErr error
	_, err := openai.New("synthetic", openai.WithBaseURL(server.URL)).Complete(observeAttempts(t.Context(), ledger, &mergeErr), llm.CompletionRequest{Model: "requested"})
	summary := ledger.Summary()
	if err != nil || mergeErr != nil || calls.Load() != 2 || summary.UnknownAttempts != 1 || summary.ReportedAttempts != 1 || summary.Usage.TotalTokens != 0 {
		t.Fatalf("error=%v merge=%v physical=%d summary=%+v", err, mergeErr, calls.Load(), summary)
	}
}

func TestHTTPAttemptDecodeRetryRetainsFirstResponseUsage(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			fmt.Fprint(w, `{"id":"truncated","model":"actual-first","usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5},"choices":[`)
			return
		}
		fmt.Fprint(w, `{"id":"final","model":"actual-final","choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`)
	}))
	defer server.Close()
	ledger := llm.NewAttemptLedger()
	var mergeErr error
	_, err := openai.New("synthetic", openai.WithBaseURL(server.URL)).Complete(observeAttempts(t.Context(), ledger, &mergeErr), llm.CompletionRequest{Model: "requested"})
	summary := ledger.Summary()
	if err != nil || mergeErr != nil || calls.Load() != 2 || summary.ReportedAttempts != 2 || summary.Usage.TotalTokens != 14 {
		t.Fatalf("error=%v merge=%v calls=%d summary=%+v", err, mergeErr, calls.Load(), summary)
	}
	failed := 0
	for _, receipt := range ledger.Snapshot() {
		if receipt.Error != nil {
			failed++
		}
	}
	if failed != 1 {
		t.Fatalf("truncated response did not keep its failure fact: %+v", ledger.Snapshot())
	}
}

func TestHTTPAttemptCumulativeStreamIsNotAddedPerChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"stream-response\",\"model\":\"actual-model\",\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1,\"total_tokens\":5}}\n\n")
		fmt.Fprint(w, "data: {\"id\":\"stream-response\",\"model\":\"actual-model\",\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	ledger := llm.NewAttemptLedger()
	var mergeErr error
	stream, err := openai.New("synthetic", openai.WithBaseURL(server.URL)).Stream(observeAttempts(t.Context(), ledger, &mergeErr), llm.CompletionRequest{Model: "requested"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := stream.Collect()
	_ = stream.Close()
	summary := ledger.Summary()
	if err != nil || mergeErr != nil || result.Content != "ab" || summary.ReportedAttempts != 1 || summary.Usage.TotalTokens != 6 || summary.Usage.PromptTokens != 4 || summary.Usage.CompletionTokens != 2 {
		t.Fatalf("result=%+v error=%v merge=%v summary=%+v", result, err, mergeErr, summary)
	}
}

func TestHTTPAttemptPolicyRejectionIsNotDispatchedOrRetried(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); fmt.Fprint(w, `{}`) }))
	defer server.Close()
	ledger := llm.NewAttemptLedger()
	var mergeErr error
	denied := errors.New("existing host budget denied attempt")
	ctx := llm.WithBeforeHTTPAttempt(observeAttempts(t.Context(), ledger, &mergeErr), func(context.Context, llm.AttemptReceipt) error { return denied })
	provider := llm.Chain(openai.New("synthetic", openai.WithBaseURL(server.URL)), llm.WithRetry(2, time.Millisecond))
	_, err := provider.Complete(ctx, llm.CompletionRequest{Model: "requested"})
	receipts := ledger.Snapshot()
	summary := ledger.Summary()
	if !errors.Is(err, denied) || mergeErr != nil || calls.Load() != 0 || len(receipts) != 1 || receipts[0].Dispatched || !receipts[0].Complete || summary.UnknownAttempts != 0 || summary.ReportedAttempts != 0 {
		t.Fatalf("error=%v merge=%v HTTP=%d receipts=%+v summary=%+v", err, mergeErr, calls.Load(), receipts, summary)
	}
}

func TestHTTPAttemptObservationDoesNotBufferLargeResponseContent(t *testing.T) {
	content := strings.Repeat("large-output-", 20000) + ` quoted "usage":{"prompt_tokens":999999} `
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"large","model":"actual","choices":[{"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`, content)
	}))
	defer server.Close()
	ledger := llm.NewAttemptLedger()
	var mergeErr error
	response, err := openai.New("synthetic", openai.WithBaseURL(server.URL), openai.WithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true, AllowPrivate: true})).Complete(observeAttempts(t.Context(), ledger, &mergeErr), llm.CompletionRequest{Model: "requested"})
	if err != nil || mergeErr != nil || response.Content != content || ledger.Summary().Usage.TotalTokens != 5 {
		t.Fatalf("large content or receipt changed: error=%v merge=%v summary=%+v", err, mergeErr, ledger.Summary())
	}
}

func TestHTTPAttemptStreamCancellationRetainsReceiptAndFinishesOnce(t *testing.T) {
	for _, withUsage := range []bool{false, true} {
		t.Run(fmt.Sprint(withUsage), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				usage := ""
				if withUsage {
					usage = `,"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}`
				}
				fmt.Fprintf(w, "data: {\"id\":\"partial\",\"model\":\"actual\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]%s}\n\n", usage)
				w.(http.Flusher).Flush()
				<-r.Context().Done()
			}))
			defer server.Close()
			ledger := llm.NewAttemptLedger()
			var mergeErr error
			var finished atomic.Int32
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			ctx = llm.WithAttemptObserver(ctx, llm.AttemptObserverFunc(func(_ context.Context, receipt llm.AttemptReceipt) {
				if receipt.Complete {
					finished.Add(1)
				}
				if err := ledger.Merge(receipt); err != nil {
					mergeErr = err
				}
			}))
			stream, err := openai.New("synthetic", openai.WithBaseURL(server.URL)).Stream(ctx, llm.CompletionRequest{Model: "requested"})
			if err != nil {
				t.Fatal(err)
			}
			for chunk := range stream.Chunks() {
				if chunk.Content != "" {
					break
				}
			}
			cancel()
			_ = stream.Close()
			_ = stream.Close()
			<-stream.Done()
			summary := ledger.Summary()
			receipts := ledger.Snapshot()
			if mergeErr != nil || finished.Load() != 1 || len(receipts) != 1 || !receipts[0].Complete || !errors.Is(receipts[0].Error, context.Canceled) {
				t.Fatalf("finish=%d merge=%v receipts=%+v", finished.Load(), mergeErr, receipts)
			}
			if withUsage && (summary.ReportedAttempts != 1 || summary.Usage.TotalTokens != 7) {
				t.Fatalf("partial usage lost: %+v", summary)
			}
			if !withUsage && (summary.UnknownAttempts != 1 || summary.ReportedAttempts != 0) {
				t.Fatalf("unreported stream became zero usage: %+v", summary)
			}
		})
	}
}
