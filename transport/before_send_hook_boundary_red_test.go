package transport

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestBeforeSendHookBoundPOSTDoesNotAutomaticallyFollowPreservingRedirect(
	t *testing.T,
) {
	for _, statusCode := range []int{
		http.StatusTemporaryRedirect,
		http.StatusPermanentRedirect,
	} {
		t.Run(http.StatusText(statusCode), func(t *testing.T) {
			var hookCalls atomic.Int32
			var initialPOSTs atomic.Int32
			var redirectedPOSTs atomic.Int32

			server := httptest.NewServer(http.HandlerFunc(
				func(w http.ResponseWriter, req *http.Request) {
					switch req.URL.Path {
					case "/initial":
						if req.Method != http.MethodPost {
							t.Errorf("initial method = %s, want POST", req.Method)
						}
						initialPOSTs.Add(1)
						w.Header().Set("Location", "/redirected")
						w.WriteHeader(statusCode)
					case "/redirected":
						if req.Method != http.MethodPost {
							t.Errorf("redirected method = %s, want POST", req.Method)
						}
						redirectedPOSTs.Add(1)
						w.WriteHeader(http.StatusOK)
					default:
						http.NotFound(w, req)
					}
				},
			))
			defer server.Close()

			ctx := WithBeforeSendHook(
				context.Background(),
				func(context.Context) error {
					hookCalls.Add(1)
					return nil
				},
			)
			resp, err := Do(ctx, Request{
				Provider: "test",
				Action:   "complete",
				Method:   http.MethodPost,
				URL:      server.URL + "/initial",
				Body:     []byte(`{"messages":[]}`),
				Client:   server.Client(),
				Retry: RetryPolicy{
					MaxAttempts: 1,
				},
			})
			if resp != nil && resp.Body != nil {
				_ = resp.Body.Close()
			}
			// A hook-bound operation may fail closed on the redirect response.
			// The invariant is the number of physical POSTs, not the returned
			// status/error shape.
			_ = err

			if got := hookCalls.Load(); got != 1 {
				t.Fatalf("before-send hook calls = %d, want 1", got)
			}
			if got := initialPOSTs.Load(); got != 1 {
				t.Fatalf("initial POSTs = %d, want 1", got)
			}
			if got := redirectedPOSTs.Load(); got != 0 {
				t.Fatalf(
					"redirected POSTs = %d, want 0: one durable hook must authorize exactly one physical POST",
					got,
				)
			}
		})
	}
}

func TestBeforeSendHookForActionSkipsAuxiliaryRequestAndRunsForTarget(
	t *testing.T,
) {
	var hookCalls atomic.Int32
	var tokenRequests atomic.Int32
	var completionRequests atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, req *http.Request) {
			switch req.URL.Path {
			case "/token":
				tokenRequests.Add(1)
			case "/complete":
				completionRequests.Add(1)
			default:
				http.NotFound(w, req)
				return
			}
			w.WriteHeader(http.StatusOK)
		},
	))
	defer server.Close()

	ctx := WithBeforeSendHookForAction(
		context.Background(),
		"complete",
		func(context.Context) error {
			hookCalls.Add(1)
			return nil
		},
	)

	tokenResp, err := Do(ctx, Request{
		Provider: "test",
		Action:   "token",
		Method:   http.MethodGet,
		URL:      server.URL + "/token",
		Client:   server.Client(),
		Retry: RetryPolicy{
			MaxAttempts: 1,
		},
	})
	if err != nil {
		t.Fatalf("auxiliary token request: %v", err)
	}
	if tokenResp != nil && tokenResp.Body != nil {
		_ = tokenResp.Body.Close()
	}
	if got := hookCalls.Load(); got != 0 {
		t.Fatalf(
			"before-send hook calls after auxiliary action = %d, want 0",
			got,
		)
	}

	completionResp, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      server.URL + "/complete",
		Body:     []byte(`{"messages":[]}`),
		Client:   server.Client(),
		Retry: RetryPolicy{
			MaxAttempts: 1,
		},
	})
	if err != nil {
		t.Fatalf("target completion request: %v", err)
	}
	if completionResp != nil && completionResp.Body != nil {
		_ = completionResp.Body.Close()
	}

	if got := tokenRequests.Load(); got != 1 {
		t.Fatalf("auxiliary token requests = %d, want 1", got)
	}
	if got := completionRequests.Load(); got != 1 {
		t.Fatalf("target completion requests = %d, want 1", got)
	}
	if got := hookCalls.Load(); got != 1 {
		t.Fatalf("target before-send hook calls = %d, want 1", got)
	}
}
