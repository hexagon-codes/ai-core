package transport

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type beforeSendRoundTripperFunc func(*http.Request) (*http.Response, error)

func (f beforeSendRoundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestDoBeforeSendHookIsNotCalledForLocalPreflightFailures(t *testing.T) {
	tests := []struct {
		name    string
		request Request
	}{
		{
			name: "network policy URL validation",
			request: Request{
				Provider: "test",
				Action:   "complete",
				URL:      "http://127.0.0.1/v1/chat/completions",
				NetworkPolicy: &NetworkPolicy{
					AllowHTTP: true,
				},
			},
		},
		{
			name: "guarded client construction",
			request: Request{
				Provider: "test",
				Action:   "complete",
				URL:      "https://1.1.1.1/v1/chat/completions",
				Client: &http.Client{
					Transport: beforeSendRoundTripperFunc(func(*http.Request) (*http.Response, error) {
						return nil, errors.New("must not reach RoundTripper")
					}),
				},
				NetworkPolicy: &NetworkPolicy{},
			},
		},
		{
			name: "HTTP request construction",
			request: Request{
				Provider: "test",
				Action:   "complete",
				URL:      "://invalid-url",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var hookCalls atomic.Int32
			ctx := WithBeforeSendHook(context.Background(), func(context.Context) error {
				hookCalls.Add(1)
				return nil
			})

			_, err := Do(ctx, tt.request)
			if err == nil {
				t.Fatal("expected local preflight error")
			}
			if got := hookCalls.Load(); got != 0 {
				t.Fatalf("before-send hook calls = %d, want 0 before local preflight completes", got)
			}
		})
	}
}

func TestDoBeforeSendHookErrorPreventsHTTPClientDo(t *testing.T) {
	hookErr := errors.New("durable physical receipt claim failed")
	var hookCalls atomic.Int32
	var roundTripCalls atomic.Int32
	client := &http.Client{
		Transport: beforeSendRoundTripperFunc(func(*http.Request) (*http.Response, error) {
			roundTripCalls.Add(1)
			return nil, errors.New("must not reach RoundTripper")
		}),
	}
	ctx := WithBeforeSendHook(context.Background(), func(context.Context) error {
		hookCalls.Add(1)
		return hookErr
	})

	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      "https://example.test/v1/chat/completions",
		Body:     []byte(`{"messages":[]}`),
		Client:   client,
	})
	if !errors.Is(err, hookErr) {
		t.Fatalf("error = %v, want before-send hook error", err)
	}
	if got := hookCalls.Load(); got != 1 {
		t.Fatalf("before-send hook calls = %d, want 1", got)
	}
	if got := roundTripCalls.Load(); got != 0 {
		t.Fatalf("RoundTripper calls = %d, want 0 when before-send hook fails", got)
	}
}

func TestDoBeforeSendHookIsNotCalledAfterContextAlreadyCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var hookCalls atomic.Int32
	var roundTripCalls atomic.Int32
	ctx = WithBeforeSendHook(ctx, func(context.Context) error {
		hookCalls.Add(1)
		return nil
	})
	client := &http.Client{
		Transport: beforeSendRoundTripperFunc(func(*http.Request) (*http.Response, error) {
			roundTripCalls.Add(1)
			return nil, errors.New("cancelled request must not reach RoundTripper")
		}),
	}

	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      "https://example.test/v1/chat/completions",
		Client:   client,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error=%v, want context cancellation", err)
	}
	if got := hookCalls.Load(); got != 0 {
		t.Fatalf("cancelled request hook calls=%d, want 0", got)
	}
	if got := roundTripCalls.Load(); got != 0 {
		t.Fatalf("cancelled request RoundTrip calls=%d, want 0", got)
	}
}

func TestDoBeforeSendHookRunsExactlyOnceBeforeRoundTrip(t *testing.T) {
	type contextKey struct{}
	const contextValue = "durable-receipt-context"

	var hookCalls atomic.Int32
	var roundTripCalls atomic.Int32
	var hookCompleted atomic.Bool
	client := &http.Client{
		Transport: beforeSendRoundTripperFunc(func(req *http.Request) (*http.Response, error) {
			roundTripCalls.Add(1)
			if !hookCompleted.Load() {
				return nil, errors.New("RoundTripper observed request before before-send hook completed")
			}
			if got := req.Context().Value(contextKey{}); got != contextValue {
				return nil, errors.New("request context lost caller value")
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{"ok":true}`)),
				Request:    req,
			}, nil
		}),
	}
	baseCtx := context.WithValue(context.Background(), contextKey{}, contextValue)
	ctx := WithBeforeSendHook(baseCtx, func(hookCtx context.Context) error {
		hookCalls.Add(1)
		if got := hookCtx.Value(contextKey{}); got != contextValue {
			return errors.New("before-send hook did not receive request context")
		}
		hookCompleted.Store(true)
		return nil
	})

	resp, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      "https://example.test/v1/chat/completions",
		Body:     []byte(`{"messages":[]}`),
		Client:   client,
	})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()

	if got := hookCalls.Load(); got != 1 {
		t.Fatalf("before-send hook calls = %d, want 1", got)
	}
	if got := roundTripCalls.Load(); got != 1 {
		t.Fatalf("RoundTripper calls = %d, want 1", got)
	}
}

func TestDoBeforeSendHookNonIdempotentFailureIsNotReplayed(t *testing.T) {
	var hookCalls atomic.Int32
	var roundTripCalls atomic.Int32
	client := &http.Client{
		Transport: beforeSendRoundTripperFunc(func(*http.Request) (*http.Response, error) {
			roundTripCalls.Add(1)
			return nil, errors.New("connection reset after physical send")
		}),
	}
	ctx := WithOperationSafety(context.Background(), OperationSafetyNonIdempotent)
	ctx = WithBeforeSendHook(ctx, func(context.Context) error {
		hookCalls.Add(1)
		return nil
	})

	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      "https://example.test/v1/chat/completions",
		Body:     []byte(`{"messages":[]}`),
		Client:   client,
		Retry: RetryPolicy{
			MaxAttempts: 3,
			BaseDelay:   time.Nanosecond,
			MaxDelay:    time.Nanosecond,
		},
	})
	if err == nil {
		t.Fatal("expected transport error")
	}
	if got := hookCalls.Load(); got != 1 {
		t.Fatalf("before-send hook calls = %d, want 1 for non-idempotent operation", got)
	}
	if got := roundTripCalls.Load(); got != 1 {
		t.Fatalf("physical HTTP attempts = %d, want 1 for non-idempotent operation", got)
	}
}
