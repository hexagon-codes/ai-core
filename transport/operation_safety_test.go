package transport

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

type operationSafetyRoundTripperFunc func(*http.Request) (*http.Response, error)

func (f operationSafetyRoundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func connectionResetError() error {
	return &net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET}
}

type operationSafetyTimeoutError struct{}

func (operationSafetyTimeoutError) Error() string   { return "upstream response header timeout" }
func (operationSafetyTimeoutError) Timeout() bool   { return true }
func (operationSafetyTimeoutError) Temporary() bool { return true }

func TestDoNonIdempotentNetworkFailureIsNotReplayed(t *testing.T) {
	var calls atomic.Int32
	client := &http.Client{Transport: operationSafetyRoundTripperFunc(
		func(req *http.Request) (*http.Response, error) {
			if req.Method != http.MethodPost {
				t.Fatalf("method = %s, want POST", req.Method)
			}
			calls.Add(1)
			return nil, connectionResetError()
		},
	)}
	ctx := WithOperationSafety(context.Background(), OperationSafetyNonIdempotent)

	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      "https://example.test/v1/chat/completions",
		Body:     []byte(`{"messages":[]}`),
		Client:   client,
	})
	if err == nil {
		t.Fatal("expected transport error")
	}
	if calls.Load() != 1 {
		t.Fatalf("physical POST attempts = %d, want 1", calls.Load())
	}
}

func TestDoRetriesIdempotentConnectionReset(t *testing.T) {
	var calls atomic.Int32
	client := &http.Client{Transport: operationSafetyRoundTripperFunc(
		func(req *http.Request) (*http.Response, error) {
			if calls.Add(1) == 1 {
				return nil, connectionResetError()
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`{}`)),
				Request:    req,
			}, nil
		},
	)}

	resp, err := Do(context.Background(), Request{
		Provider: "test",
		Action:   "list",
		Method:   http.MethodGet,
		URL:      "https://example.test/v1/models",
		Client:   client,
		Retry:    RetryPolicy{MaxAttempts: 2, BaseDelay: time.Millisecond},
	})
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()
	if calls.Load() != 2 {
		t.Fatalf("attempts = %d, want 2 for idempotent connection reset", calls.Load())
	}
}

func TestDoNonIdempotentNetworkTimeoutIsNotReplayed(t *testing.T) {
	var calls atomic.Int32
	client := &http.Client{Transport: operationSafetyRoundTripperFunc(
		func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, operationSafetyTimeoutError{}
		},
	)}
	ctx := WithOperationSafety(context.Background(), OperationSafetyNonIdempotent)

	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      "https://example.test/v1/chat/completions",
		Body:     []byte(`{"messages":[]}`),
		Client:   client,
	})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if calls.Load() != 1 {
		t.Fatalf("physical POST attempts = %d, want 1", calls.Load())
	}
}

func TestDoNonIdempotentDefinitiveHTTPStatusIsNotReplayed(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("upstream unavailable"))
	}))
	defer srv.Close()
	ctx := WithOperationSafety(context.Background(), OperationSafetyNonIdempotent)

	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		Method:   http.MethodPost,
		URL:      srv.URL,
		Body:     []byte(`{"messages":[]}`),
	})
	if err == nil {
		t.Fatal("expected provider error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("error = %T %v, want ProviderError", err, err)
	}
	if providerErr.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", providerErr.StatusCode, http.StatusServiceUnavailable)
	}
	if calls.Load() != 1 {
		t.Fatalf("physical POST attempts = %d, want 1", calls.Load())
	}
}
