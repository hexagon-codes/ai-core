package transport

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

type operationSafetyRoundTripperFunc func(*http.Request) (*http.Response, error)

func (f operationSafetyRoundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type operationSafetyTemporaryError struct{}

func (operationSafetyTemporaryError) Error() string   { return "connection reset by peer" }
func (operationSafetyTemporaryError) Timeout() bool   { return false }
func (operationSafetyTemporaryError) Temporary() bool { return true }

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
			return nil, operationSafetyTemporaryError{}
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
