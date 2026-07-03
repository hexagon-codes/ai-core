package transport

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestDoReturnsProviderErrorWithDiagnostics(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("x-request-id", "req_123")
		w.Header().Set("retry-after-ms", "250")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("0123456789"))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), Request{
		Provider:       "test",
		Action:         "complete",
		URL:            srv.URL,
		Body:           []byte(`{"secret":"value"}`),
		ErrorBodyLimit: 4,
		PreviewLimit:   5,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if providerErr.StatusCode != http.StatusTooManyRequests || providerErr.Body != "0123" || !providerErr.BodyTruncated {
		t.Fatalf("unexpected provider error: %+v", providerErr)
	}
	if providerErr.RequestID != "req_123" || providerErr.RetryAfter != 250*time.Millisecond {
		t.Fatalf("missing diagnostics: %+v", providerErr)
	}
	if providerErr.RequestPreview != `{"sec` || !providerErr.RequestPreviewTruncated {
		t.Fatalf("unexpected request preview: %+v", providerErr)
	}
}

func TestDoRedactsSensitiveURLInProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), Request{
		Provider: "test",
		Action:   "complete",
		URL:      srv.URL + "/v1?key=SECRET-123&client_secret=SECRET-456&normal=keep",
		Retry:    RetryPolicy{MaxAttempts: 1},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if strings.Contains(providerErr.URL, "SECRET-123") || strings.Contains(providerErr.URL, "SECRET-456") {
		t.Fatalf("ProviderError.URL leaked secret query values: %q", providerErr.URL)
	}
	if !strings.Contains(providerErr.URL, "key=%5Bredacted%5D") ||
		!strings.Contains(providerErr.URL, "client_secret=%5Bredacted%5D") ||
		!strings.Contains(providerErr.URL, "normal=keep") {
		t.Fatalf("ProviderError.URL not sanitized as expected: %q", providerErr.URL)
	}
}

func TestDoRetriesTransientStatusThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	resp, err := Do(context.Background(), Request{
		Provider: "test",
		Action:   "complete",
		URL:      srv.URL,
		Retry: RetryPolicy{
			MaxAttempts: 2,
			BaseDelay:   time.Millisecond,
			MaxDelay:    time.Millisecond,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func TestDoHonorsRetryAfterBeyondBackoffMaxDelay(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	start := time.Now()
	resp, err := Do(context.Background(), Request{
		Provider: "test",
		Action:   "complete",
		URL:      srv.URL,
		Retry: RetryPolicy{
			MaxAttempts: 2,
			BaseDelay:   time.Millisecond,
			MaxDelay:    100 * time.Millisecond,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
	if elapsed := time.Since(start); elapsed < 900*time.Millisecond {
		t.Fatalf("elapsed = %s, want >= ~1s: server Retry-After must not be capped by backoff MaxDelay", elapsed)
	}
}

func TestDoCapsRetryAfterByMaxServerDelay(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "5")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	start := time.Now()
	resp, err := Do(context.Background(), Request{
		Provider: "test",
		Action:   "complete",
		URL:      srv.URL,
		Retry: RetryPolicy{
			MaxAttempts:    2,
			BaseDelay:      time.Millisecond,
			MaxDelay:       10 * time.Millisecond,
			MaxServerDelay: 150 * time.Millisecond,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	elapsed := time.Since(start)
	if elapsed < 150*time.Millisecond {
		t.Fatalf("elapsed = %s, want >= 150ms (capped Retry-After still beats MaxDelay)", elapsed)
	}
	if elapsed >= 2*time.Second {
		t.Fatalf("elapsed = %s, want Retry-After capped to MaxServerDelay=150ms", elapsed)
	}
}

func TestDoRetryAfterWaitInterruptedByContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("rate limited"))
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := Do(ctx, Request{
		Provider: "test",
		Action:   "complete",
		URL:      srv.URL,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded while waiting on Retry-After, got %v", err)
	}
	if elapsed := time.Since(start); elapsed >= time.Second {
		t.Fatalf("elapsed = %s, context cancellation must interrupt the Retry-After wait", elapsed)
	}
}

func TestRetryPolicyMaxServerDelayDefaults(t *testing.T) {
	if got := DefaultRetryPolicy().MaxServerDelay; got != 30*time.Second {
		t.Fatalf("DefaultRetryPolicy().MaxServerDelay = %s, want 30s", got)
	}
	normalized := normalizeRetryPolicy(RetryPolicy{MaxAttempts: 2})
	if normalized.MaxServerDelay != 30*time.Second {
		t.Fatalf("normalized zero MaxServerDelay = %s, want 30s", normalized.MaxServerDelay)
	}
}

func TestDoBlocksUnsafeCallerHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer provider" {
			t.Fatalf("Authorization = %q", got)
		}
		if got := r.Header.Get("X-Trace"); got != "trace-1" {
			t.Fatalf("X-Trace = %q", got)
		}
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("boom"))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), Request{
		Provider: "test",
		Action:   "complete",
		URL:      srv.URL,
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Authorization", "Bearer provider")
		},
		Headers: map[string]string{
			"Authorization": "Bearer attacker",
			"X-Trace":       "trace-1",
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if len(providerErr.RequestHeadersBlocked) != 1 || providerErr.RequestHeadersBlocked[0] != "Authorization" {
		t.Fatalf("blocked headers = %+v", providerErr.RequestHeadersBlocked)
	}
}

func TestValidateURLBlocksPrivateAndRequiresHTTPS(t *testing.T) {
	err := ValidateURL(context.Background(), "http://127.0.0.1/v1", &NetworkPolicy{})
	if err == nil {
		t.Fatal("expected network policy error")
	}
	if !errors.Is(err, ErrNetworkPolicy) || !strings.Contains(err.Error(), `scheme "http" is not allowed`) {
		t.Fatalf("unexpected error: %v", err)
	}

	err = ValidateURL(context.Background(), "http://255.255.255.255/v1", &NetworkPolicy{AllowHTTP: true})
	if err == nil {
		t.Fatal("expected private/internal error")
	}
	if !errors.Is(err, ErrNetworkPolicy) || !strings.Contains(err.Error(), "private/internal") {
		t.Fatalf("unexpected error: %v", err)
	}
}
