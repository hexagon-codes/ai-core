package transport

// 本文件承接原 llm/internal/upstream/request_test.go 的全部用例：
// upstream 包曾是 transport 的纯转发壳（含一份复制的私有 SSRF 判定），
// 壳删除后用例整体迁移至此，对私有复制品的直测改写为对 transport
// 公开行为（Do / ValidateURL / ClientWithNetworkPolicy）的验证。

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestDoErrorStatusReturnsProviderErrorWithDiagnostics(t *testing.T) {
	var gotContentLength int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotContentLength = r.ContentLength
		w.Header().Set("x-request-id", "req_123")
		w.Header().Set("retry-after-ms", "250")
		w.Header().Set("x-ratelimit-remaining-requests", "0")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("0123456789"))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), Request{
		Provider:       "test-provider",
		Action:         "complete",
		Method:         http.MethodPost,
		URL:            srv.URL,
		Body:           []byte(`{"hello":"world"}`),
		ErrorBodyLimit: 4,
		PreviewLimit:   8,
		SetHeaders: func(req *http.Request) {
			req.Header.Set("Content-Type", "application/json")
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if providerErr.StatusCode != http.StatusBadGateway || providerErr.Body != "0123" || !providerErr.BodyTruncated {
		t.Fatalf("unexpected provider error diagnostics: %+v", providerErr)
	}
	if providerErr.RequestPreview != `{"hello"` || !providerErr.RequestPreviewTruncated {
		t.Fatalf("unexpected request preview: %+v", providerErr)
	}
	if providerErr.RequestID != "req_123" || providerErr.RetryAfter != 250*time.Millisecond {
		t.Fatalf("unexpected response diagnostics: %+v", providerErr)
	}
	if providerErr.ResponseHeaders["x-ratelimit-remaining-requests"] != "0" {
		t.Fatalf("expected diagnostic response headers, got %+v", providerErr.ResponseHeaders)
	}
	if providerErr.Elapsed <= 0 {
		t.Fatalf("expected elapsed duration, got %s", providerErr.Elapsed)
	}
	if gotContentLength != int64(len(`{"hello":"world"}`)) {
		t.Fatalf("ContentLength = %d", gotContentLength)
	}
}

func TestDoAppliesOnlySafeHeaders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer provider" {
			t.Fatalf("Authorization = %q, want provider auth", got)
		}
		if got := r.Header.Get("X-Api-Key"); got != "provider-key" {
			t.Fatalf("X-Api-Key = %q, want provider key", got)
		}
		if got := r.Header.Get("X-Debug-Trace"); got != "trace-1" {
			t.Fatalf("safe custom header missing, got %q", got)
		}
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	resp, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      srv.URL,
		SetHeaders: func(req *http.Request) {
			req.Header.Set("Authorization", "Bearer provider")
			req.Header.Set("X-Api-Key", "provider-key")
		},
		Headers: map[string]string{
			"Authorization": "Bearer attacker",
			"x-api-key":     "attacker-key",
			"Content-Type":  "text/plain",
			"X-Debug-Trace": "trace-1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
}

func TestDoBlockedHeadersAppearInProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      srv.URL,
		Headers: map[string]string{
			"Authorization": "Bearer attacker",
			"X-Trace":       "trace",
		},
	})
	if err == nil {
		t.Fatal("expected provider error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if len(providerErr.RequestHeadersBlocked) != 1 || providerErr.RequestHeadersBlocked[0] != "Authorization" {
		t.Fatalf("unexpected blocked headers: %+v", providerErr.RequestHeadersBlocked)
	}
}

func TestDoTimeoutWrapsCause(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      srv.URL,
		Timeout:  5 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected timeout")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if !strings.Contains(err.Error(), "request failed") {
		t.Fatalf("error should include request failure context, got %v", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded cause, got %v", err)
	}
}

func TestDoTimeoutKeepsSuccessfulResponseBodyReadable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected flusher")
		}
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
		time.Sleep(20 * time.Millisecond)
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	resp, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      srv.URL,
		Timeout:  time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("successful response body should remain readable within timeout: %v", err)
	}
	if string(body) != "ok" {
		t.Fatalf("body = %q, want ok", body)
	}
}

func TestDoStreamIdleTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
		time.Sleep(80 * time.Millisecond)
		_, _ = w.Write([]byte("data: late\n\n"))
	}))
	defer srv.Close()

	resp, err := Do(context.Background(), Request{
		Provider:   "test-provider",
		Action:     "stream",
		URL:        srv.URL,
		StreamIdle: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	buf := make([]byte, 16)
	_, err = resp.Body.Read(buf)
	if err == nil {
		t.Fatal("expected stream idle timeout")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) || providerErr.Action != "stream.stream_read" {
		t.Fatalf("expected stream read ProviderError, got %T %[1]v", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded cause, got %v", err)
	}
}

func TestDoNetworkPolicyBlocksPrivateHost(t *testing.T) {
	_, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      "http://127.0.0.1:1",
		NetworkPolicy: &NetworkPolicy{
			AllowHTTP: true,
		},
	})
	if err == nil {
		t.Fatal("expected network policy error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if !strings.Contains(providerErr.Cause.Error(), "private/internal") {
		t.Fatalf("unexpected network policy error: %v", providerErr.Cause)
	}
}

func TestDoNetworkPolicyGuardsActualDialTarget(t *testing.T) {
	var reached atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reached.Store(true)
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, network, srv.Listener.Addr().String())
			},
		},
	}

	resp, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      "http://1.1.1.1/",
		Client:   client,
		NetworkPolicy: &NetworkPolicy{
			AllowHTTP: true,
		},
	})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err == nil {
		t.Fatal("expected network policy to block private/internal actual dial target")
	}
	if reached.Load() {
		t.Fatal("network policy preflight passed but request reached local server")
	}
}

func TestValidateURLAllowedHostDoesNotBypassPrivateIP(t *testing.T) {
	err := ValidateURL(context.Background(), "http://169.254.169.254/latest/meta-data", &NetworkPolicy{
		AllowHTTP:    true,
		AllowedHosts: []string{"169.254.169.254"},
	})
	if err == nil {
		t.Fatal("allowed host must not bypass private/metadata IP checks")
	}
	if !errors.Is(err, ErrNetworkPolicy) || !strings.Contains(err.Error(), "private/internal") {
		t.Fatalf("expected private/internal network policy error, got %v", err)
	}
}

func TestValidateURLBlocksSpecialPurposeAddresses(t *testing.T) {
	cases := []struct {
		name string
		url  string
	}{
		{name: "ipv4 unspecified", url: "http://0.0.0.0/v1"},
		{name: "ipv6 unspecified", url: "http://[::]/v1"},
		{name: "ipv4 multicast", url: "http://239.255.255.250/v1"},
		{name: "ipv6 multicast", url: "http://[ff05::1]/v1"},
		{name: "carrier grade nat", url: "http://100.64.0.1/v1"},
		{name: "ipv4 limited broadcast", url: "http://255.255.255.255/v1"},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateURL(context.Background(), tt.url, &NetworkPolicy{AllowHTTP: true})
			if err == nil {
				t.Fatal("expected network policy error")
			}
			if !errors.Is(err, ErrNetworkPolicy) || !strings.Contains(err.Error(), "private/internal") {
				t.Fatalf("expected private/internal network policy error, got %v", err)
			}
		})
	}
}

// stubRemoteAddrConn 伪造 RemoteAddr，用于验证守卫拨号对实际连入地址的拦截。
// 拦截发生在拨号校验阶段，连接上除 RemoteAddr/Close 外不会有任何读写。
type stubRemoteAddrConn struct {
	net.Conn
	remote net.Addr
}

func (c stubRemoteAddrConn) RemoteAddr() net.Addr { return c.remote }

func (c stubRemoteAddrConn) Close() error {
	if c.Conn != nil {
		return c.Conn.Close()
	}
	return nil
}

// 原 upstream 包直测其私有 validateDialTarget 复制品；迁移后改测公开行为：
// Do + NetworkPolicy 组合的守卫拨号必须拦截连到特殊用途地址的实际连接。
func TestDoNetworkPolicyBlocksSpecialPurposeRemoteAddresses(t *testing.T) {
	cases := []struct {
		name string
		ip   string
	}{
		{name: "ipv4 unspecified", ip: "0.0.0.0"},
		{name: "ipv6 unspecified", ip: "::"},
		{name: "ipv4 multicast", ip: "239.255.255.250"},
		{name: "ipv6 multicast", ip: "ff05::1"},
		{name: "carrier grade nat", ip: "100.64.0.1"},
		{name: "ipv4 limited broadcast", ip: "255.255.255.255"},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			client := &http.Client{
				Transport: &http.Transport{
					DialContext: func(context.Context, string, string) (net.Conn, error) {
						return stubRemoteAddrConn{remote: &net.TCPAddr{IP: net.ParseIP(tt.ip), Port: 443}}, nil
					},
				},
			}
			_, err := Do(context.Background(), Request{
				Provider:      "test-provider",
				Action:        "complete",
				URL:           "http://1.1.1.1/",
				Client:        client,
				NetworkPolicy: &NetworkPolicy{AllowHTTP: true},
			})
			if err == nil {
				t.Fatal("expected network policy error")
			}
			if !errors.Is(err, ErrNetworkPolicy) || !strings.Contains(err.Error(), "private/internal") {
				t.Fatalf("expected private/internal network policy error, got %v", err)
			}
		})
	}
}

func TestValidateURLAllowedHostsRestrictsHost(t *testing.T) {
	err := ValidateURL(context.Background(), "https://example.org/v1", &NetworkPolicy{
		AllowedHosts: []string{"api.openai.com"},
	})
	if err == nil {
		t.Fatal("expected host allowlist to reject unlisted host")
	}
	if !errors.Is(err, ErrNetworkPolicy) || !strings.Contains(err.Error(), "not allowed") {
		t.Fatalf("expected host allowlist network policy error, got %v", err)
	}
}

func TestDoNetworkPolicyAllowedHostDoesNotBypassActualPrivateDialTarget(t *testing.T) {
	var reached atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reached.Store(true)
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, network, srv.Listener.Addr().String())
			},
		},
	}

	resp, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      "http://1.1.1.1/",
		Client:   client,
		NetworkPolicy: &NetworkPolicy{
			AllowHTTP:    true,
			AllowedHosts: []string{"1.1.1.1"},
		},
	})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err == nil {
		t.Fatal("allowed host must not bypass private/internal actual dial target")
	}
	if reached.Load() {
		t.Fatal("request reached local server through allowed-host bypass")
	}
}

func TestDoNetworkPolicyRequiresHTTPSUnlessAllowed(t *testing.T) {
	_, err := Do(context.Background(), Request{
		Provider:      "test-provider",
		Action:        "complete",
		URL:           "http://example.com",
		NetworkPolicy: &NetworkPolicy{},
	})
	if err == nil {
		t.Fatal("expected network policy error")
	}
	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("expected ProviderError, got %T %[1]v", err)
	}
	if !strings.Contains(providerErr.Cause.Error(), `scheme "http" is not allowed`) {
		t.Fatalf("unexpected network policy error: %v", providerErr.Cause)
	}
}

func TestDoNetworkPolicyAllowsExplicitHost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	parsed, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := Do(context.Background(), Request{
		Provider: "test-provider",
		Action:   "complete",
		URL:      srv.URL,
		NetworkPolicy: &NetworkPolicy{
			AllowHTTP:    true,
			AllowPrivate: true,
			AllowedHosts: []string{parsed.Hostname()},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()
}
