// Package transport provides the shared production-grade upstream HTTP layer
// used by all ai-core adapters.
package transport

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	DefaultErrorBodyLimit = 64 * 1024
	DefaultPreviewLimit   = 2 * 1024
)

// Request describes one upstream HTTP call.
type Request struct {
	Provider string
	Action   string
	Method   string
	URL      string
	Body     []byte

	Client         *http.Client
	Transport      *Transport
	SetHeaders     func(*http.Request)
	Headers        map[string]string
	NetworkPolicy  *NetworkPolicy
	ExpectedCode   int
	Timeout        time.Duration
	StreamIdle     time.Duration
	Retry          RetryPolicy
	ErrorBodyLimit int64
	PreviewLimit   int64
}

// RetryPolicy controls transport-level retries for transient upstream failures.
//
// MaxAttempts is the total attempts count, including the first attempt. A zero
// value uses DefaultRetryPolicy; set MaxAttempts to 1 to disable retries.
//
// MaxDelay caps the computed exponential backoff only. A server-provided
// Retry-After is capped separately by MaxServerDelay so an explicit upstream
// hint (e.g. 429 with Retry-After: 30) is not flattened into a delay that is
// guaranteed to hit the same rate limit again.
type RetryPolicy struct {
	MaxAttempts          int
	BaseDelay            time.Duration
	MaxDelay             time.Duration
	MaxServerDelay       time.Duration
	Multiplier           float64
	RetryStatuses        []int
	DisableNetworkErrors bool
}

// DefaultRetryPolicy returns the conservative default used by Do.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxAttempts:    3,
		BaseDelay:      100 * time.Millisecond,
		MaxDelay:       2 * time.Second,
		MaxServerDelay: 30 * time.Second,
		Multiplier:     2,
	}
}

// Do sends an upstream request with diagnostics, bounded error bodies, safe
// custom headers, optional request timeout, stream idle timeout, and SSRF
// network policy enforcement.
func Do(ctx context.Context, cfg Request) (*http.Response, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	var cancel context.CancelFunc
	if cfg.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, cfg.Timeout)
	}
	cancelRequest := func() {
		if cancel != nil {
			cancel()
			cancel = nil
		}
	}

	method := cfg.Method
	if method == "" {
		method = http.MethodPost
	}
	cfg.Method = method

	bodyPreview, bodyPreviewTruncated := previewBytes(cfg.Body, limitOrDefault(cfg.PreviewLimit, DefaultPreviewLimit))
	policy := cfg.NetworkPolicy
	if cfg.Transport != nil {
		policy = cfg.Transport.Policy()
	}
	if err := ValidateURL(ctx, cfg.URL, policy); err != nil {
		cancelRequest()
		return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, nil, 0, err)
	}

	client := cfg.Client
	if cfg.Transport != nil {
		var err error
		client, err = cfg.Transport.Client()
		if err != nil {
			cancelRequest()
			return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, nil, 0, err)
		}
	} else if cfg.NetworkPolicy != nil {
		var err error
		client, err = ClientWithNetworkPolicy(client, cfg.NetworkPolicy)
		if err != nil {
			cancelRequest()
			return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, nil, 0, err)
		}
	}
	if client == nil {
		client = http.DefaultClient
	}

	expectedCode := cfg.ExpectedCode
	if expectedCode == 0 {
		expectedCode = http.StatusOK
	}
	retryPolicy := normalizeRetryPolicy(cfg.Retry)
	if OperationSafetyFromContext(ctx) == OperationSafetyNonIdempotent {
		retryPolicy.MaxAttempts = 1
	}

	for attempt := 1; ; attempt++ {
		httpReq, blockedHeaders, err := newHTTPRequest(ctx, cfg, method)
		if err != nil {
			cancelRequest()
			return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, blockedHeaders, 0, err)
		}

		startedAt := time.Now()
		resp, err := client.Do(httpReq)
		elapsed := time.Since(startedAt)
		if err != nil {
			if attempt < retryPolicy.MaxAttempts && shouldRetryNetworkError(ctx, err, retryPolicy) {
				if waitErr := waitForRetry(ctx, retryDelay(retryPolicy, attempt, 0)); waitErr != nil {
					cancelRequest()
					return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, blockedHeaders, elapsed, waitErr)
				}
				continue
			}
			cancelRequest()
			return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, blockedHeaders, elapsed, err)
		}
		if resp == nil {
			err := errors.New("nil http response")
			if attempt < retryPolicy.MaxAttempts && !retryPolicy.DisableNetworkErrors {
				if waitErr := waitForRetry(ctx, retryDelay(retryPolicy, attempt, 0)); waitErr != nil {
					cancelRequest()
					return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, blockedHeaders, elapsed, waitErr)
				}
				continue
			}
			cancelRequest()
			return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, blockedHeaders, elapsed, err)
		}

		if resp.StatusCode != expectedCode {
			body, truncated, readErr := readLimited(resp.Body, limitOrDefault(cfg.ErrorBodyLimit, DefaultErrorBodyLimit))
			_ = resp.Body.Close()
			providerErr := statusError(cfg, method, bodyPreview, bodyPreviewTruncated, blockedHeaders, elapsed, resp, body, truncated, readErr)
			if attempt < retryPolicy.MaxAttempts && shouldRetryStatus(resp.StatusCode, retryPolicy) {
				if waitErr := waitForRetry(ctx, retryDelay(retryPolicy, attempt, providerErr.RetryAfter)); waitErr != nil {
					cancelRequest()
					return nil, requestError(cfg, bodyPreview, bodyPreviewTruncated, blockedHeaders, elapsed, waitErr)
				}
				continue
			}
			cancelRequest()
			return nil, providerErr
		}

		if cfg.StreamIdle > 0 && resp.Body != nil {
			resp.Body = NewIdleTimeoutReadCloser(resp.Body, cfg.StreamIdle, cfg.Provider, cfg.Action)
		}
		if cancel != nil {
			if resp.Body != nil {
				resp.Body = NewCancelOnCloseReadCloser(resp.Body, cancel)
				cancel = nil
			} else {
				cancelRequest()
			}
		}

		return resp, nil
	}
}

func newHTTPRequest(ctx context.Context, cfg Request, method string) (*http.Request, []string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, method, cfg.URL, bytes.NewReader(cfg.Body))
	if err != nil {
		return nil, nil, err
	}
	if cfg.Body != nil {
		httpReq.ContentLength = int64(len(cfg.Body))
		httpReq.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(cfg.Body)), nil
		}
	}
	if cfg.SetHeaders != nil {
		cfg.SetHeaders(httpReq)
	}
	blockedHeaders := ApplySafeHeaders(httpReq, cfg.Headers)
	return httpReq, blockedHeaders, nil
}

func statusError(cfg Request, method string, bodyPreview string, bodyPreviewTruncated bool, blockedHeaders []string, elapsed time.Duration, resp *http.Response, body []byte, bodyTruncated bool, readErr error) *ProviderError {
	err := &ProviderError{
		Provider:                cfg.Provider,
		Action:                  cfg.Action,
		Method:                  method,
		URL:                     sanitizeDiagnosticURL(cfg.URL),
		StatusCode:              resp.StatusCode,
		Status:                  resp.Status,
		RequestPreview:          bodyPreview,
		RequestPreviewTruncated: bodyPreviewTruncated,
		RequestHeadersBlocked:   blockedHeaders,
		ResponseHeaders:         diagnosticHeaders(resp.Header),
		RequestID:               requestID(resp.Header),
		RetryAfter:              retryAfter(resp.Header, time.Now()),
		Elapsed:                 elapsed,
	}
	if readErr != nil {
		err.Body = "failed to read body: " + readErr.Error()
		err.Cause = readErr
		return err
	}
	err.Body = string(body)
	err.BodyTruncated = bodyTruncated
	return err
}

func requestError(cfg Request, preview string, previewTruncated bool, blockedHeaders []string, elapsed time.Duration, err error) error {
	return &ProviderError{
		Provider:                cfg.Provider,
		Action:                  cfg.Action,
		Method:                  cfg.Method,
		URL:                     sanitizeDiagnosticURL(cfg.URL),
		RequestPreview:          preview,
		RequestPreviewTruncated: previewTruncated,
		RequestHeadersBlocked:   blockedHeaders,
		Elapsed:                 elapsed,
		Cause:                   err,
	}
}

func normalizeRetryPolicy(policy RetryPolicy) RetryPolicy {
	if policy.MaxAttempts == 0 {
		policy = DefaultRetryPolicy()
	}
	if policy.MaxAttempts < 1 {
		policy.MaxAttempts = 1
	}
	if policy.BaseDelay <= 0 {
		policy.BaseDelay = 100 * time.Millisecond
	}
	if policy.MaxDelay <= 0 {
		policy.MaxDelay = 2 * time.Second
	}
	if policy.MaxServerDelay <= 0 {
		policy.MaxServerDelay = 30 * time.Second
	}
	if policy.Multiplier <= 0 {
		policy.Multiplier = 2
	}
	return policy
}

func shouldRetryStatus(status int, policy RetryPolicy) bool {
	if len(policy.RetryStatuses) > 0 {
		for _, code := range policy.RetryStatuses {
			if status == code {
				return true
			}
		}
		return false
	}
	return status == http.StatusRequestTimeout ||
		status == http.StatusConflict ||
		status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError
}

func shouldRetryNetworkError(ctx context.Context, err error, policy RetryPolicy) bool {
	if policy.DisableNetworkErrors || err == nil {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrNetworkPolicy) {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return netErr.Timeout() || netErr.Temporary()
	}
	return true
}

func retryDelay(policy RetryPolicy, attempt int, retryAfter time.Duration) time.Duration {
	delay := policy.BaseDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(float64(delay) * policy.Multiplier)
		if delay >= policy.MaxDelay {
			delay = policy.MaxDelay
			break
		}
	}
	if delay > policy.MaxDelay {
		delay = policy.MaxDelay
	}
	// A server-provided Retry-After is an explicit upstream hint: cap it by
	// MaxServerDelay instead of the (much smaller) backoff MaxDelay.
	if retryAfter > policy.MaxServerDelay {
		retryAfter = policy.MaxServerDelay
	}
	if retryAfter > delay {
		delay = retryAfter
	}
	return delay
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func sanitizeDiagnosticURL(rawURL string) string {
	if rawURL == "" {
		return ""
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		if idx := strings.Index(rawURL, "?"); idx >= 0 {
			return rawURL[:idx] + "?[redacted]"
		}
		return rawURL
	}
	if parsed.User != nil {
		parsed.User = url.User("[redacted]")
	}
	parsed.Fragment = ""
	values := parsed.Query()
	for key := range values {
		if isSensitiveQueryKey(key) {
			values[key] = []string{"[redacted]"}
		}
	}
	parsed.RawQuery = values.Encode()
	return parsed.String()
}

func isSensitiveQueryKey(key string) bool {
	normalized := strings.ToLower(strings.NewReplacer("-", "", "_", "", ".", "").Replace(strings.TrimSpace(key)))
	for _, marker := range []string{"key", "secret", "token", "password", "passwd", "pwd", "signature", "credential", "authorization", "auth"} {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	return false
}

func readLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	if r == nil {
		return nil, false, nil
	}
	if limit <= 0 {
		limit = DefaultErrorBodyLimit
	}
	data, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(data)) > limit {
		return data[:limit], true, nil
	}
	return data, false, nil
}

func previewBytes(data []byte, limit int64) (string, bool) {
	if limit <= 0 {
		limit = DefaultPreviewLimit
	}
	if int64(len(data)) > limit {
		return string(data[:limit]), true
	}
	return string(data), false
}

func limitOrDefault(value, fallback int64) int64 {
	if value > 0 {
		return value
	}
	return fallback
}

func diagnosticHeaders(headers http.Header) map[string]string {
	keys := []string{
		"x-request-id",
		"request-id",
		"openai-request-id",
		"anthropic-request-id",
		"cf-ray",
		"x-ratelimit-limit-requests",
		"x-ratelimit-remaining-requests",
		"x-ratelimit-reset-requests",
		"x-ratelimit-limit-tokens",
		"x-ratelimit-remaining-tokens",
		"x-ratelimit-reset-tokens",
		"retry-after",
		"retry-after-ms",
		"x-should-retry",
		"x-openrouter-provider",
		"x-volc-request-id",
		"x-tt-logid",
	}
	out := make(map[string]string)
	for _, key := range keys {
		if value := headers.Get(key); value != "" {
			out[key] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func requestID(headers http.Header) string {
	for _, key := range []string{"x-request-id", "request-id", "openai-request-id", "anthropic-request-id", "x-volc-request-id", "x-tt-logid"} {
		if value := headers.Get(key); value != "" {
			return value
		}
	}
	return ""
}

func retryAfter(headers http.Header, now time.Time) time.Duration {
	if value := headers.Get("retry-after-ms"); value != "" {
		ms, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
		if err == nil && ms > 0 {
			return time.Duration(ms) * time.Millisecond
		}
	}
	if value := headers.Get("retry-after"); value != "" {
		value = strings.TrimSpace(value)
		if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds > 0 {
			return time.Duration(seconds) * time.Second
		}
		if at, err := http.ParseTime(value); err == nil && at.After(now) {
			return at.Sub(now)
		}
	}
	return 0
}

// Transport owns the guarded HTTP client for a network policy. It is
// intentionally reusable because http.Transport keeps the connection pool.
type Transport struct {
	baseClient *http.Client
	policy     *NetworkPolicy

	once   sync.Once
	client *http.Client
	err    error
}

// NewTransport constructs a reusable guarded transport.
func NewTransport(client *http.Client, policy *NetworkPolicy) *Transport {
	return &Transport{
		baseClient: client,
		policy:     CloneNetworkPolicy(policy),
	}
}

// Policy returns a cloned network policy pointer owned by the transport.
func (t *Transport) Policy() *NetworkPolicy {
	if t == nil {
		return nil
	}
	return t.policy
}

// Client returns a guarded HTTP client.
func (t *Transport) Client() (*http.Client, error) {
	if t == nil {
		return http.DefaultClient, nil
	}
	t.once.Do(func() {
		if t.policy == nil {
			t.client = t.baseClient
			if t.client == nil {
				t.client = http.DefaultClient
			}
			return
		}
		t.client, t.err = ClientWithNetworkPolicy(t.baseClient, t.policy)
	})
	return t.client, t.err
}

// ValidateURL validates the request URL before an upstream call.
func ValidateURL(ctx context.Context, rawURL string, policy *NetworkPolicy) error {
	if policy == nil {
		return nil
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return wrapNetworkPolicy("parse url: %v", err)
	}
	host := parsed.Hostname()
	if host == "" {
		return wrapNetworkPolicy("missing host")
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "https" && !(scheme == "http" && policy.AllowHTTP) {
		return wrapNetworkPolicy("scheme %q is not allowed", parsed.Scheme)
	}
	if len(policy.AllowedHosts) > 0 && !hostAllowed(host, policy.AllowedHosts) {
		return wrapNetworkPolicy("host %q is not allowed", host)
	}
	if policy.AllowPrivate {
		return nil
	}

	if ip := net.ParseIP(host); ip != nil {
		if isPrivateOrMetadataIP(ip) {
			return wrapNetworkPolicy("host %q resolves to private/internal IP", host)
		}
		return nil
	}

	timeout := policy.DNSLookupTimeout
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	lookupCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	addrs, err := net.DefaultResolver.LookupIPAddr(lookupCtx, host)
	if err != nil {
		return wrapNetworkPolicy("resolve host %q: %v", host, err)
	}
	if len(addrs) == 0 {
		return wrapNetworkPolicy("host %q has no IP addresses", host)
	}
	for _, addr := range addrs {
		if isPrivateOrMetadataIP(addr.IP) {
			return wrapNetworkPolicy("host %q resolves to private/internal IP", host)
		}
	}
	return nil
}

// ClientWithNetworkPolicy clones client with a guarded transport that verifies
// the actual dial target, closing DNS rebinding gaps.
func ClientWithNetworkPolicy(client *http.Client, policy *NetworkPolicy) (*http.Client, error) {
	if policy == nil {
		return client, nil
	}
	if client == nil {
		client = http.DefaultClient
	}

	rt := client.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	base, ok := rt.(*http.Transport)
	if !ok {
		return nil, wrapNetworkPolicy("unsupported transport %T", rt)
	}

	guarded := base.Clone()
	dialContext := guarded.DialContext
	if dialContext == nil {
		dialer := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}
		dialContext = dialer.DialContext
	}
	guarded.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		conn, err := dialContext(ctx, network, addr)
		if err != nil {
			return nil, err
		}
		if err := validateDialTarget(addr, conn.RemoteAddr(), policy); err != nil {
			_ = conn.Close()
			return nil, err
		}
		return conn, nil
	}

	if guarded.DialTLSContext != nil {
		dialTLSContext := guarded.DialTLSContext
		guarded.DialTLSContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := dialTLSContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			if err := validateDialTarget(addr, conn.RemoteAddr(), policy); err != nil {
				_ = conn.Close()
				return nil, err
			}
			return conn, nil
		}
	}
	if guarded.DialTLS != nil {
		dialTLS := guarded.DialTLS
		guarded.DialTLS = func(network, addr string) (net.Conn, error) {
			conn, err := dialTLS(network, addr)
			if err != nil {
				return nil, err
			}
			if err := validateDialTarget(addr, conn.RemoteAddr(), policy); err != nil {
				_ = conn.Close()
				return nil, err
			}
			return conn, nil
		}
	}

	cloned := *client
	cloned.Transport = guarded
	return &cloned, nil
}

func validateDialTarget(addr string, remote net.Addr, policy *NetworkPolicy) error {
	if policy == nil || policy.AllowPrivate {
		return nil
	}

	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}

	if tcpAddr, ok := remote.(*net.TCPAddr); ok && isPrivateOrMetadataIP(tcpAddr.IP) {
		return wrapNetworkPolicy("dial target %q connected to private/internal IP %s", host, tcpAddr.IP.String())
	}

	remoteHost, _, err := net.SplitHostPort(remote.String())
	if err != nil {
		remoteHost = remote.String()
	}
	if ip := net.ParseIP(remoteHost); isPrivateOrMetadataIP(ip) {
		return wrapNetworkPolicy("dial target %q connected to private/internal IP %s", host, ip.String())
	}
	return nil
}

func hostAllowed(host string, allowedHosts []string) bool {
	if len(allowedHosts) == 0 {
		return false
	}
	host = strings.ToLower(strings.TrimSuffix(strings.TrimSpace(host), "."))
	for _, allowed := range allowedHosts {
		allowed = strings.ToLower(strings.TrimSuffix(strings.TrimSpace(allowed), "."))
		if allowed == "" {
			continue
		}
		if allowed == host {
			return true
		}
		if strings.HasPrefix(allowed, "*.") && strings.HasSuffix(host, strings.TrimPrefix(allowed, "*")) {
			return true
		}
	}
	return false
}

// CloneNetworkPolicy returns a defensive copy of policy (nil-safe), so a
// stored policy cannot be mutated through the caller's pointer afterwards.
// It is the single shared implementation for all adapters.
func CloneNetworkPolicy(policy *NetworkPolicy) *NetworkPolicy {
	if policy == nil {
		return nil
	}
	cloned := *policy
	cloned.AllowedHosts = append([]string(nil), policy.AllowedHosts...)
	return &cloned
}

// CloneHeaders returns a defensive copy of headers; empty input yields nil.
// It is the single shared implementation for all adapters.
func CloneHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(headers))
	for key, value := range headers {
		cloned[key] = value
	}
	return cloned
}

func wrapNetworkPolicy(format string, args ...any) error {
	return &networkPolicyError{message: fmt.Sprintf(format, args...)}
}

type networkPolicyError struct {
	message string
}

func (e *networkPolicyError) Error() string {
	return ErrNetworkPolicy.Error() + ": " + e.message
}

func (e *networkPolicyError) Unwrap() error {
	return ErrNetworkPolicy
}
