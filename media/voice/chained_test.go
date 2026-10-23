package voice

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/hexagon-codes/ai-core/transport"
)

func TestChainedTTS_NonIdempotentUnknownDoesNotFallback(t *testing.T) {
	first := &fakeTTS{name: "first", err: errors.New("response lost after submit")}
	second := &fakeTTS{name: "second"}
	ctx := transport.WithOperationSafety(context.Background(), transport.OperationSafetyNonIdempotent)
	_, err := NewChainedTTS(first, second).Synthesize(ctx, "hello", SynthesizeOptions{})
	if err == nil || first.calls != 1 || second.calls != 0 {
		t.Fatalf("unknown generation replayed: err=%v first=%d second=%d", err, first.calls, second.calls)
	}
}

func TestChainedTTS_CancelledDoesNotFallback(t *testing.T) {
	first := &fakeTTS{name: "first", err: context.Canceled}
	second := &fakeTTS{name: "second"}
	_, err := NewChainedTTS(first, second).Synthesize(context.Background(), "hello", SynthesizeOptions{})
	if !errors.Is(err, context.Canceled) || second.calls != 0 {
		t.Fatalf("cancellation replayed: err=%v second=%d", err, second.calls)
	}
}

func TestChainedTTS_NonIdempotentDefiniteRejectionAllowsFallback(t *testing.T) {
	first := &fakeTTS{name: "first", err: &transport.ProviderError{StatusCode: 429}}
	second := &fakeTTS{name: "second"}
	ctx := transport.WithOperationSafety(context.Background(), transport.OperationSafetyNonIdempotent)
	res, err := NewChainedTTS(first, second).Synthesize(ctx, "hello", SynthesizeOptions{})
	if err != nil || string(res.Audio) != "second" || first.calls != 1 || second.calls != 1 {
		t.Fatalf("definite rejection fallback lost: result=%+v err=%v", res, err)
	}
}

func TestChainedTTS_NestedPreservesSuccessfulProvider(t *testing.T) {
	first := &fakeTTS{name: "rejected", err: &transport.ProviderError{StatusCode: 429}}
	winner := &fakeTTS{name: "winner"}
	unused := &fakeTTS{name: "unused"}
	ctx := transport.WithOperationSafety(context.Background(), transport.OperationSafetyNonIdempotent)
	res, err := NewChainedTTS(NewChainedTTS(first, winner), unused).Synthesize(ctx, "hello", SynthesizeOptions{})
	if err != nil || res.Provider != "winner" || unused.calls != 0 {
		t.Fatalf("nested chain lost successful provider: result=%+v err=%v unused=%d", res, err, unused.calls)
	}
}

func TestChainedTTS_AggregatesPreserveFailureSemantics(t *testing.T) {
	for _, tc := range []struct {
		name     string
		second   error
		definite bool
	}{
		{name: "all rejected", second: &transport.ProviderError{StatusCode: 400}, definite: true},
		{name: "unknown response", second: errors.New("response lost")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			firstErr := &transport.ProviderError{StatusCode: 429}
			first := &fakeTTS{name: "first", err: firstErr}
			second := &fakeTTS{name: "second", err: tc.second}
			_, err := NewChainedTTS(first, second).Synthesize(context.Background(), "hello", SynthesizeOptions{})
			var failure interface{ DefinitiveFailure() bool }
			if !errors.As(err, &failure) || failure.DefinitiveFailure() != tc.definite || IsDefinitiveSynthesisFailure(err) != tc.definite {
				t.Fatalf("aggregate classification changed: err=%v want definitive=%v", err, tc.definite)
			}
			if !errors.Is(err, firstErr) || !errors.Is(err, tc.second) || first.calls != 1 || second.calls != 1 {
				t.Fatalf("aggregate lost causes or unmarked fallback: err=%v calls=%d,%d", err, first.calls, second.calls)
			}
		})
	}
}

func TestChainedTTS_PreflightFailureAllowsFallback(t *testing.T) {
	second := &fakeTTS{name: "second"}
	ctx := transport.WithOperationSafety(context.Background(), transport.OperationSafetyNonIdempotent)
	res, err := NewChainedTTS(NewMiniMaxTTS("", ""), second).Synthesize(ctx, "hello", SynthesizeOptions{})
	if err != nil || res.Provider != "second" || second.calls != 1 {
		t.Fatalf("preflight rejection prevented fallback: result=%+v err=%v calls=%d", res, err, second.calls)
	}
}

func TestChainedTTS_MiniMaxBusinessRejectionSemantics(t *testing.T) {
	for _, tc := range []struct {
		name     string
		code     int
		fallback bool
	}{
		{name: "explicit rejection", code: 1004, fallback: true},
		{name: "unclassified failure", code: 1000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				_, _ = fmt.Fprintf(w, `{"base_resp":{"status_code":%d,"status_msg":"request failed"}}`, tc.code)
			}))
			defer server.Close()
			first := NewMiniMaxTTS("fixture-key", "fixture-group", WithMiniMaxBaseURL(server.URL), WithMiniMaxHTTPClient(server.Client()))
			second := &fakeTTS{name: "second"}
			ctx := transport.WithOperationSafety(context.Background(), transport.OperationSafetyNonIdempotent)
			res, err := NewChainedTTS(first, second).Synthesize(ctx, "hello", SynthesizeOptions{})
			if calls.Load() != 1 {
				t.Fatalf("submission replayed: calls=%d", calls.Load())
			}
			if tc.fallback {
				if err != nil || res.Provider != "second" || second.calls != 1 {
					t.Fatalf("explicit rejection prevented fallback: result=%+v err=%v calls=%d", res, err, second.calls)
				}
			} else if err == nil || second.calls != 0 || IsDefinitiveSynthesisFailure(err) {
				t.Fatalf("unclassified error replayed: err=%v calls=%d", err, second.calls)
			}
		})
	}
}

func TestChainedTTS_EdgeHandshakeRejectionAllowsFallback(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusForbidden)
	}))
	defer server.Close()
	first := NewEdgeTTS(EdgeTTSWithEndpoint(wsURL(server.URL)), EdgeTTSWithHTTPClient(server.Client()))
	ctx := transport.WithOperationSafety(context.Background(), transport.OperationSafetyNonIdempotent)
	_, err := first.Synthesize(ctx, "hello", SynthesizeOptions{})
	var upstream *transport.ProviderError
	if !IsDefinitiveSynthesisFailure(err) || !errors.As(err, &upstream) || upstream.StatusCode != http.StatusForbidden {
		t.Fatalf("handshake rejection lost its status: %v", err)
	}
	second := &fakeTTS{name: "second"}
	res, err := NewChainedTTS(first, second).Synthesize(ctx, "hello", SynthesizeOptions{})
	if err != nil || res.Provider != "second" || second.calls != 1 || calls.Load() != 2 {
		t.Fatalf("handshake rejection prevented fallback: result=%+v err=%v edge=%d second=%d", res, err, calls.Load(), second.calls)
	}
}

// fakeTTS 是一个可控成功/失败的 TTSProvider 桩，用于验证 ChainedTTS 的
// fallback 语义（下沉到 ai-core 后已去掉 feature flag，链路无条件生效）。
type fakeTTS struct {
	name    string
	err     error
	calls   int
	formats []AudioFormat
	voices  []VoiceInfo
}

func (f *fakeTTS) Name() string { return f.name }
func (f *fakeTTS) Synthesize(_ context.Context, _ string, _ SynthesizeOptions) (*SynthesizeResult, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return &SynthesizeResult{Audio: []byte(f.name), Format: "mp3"}, nil
}
func (f *fakeTTS) Voices() []VoiceInfo             { return f.voices }
func (f *fakeTTS) SupportedFormats() []AudioFormat { return f.formats }

func TestChainedTTS_NilProvidersSkipped(t *testing.T) {
	p := &fakeTTS{name: "p"}
	c := NewChainedTTS(nil, p, nil)
	if got := len(c.providers); got != 1 {
		t.Fatalf("expected 1 provider after nil-skip, got %d", got)
	}
}

func TestChainedTTS_FirstSucceedsShortCircuits(t *testing.T) {
	first := &fakeTTS{name: "first"}
	second := &fakeTTS{name: "second"}
	c := NewChainedTTS(first, second)

	res, err := c.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(res.Audio) != "first" {
		t.Errorf("expected first provider result, got %q", res.Audio)
	}
	if second.calls != 0 {
		t.Errorf("second provider should not be called when first succeeds, calls=%d", second.calls)
	}
}

func TestChainedTTS_FallsBackToNext(t *testing.T) {
	first := &fakeTTS{name: "first", err: errors.New("boom")}
	second := &fakeTTS{name: "second"}
	c := NewChainedTTS(first, second)

	res, err := c.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(res.Audio) != "second" {
		t.Errorf("expected fallback to second provider, got %q", res.Audio)
	}
	if first.calls != 1 || second.calls != 1 {
		t.Errorf("expected both tried once, got first=%d second=%d", first.calls, second.calls)
	}
}

func TestChainedTTS_AllFailAggregatesErrors(t *testing.T) {
	c := NewChainedTTS(
		&fakeTTS{name: "a", err: errors.New("err-a")},
		&fakeTTS{name: "b", err: errors.New("err-b")},
	)
	_, err := c.Synthesize(context.Background(), "hi", SynthesizeOptions{})
	if err == nil {
		t.Fatal("expected error when all providers fail")
	}
	if !strings.Contains(err.Error(), "err-a") || !strings.Contains(err.Error(), "err-b") {
		t.Errorf("aggregated error should mention both failures, got %v", err)
	}
}

func TestChainedTTS_EmptyErrors(t *testing.T) {
	c := NewChainedTTS()
	if _, err := c.Synthesize(context.Background(), "hi", SynthesizeOptions{}); err == nil {
		t.Error("expected error for empty chain")
	}
}

func TestChainedTTS_FormatsUnionAndVoicesDedup(t *testing.T) {
	c := NewChainedTTS(
		&fakeTTS{name: "a", formats: []AudioFormat{"mp3"}, voices: []VoiceInfo{{ID: "v1"}}},
		&fakeTTS{name: "b", formats: []AudioFormat{"mp3", "wav"}, voices: []VoiceInfo{{ID: "v1"}, {ID: "v2"}}},
	)
	if got := len(c.SupportedFormats()); got != 2 {
		t.Errorf("expected union of 2 formats, got %d", got)
	}
	if got := len(c.Voices()); got != 2 {
		t.Errorf("expected 2 deduped voices, got %d", got)
	}
}
