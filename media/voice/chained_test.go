package voice

import (
	"context"
	"errors"
	"strings"
	"testing"
)

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
