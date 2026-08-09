package voice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
	"github.com/hexagon-codes/ai-core/transport"
	"github.com/hexagon-codes/toolkit/net/httpx"
)

const defaultElevenLabsBase = "https://api.elevenlabs.io/v1"

type ElevenLabsTTS struct {
	apiKey       string
	baseURL      string
	model        string
	defaultVoice string
	client       *http.Client
	timeout      time.Duration
	headers      map[string]string
	policy       *llm.NetworkPolicy
	transport    *transport.Transport
}

type ElevenLabsOption func(*ElevenLabsTTS)

func NewElevenLabsTTS(apiKey string, opts ...ElevenLabsOption) *ElevenLabsTTS {
	t := &ElevenLabsTTS{
		apiKey:       apiKey,
		baseURL:      defaultElevenLabsBase,
		model:        "eleven_multilingual_v2",
		defaultVoice: "Rachel",
		client:       httpx.MustNewRawClient(httpx.WithResponseHeaderTimeout(120 * time.Second)),
		timeout:      60 * time.Second,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(t)
		}
	}
	t.baseURL = strings.TrimRight(t.baseURL, "/")
	t.transport = transport.NewTransport(t.client, t.policy)
	return t
}

func ElevenLabsWithBaseURL(baseURL string) ElevenLabsOption {
	return func(t *ElevenLabsTTS) {
		if strings.TrimSpace(baseURL) != "" {
			t.baseURL = baseURL
		}
	}
}

func ElevenLabsWithModel(model string) ElevenLabsOption {
	return func(t *ElevenLabsTTS) {
		if strings.TrimSpace(model) != "" {
			t.model = model
		}
	}
}

func ElevenLabsWithVoice(voice string) ElevenLabsOption {
	return func(t *ElevenLabsTTS) {
		if strings.TrimSpace(voice) != "" {
			t.defaultVoice = voice
		}
	}
}

func ElevenLabsWithHTTPClient(client *http.Client) ElevenLabsOption {
	return func(t *ElevenLabsTTS) {
		if client != nil {
			t.client = client
		}
	}
}

func ElevenLabsWithNetworkPolicy(policy llm.NetworkPolicy) ElevenLabsOption {
	return func(t *ElevenLabsTTS) {
		t.policy = cloneVoiceNetworkPolicy(&policy)
	}
}

func ElevenLabsWithHeaders(headers map[string]string) ElevenLabsOption {
	return func(t *ElevenLabsTTS) {
		t.headers = cloneVoiceHeaders(headers)
	}
}

func (t *ElevenLabsTTS) Name() string { return "elevenlabs-tts" }

func (t *ElevenLabsTTS) SupportedFormats() []AudioFormat {
	return []AudioFormat{FormatMP3, FormatPCM}
}

func (t *ElevenLabsTTS) Voices() []VoiceInfo {
	return []VoiceInfo{
		{ID: "Rachel", Name: "Rachel", Language: "multi", Gender: "female"},
		{ID: "Adam", Name: "Adam", Language: "multi", Gender: "male"},
		{ID: "Antoni", Name: "Antoni", Language: "multi", Gender: "male"},
		{ID: "Bella", Name: "Bella", Language: "multi", Gender: "female"},
	}
}

func (t *ElevenLabsTTS) Capabilities() []catalog.Capability {
	return []catalog.Capability{{
		Provider:    t.Name(),
		Model:       t.model,
		Name:        t.model,
		Modality:    catalog.ModalityTextToSpeech,
		Features:    []string{catalog.FeatureAudioOutput},
		Stable:      true,
		OfficialAPI: true,
		Source:      "https://elevenlabs.io/docs/api-reference/text-to-speech/convert",
	}}
}

func (t *ElevenLabsTTS) Synthesize(ctx context.Context, text string, opts SynthesizeOptions) (*SynthesizeResult, error) {
	if t.apiKey == "" {
		return nil, fmt.Errorf("elevenlabs: 未配置 API Key")
	}
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("elevenlabs: text empty")
	}
	voice := strings.TrimSpace(opts.Voice)
	if voice == "" {
		voice = t.defaultVoice
	}
	body := map[string]any{
		"text":     text,
		"model_id": t.model,
	}
	if opts.Speed > 0 {
		body["voice_settings"] = map[string]any{"speed": opts.Speed}
	}
	for k, v := range opts.Extra {
		if strings.TrimSpace(k) != "" {
			body[k] = v
		}
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs: marshal request: %w", err)
	}
	resp, err := transport.Do(ctx, transport.Request{
		Provider:  t.Name(),
		Action:    "tts.synthesize",
		Method:    http.MethodPost,
		URL:       t.baseURL + "/text-to-speech/" + urlPathEscape(voice),
		Body:      payload,
		Client:    t.client,
		Transport: t.transport,
		Headers:   t.headers,
		Timeout:   t.timeout,
		Retry:     media.SubmitRetryPolicy(opts.IdempotencyKey),
		SetHeaders: func(r *http.Request) {
			r.Header.Set("Content-Type", "application/json")
			r.Header.Set("xi-api-key", t.apiKey)
			if opts.IdempotencyKey != "" {
				r.Header.Set("Idempotency-Key", opts.IdempotencyKey)
			}
		},
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, io.LimitReader(resp.Body, 50<<20)); err != nil {
		return nil, fmt.Errorf("elevenlabs: read audio: %w", err)
	}
	format := opts.Format
	if format == "" {
		format = FormatMP3
	}
	return &SynthesizeResult{
		Audio:     buf.Bytes(),
		Format:    format,
		Size:      buf.Len(),
		RequestID: resp.Header.Get("request-id"),
	}, nil
}

func urlPathEscape(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "/", "%2F")
	return s
}

var _ TTSProvider = (*ElevenLabsTTS)(nil)
var _ catalog.Provider = (*ElevenLabsTTS)(nil)
