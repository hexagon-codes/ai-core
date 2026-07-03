package voice

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestElevenLabsTTS(t *testing.T) {
	var gotPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("xi-api-key")
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["text"] == "" || body["model_id"] == "" {
			t.Fatalf("unexpected body: %+v", body)
		}
		w.Header().Set("request-id", "req_123")
		_, _ = w.Write([]byte("audio-bytes"))
	}))
	defer srv.Close()

	tts := NewElevenLabsTTS("key", ElevenLabsWithBaseURL(srv.URL), ElevenLabsWithVoice("voice-1"))
	res, err := tts.Synthesize(context.Background(), "hello", SynthesizeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/text-to-speech/voice-1" || gotKey != "key" {
		t.Fatalf("path/key = %q/%q", gotPath, gotKey)
	}
	if string(res.Audio) != "audio-bytes" || res.RequestID != "req_123" {
		t.Fatalf("unexpected result: %+v", res)
	}
}
