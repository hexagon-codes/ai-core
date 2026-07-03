package video

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

func TestVeoSubmitPoll(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-goog-api-key")
		switch r.Method {
		case http.MethodPost:
			gotSubmitPath = r.URL.Path
			_, _ = w.Write([]byte(`{"name":"operations/op_123"}`))
		case http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"name":"operations/op_123","done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":"https://cdn.example/veo.mp4"}}]}}}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewVeo("key", VeoWithBaseURL(srv.URL)).(*veoProvider)
	taskID, err := p.Submit(context.Background(), Request{Model: "veo-3.1-generate-preview", Prompt: "ocean", WithAudio: true})
	if err != nil {
		t.Fatal(err)
	}
	st, err := p.Poll(context.Background(), taskID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSubmitPath != "/models/veo-3.1-generate-preview:predictLongRunning" || gotPollPath != "/operations/op_123" || gotKey != "key" {
		t.Fatalf("unexpected request submit=%q poll=%q key=%q", gotSubmitPath, gotPollPath, gotKey)
	}
	if st.State != media.TaskSucceeded || st.VideoURL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}

func TestVeoSubmitTransmitsIdempotencyKey(t *testing.T) {
	var gotIdempotencyKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotIdempotencyKey = r.Header.Get("Idempotency-Key")
		_, _ = w.Write([]byte(`{"name":"operations/op_123"}`))
	}))
	defer srv.Close()

	p := NewVeo("key", VeoWithBaseURL(srv.URL)).(*veoProvider)
	if _, err := p.Submit(context.Background(), Request{Prompt: "ocean", IdempotencyKey: "task-42"}); err != nil {
		t.Fatal(err)
	}
	if gotIdempotencyKey != "task-42" {
		t.Fatalf("Idempotency-Key = %q, want %q", gotIdempotencyKey, "task-42")
	}
}
