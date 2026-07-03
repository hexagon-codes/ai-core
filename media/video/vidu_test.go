package video

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

func TestViduSubmitPoll(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		switch r.Method {
		case http.MethodPost:
			gotSubmitPath = r.URL.Path
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode submit: %v", err)
			}
			if body["model"] == "" || body["prompt"] == "" {
				t.Fatalf("unexpected body: %+v", body)
			}
			_, _ = w.Write([]byte(`{"task_id":"vidu_123"}`))
		case http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"status":"success","creations":[{"url":"https://cdn.example/vidu.mp4","cover_url":"https://cdn.example/vidu.jpg"}]}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewVidu("key", ViduWithBaseURL(srv.URL)).(*viduProvider)
	taskID, err := p.Submit(context.Background(), Request{Prompt: "city"})
	if err != nil {
		t.Fatal(err)
	}
	st, err := p.Poll(context.Background(), taskID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSubmitPath != "/ent/v2/text2video" || gotPollPath != "/ent/v2/tasks/vidu_123/creations" {
		t.Fatalf("paths submit=%q poll=%q", gotSubmitPath, gotPollPath)
	}
	if gotAuth != "Token key" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if st.State != media.TaskSucceeded || st.VideoURL == "" || st.CoverURL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}

func TestViduSubmitTransmitsIdempotencyKey(t *testing.T) {
	var gotIdempotencyKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotIdempotencyKey = r.Header.Get("Idempotency-Key")
		_, _ = w.Write([]byte(`{"task_id":"vidu_123"}`))
	}))
	defer srv.Close()

	p := NewVidu("key", ViduWithBaseURL(srv.URL)).(*viduProvider)
	if _, err := p.Submit(context.Background(), Request{Prompt: "city", IdempotencyKey: "task-42"}); err != nil {
		t.Fatal(err)
	}
	if gotIdempotencyKey != "task-42" {
		t.Fatalf("Idempotency-Key = %q, want %q", gotIdempotencyKey, "task-42")
	}
}
