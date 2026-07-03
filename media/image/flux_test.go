package image

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

func TestFluxSubmitPoll(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-key")
		switch r.Method {
		case http.MethodPost:
			gotSubmitPath = r.URL.Path
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode submit: %v", err)
			}
			if body["prompt"] == "" || body["width"] == nil || body["height"] == nil {
				t.Fatalf("unexpected body: %+v", body)
			}
			_, _ = w.Write([]byte(`{"id":"flux_123"}`))
		case http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"id":"flux_123","status":"Ready","result":{"sample":"https://cdn.example/flux.png"}}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewFlux("key", FluxWithBaseURL(srv.URL)).(*fluxProvider)
	taskID, err := p.Submit(context.Background(), Request{Model: "flux-pro-1.1", Prompt: "forest", Size: "1280x720"})
	if err != nil {
		t.Fatal(err)
	}
	st, err := p.Poll(context.Background(), taskID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSubmitPath != "/flux-pro-1.1" || gotPollPath != "/get_result" || gotKey != "key" {
		t.Fatalf("unexpected request submit=%q poll=%q key=%q", gotSubmitPath, gotPollPath, gotKey)
	}
	if st.State != media.TaskSucceeded || st.Result == nil || st.Result.Images[0].URL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}
