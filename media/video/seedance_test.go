package video

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

func TestSeedanceGlobalSubmitPoll(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotModel, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		switch {
		case r.Method == http.MethodPost:
			gotSubmitPath = r.URL.Path
			var body struct {
				Model   string           `json:"model"`
				Content []map[string]any `json:"content"`
				Output  map[string]any   `json:"output"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode submit: %v", err)
			}
			gotModel = body.Model
			if len(body.Content) != 2 {
				t.Fatalf("content blocks = %+v", body.Content)
			}
			_, _ = w.Write([]byte(`{"id":"cgt_123","status":"queued"}`))
		case r.Method == http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"id":"req_1","model":"dreamina-seedance-2-0-260128","status":"succeeded","content":{"video_url":"https://cdn.example/video.mp4","cover_url":"https://cdn.example/cover.jpg"}}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewSeedanceGlobal("key", SeedanceWithBaseURL(srv.URL)).(*seedanceProvider)
	taskID, err := p.Submit(context.Background(), Request{
		Prompt:   "ocean",
		ImageURL: "https://example.com/first.png",
		Duration: 5,
		Ratio:    "16:9",
	})
	if err != nil {
		t.Fatal(err)
	}
	if taskID != "cgt_123" {
		t.Fatalf("taskID = %q", taskID)
	}
	st, err := p.Poll(context.Background(), taskID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSubmitPath != "/contents/generations/tasks" || gotPollPath != "/contents/generations/tasks/cgt_123" {
		t.Fatalf("unexpected paths: submit=%q poll=%q", gotSubmitPath, gotPollPath)
	}
	if gotAuth != "Bearer key" || !strings.HasPrefix(gotModel, "dreamina-seedance") {
		t.Fatalf("auth/model mismatch auth=%q model=%q", gotAuth, gotModel)
	}
	if st.State != media.TaskSucceeded || !st.Done || st.VideoURL == "" || st.CoverURL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}

func TestSeedanceCapabilitiesAreRegional(t *testing.T) {
	p := NewDoubaoSeedanceCN("key").(*seedanceProvider)
	rows := p.Capabilities()
	if len(rows) == 0 {
		t.Fatal("expected capabilities")
	}
	found := false
	for _, feature := range rows[0].Features {
		if feature == "region:cn" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected cn region feature: %+v", rows[0].Features)
	}
}
