package video

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

func TestCompatibleVideoProvider(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		switch r.Method {
		case http.MethodPost:
			gotSubmitPath = r.URL.Path
			_, _ = w.Write([]byte(`{"data":{"task_id":"task_1"}}`))
		case http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"data":{"status":"completed","video_url":"https://cdn.example/out.mp4"}}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewCompatible("key", CompatibleConfig{
		Name:             "gateway",
		BaseURL:          srv.URL,
		SubmitPath:       "/submit",
		PollPathTemplate: "/tasks/{task_id}",
		Models:           []string{"runway-gen4", "pika-2.2", "luma-ray2", "hailuo-02", "wan-2.7"},
	}).(*compatibleProvider)
	taskID, err := p.Submit(context.Background(), Request{Prompt: "test"})
	if err != nil {
		t.Fatal(err)
	}
	st, err := p.Poll(context.Background(), taskID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSubmitPath != "/submit" || gotPollPath != "/tasks/task_1" || gotAuth != "Bearer key" {
		t.Fatalf("unexpected request submit=%q poll=%q auth=%q", gotSubmitPath, gotPollPath, gotAuth)
	}
	if st.State != media.TaskSucceeded || st.VideoURL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}
