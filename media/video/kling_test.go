package video

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

func TestKlingSubmitPollImageToVideo(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		switch r.Method {
		case http.MethodPost:
			gotSubmitPath = r.URL.Path
			_, _ = w.Write([]byte(`{"data":{"task_id":"task_123","task_status":"submitted"}}`))
		case http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"data":{"task_id":"task_123","task_status":"succeed","task_result":{"videos":[{"url":"https://cdn.example/kling.mp4","cover_image_url":"https://cdn.example/kling.jpg"}]}}}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewKling("bearer-token", "", KlingWithBaseURL(srv.URL)).(*klingProvider)
	taskID, err := p.Submit(context.Background(), Request{Prompt: "move", ImageURL: "https://example.com/a.png"})
	if err != nil {
		t.Fatal(err)
	}
	st, err := p.Poll(context.Background(), taskID)
	if err != nil {
		t.Fatal(err)
	}
	if gotSubmitPath != "/videos/image2video" || gotPollPath != "/videos/image2video/task_123" {
		t.Fatalf("unexpected paths submit=%q poll=%q", gotSubmitPath, gotPollPath)
	}
	if gotAuth != "Bearer bearer-token" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if st.State != media.TaskSucceeded || st.VideoURL == "" || st.CoverURL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}

func TestKlingJWT(t *testing.T) {
	p := NewKling("ak", "sk").(*klingProvider)
	token := p.token()
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		t.Fatalf("token parts = %d", len(parts))
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		t.Fatal(err)
	}
	var claims map[string]any
	if err := json.Unmarshal(payload, &claims); err != nil {
		t.Fatal(err)
	}
	if claims["iss"] != "ak" {
		t.Fatalf("claims = %+v", claims)
	}
}
