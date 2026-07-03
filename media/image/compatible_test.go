package image

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/media"
)

func TestAsyncCompatibleImageProvider(t *testing.T) {
	var gotSubmitPath, gotPollPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		switch r.Method {
		case http.MethodPost:
			gotSubmitPath = r.URL.Path
			_, _ = w.Write([]byte(`{"data":{"task_id":"task_1"}}`))
		case http.MethodGet:
			gotPollPath = r.URL.Path
			_, _ = w.Write([]byte(`{"data":{"status":"succeeded","image_url":"https://cdn.example/out.png"}}`))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	p := NewAsyncCompatible("key", CompatibleConfig{
		Name:             "image-gateway",
		BaseURL:          srv.URL,
		SubmitPath:       "/submit",
		PollPathTemplate: "/tasks/{task_id}",
		Models:           []string{"midjourney-v7", "ideogram-v3", "recraft-v3", "stable-image-ultra"},
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
	if st.State != media.TaskSucceeded || st.Result == nil || st.Result.Images[0].URL == "" {
		t.Fatalf("unexpected status: %+v", st)
	}
}

func TestOpenAICompatibleImageNetworkPolicyBlocksPrivateBaseURL(t *testing.T) {
	providers := []struct {
		name string
		p    Provider
	}{
		{
			name: "dalle",
			p: NewOpenAIDallEWithOptions("key", "http://127.0.0.1:1",
				OpenAICompatWithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true})),
		},
		{
			name: "zhipu",
			p: NewZhipuCogViewWithOptions("key", "http://127.0.0.1:1",
				OpenAICompatWithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true})),
		},
		{
			name: "generic",
			p: NewOpenAICompatibleWithOptions("generic-image", "key", "http://127.0.0.1:1", []string{"m"},
				OpenAICompatWithNetworkPolicy(llm.NetworkPolicy{AllowHTTP: true})),
		},
	}
	for _, tt := range providers {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.p.Generate(context.Background(), Request{Prompt: "x"})
			if err == nil {
				t.Fatal("NetworkPolicy should block private baseURL")
			}
			if !errors.Is(err, llm.ErrNetworkPolicy) {
				t.Fatalf("expected ErrNetworkPolicy, got %v", err)
			}
		})
	}
}
