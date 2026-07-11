package ollama

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestAudit20260711OfficialShowModelInfoProvidesContextCap(t *testing.T) {
	var showCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			// Official list-models responses expose identity/details, but not
			// capabilities or context_length.
			_, _ = w.Write([]byte(`{"models":[{"name":"tiny:latest","model":"tiny:latest","size":1,"digest":"sha256:test","details":{"format":"gguf","family":"tiny","families":["tiny"],"parameter_size":"1B","quantization_level":"Q4_K_M"}}]}`))
		case "/api/show":
			showCalls++
			var req map[string]string
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode show request: %v", err)
			}
			if req["model"] != "tiny:latest" {
				t.Fatalf("show model=%q, want tiny:latest", req["model"])
			}
			_, _ = w.Write([]byte(`{"capabilities":["completion"],"model_info":{"tiny.context_length":8192}}`))
		default:
			t.Fatalf("unexpected endpoint %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL))
	models := p.Models()
	if showCalls != 1 {
		t.Fatalf("show calls=%d, want 1", showCalls)
	}
	if len(models) != 1 {
		t.Fatalf("models=%+v, want one model", models)
	}
	if models[0].MaxTokens != 8192 {
		t.Fatalf("official /api/show context cap was lost: MaxTokens=%d, want 8192", models[0].MaxTokens)
	}

	req := llm.CompletionRequest{
		Model:     "tiny:latest",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
		MaxTokens: 10_000,
	}
	if got := p.numCtxForRequest(req); got != 8192 {
		t.Fatalf("automatic num_ctx ignored the real model cap: got=%d, want 8192", got)
	}
}

func TestAudit20260711TagsCapabilitiesAndShowContextAreMerged(t *testing.T) {
	var showCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"tiny:latest","capabilities":["vision"],"details":{"family":"tiny","parameter_size":"1B"}}]}`))
		case "/api/show":
			showCalls++
			_, _ = w.Write([]byte(`{"model_info":{"tiny.context_length":8192}}`))
		default:
			t.Fatalf("unexpected endpoint %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	models := New(WithBaseURL(srv.URL)).Models()
	if showCalls != 1 {
		t.Fatalf("show calls=%d, want 1 when tags omits context length", showCalls)
	}
	if len(models) != 1 {
		t.Fatalf("models=%+v, want one model", models)
	}
	if models[0].MaxTokens != 8192 {
		t.Fatalf("show context cap was not merged with tags capabilities: MaxTokens=%d, want 8192", models[0].MaxTokens)
	}
	if !models[0].HasFeature(llm.FeatureVision) {
		t.Fatalf("tags capability was lost while merging show details: features=%v", models[0].Features)
	}
}
