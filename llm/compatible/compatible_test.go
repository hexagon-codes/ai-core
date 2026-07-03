package compatible

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/catalog"
	"github.com/hexagon-codes/ai-core/llm"
)

func TestProfileNamesIncludeRegionalDoubao(t *testing.T) {
	names := ProfileNames()
	mustContain(t, names, ProfileDoubaoCN)
	mustContain(t, names, ProfileDoubaoGlobal)
	mustContain(t, names, ProfileModelArkCN)
	mustContain(t, names, ProfileModelArkGlobal)
}

func TestCompatibleCompleteUsesProfileBaseURLAndName(t *testing.T) {
	var gotPath, gotAuth, gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotModel = body.Model
		_, _ = w.Write([]byte(`{"id":"chatcmpl_1","model":"doubao-seed-1-6","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"created":1}`))
	}))
	defer srv.Close()

	p, err := NewProfile(ProfileDoubaoGlobal, "key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != "doubao-global" {
		t.Fatalf("Name = %q", p.Name())
	}
	resp, err := p.Complete(context.Background(), llm.CompletionRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "ok" || gotPath != "/chat/completions" || gotAuth != "Bearer key" {
		t.Fatalf("unexpected call/response: content=%q path=%q auth=%q", resp.Content, gotPath, gotAuth)
	}
	if gotModel != "doubao-seed-1-6" {
		t.Fatalf("model = %q", gotModel)
	}
}

func TestCompatibleCapabilities(t *testing.T) {
	p, err := NewProfile(ProfileModelArkCN, "key")
	if err != nil {
		t.Fatal(err)
	}
	rows := p.Capabilities()
	if len(rows) == 0 {
		t.Fatal("expected capability rows")
	}
	foundRegion := false
	for _, row := range rows {
		if row.Modality != catalog.ModalityText || !row.OfficialAPI {
			t.Fatalf("unexpected capability row: %+v", row)
		}
		for _, feature := range row.Features {
			if feature == "region:cn" {
				foundRegion = true
			}
		}
	}
	if !foundRegion {
		t.Fatalf("expected region feature in %+v", rows)
	}
}

func mustContain(t *testing.T, values []string, want string) {
	t.Helper()
	for _, value := range values {
		if value == want {
			return
		}
	}
	t.Fatalf("%q not found in %+v", want, values)
}
