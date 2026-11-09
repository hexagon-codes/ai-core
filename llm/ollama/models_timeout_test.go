package ollama

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestModelsDiscoveryHasBoundedTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	p := New(WithBaseURL(srv.URL), WithModelsTimeout(50*time.Millisecond))
	started := time.Now()
	models := p.Models()
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("Models() took %s, want bounded discovery", elapsed)
	}
	if len(models) == 0 {
		t.Fatal("Models() returned no fallback models after discovery timeout")
	}
}
