package ollama

import (
	"testing"

	"github.com/hexagon-codes/ai-core/streamx"
)

func TestOllamaStreamParserThinkingDisclosurePreservesDialectAndFailsClosed(t *testing.T) {
	chunk, err := (&ollamaStreamParser{}).Parse([]byte(
		`{"model":"untrusted-body-model","message":{"thinking":"private trace"}}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	if chunk.Reasoning != "private trace" {
		t.Fatalf("legacy reasoning = %q, want raw compatibility value", chunk.Reasoning)
	}
	want := streamx.ReasoningDisclosure{
		Visibility: streamx.ReasoningNotExposed,
		Source:     "ollama",
		Dialect:    "message.thinking",
	}
	if chunk.ReasoningDisclosure == nil || *chunk.ReasoningDisclosure != want {
		t.Fatalf("reasoning disclosure = %#v, want %#v", chunk.ReasoningDisclosure, want)
	}
	if chunk.ReasoningDisclosure.Provider != "" || chunk.ReasoningDisclosure.Model != "" {
		t.Fatalf("untrusted response body supplied route provenance: %#v", chunk.ReasoningDisclosure)
	}
}

func TestOllamaStreamParserThinkingDisclosureRequiresExplicitPublicRouteEvidence(t *testing.T) {
	parser := &ollamaStreamParser{reasoningEvidence: streamx.ReasoningDisclosureEvidence{
		ExplicitlyPublic: true,
		Provider:         "ollama",
		Model:            "frozen-route-model",
	}}
	chunk, err := parser.Parse([]byte(
		`{"model":"untrusted-body-model","message":{"thinking":"public summary"}}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	want := streamx.ReasoningDisclosure{
		Visibility: streamx.ReasoningVisible,
		Source:     "ollama",
		Dialect:    "message.thinking",
		Provider:   "ollama",
		Model:      "frozen-route-model",
	}
	if chunk.ReasoningDisclosure == nil || *chunk.ReasoningDisclosure != want {
		t.Fatalf("reasoning disclosure = %#v, want %#v", chunk.ReasoningDisclosure, want)
	}
}

func TestOllamaStreamParserLegacyChunkHasNoDisplayableDisclosure(t *testing.T) {
	chunk, err := (&ollamaStreamParser{reasoningEvidence: streamx.ReasoningDisclosureEvidence{
		ExplicitlyPublic: true,
		Provider:         "ollama",
		Model:            "frozen-route-model",
	}}).Parse([]byte(`{"message":{"content":"answer"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if chunk.ReasoningDisclosure != nil || chunk.Reasoning != "" {
		t.Fatalf("legacy chunk inferred as disclosure: %#v", chunk)
	}
}
