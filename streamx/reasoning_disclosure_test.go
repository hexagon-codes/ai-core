package streamx

import "testing"

func TestOpenAIParserReasoningDisclosurePreservesDialectAndFailsClosed(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		dialect string
	}{
		{
			name:    "reasoning",
			payload: `{"model":"route-body-model","choices":[{"delta":{"reasoning":"private trace"}}]}`,
			dialect: "delta.reasoning",
		},
		{
			name:    "reasoning_content",
			payload: `{"model":"route-body-model","choices":[{"delta":{"reasoning_content":"private trace"}}]}`,
			dialect: "delta.reasoning_content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunk, err := (&OpenAIParser{}).Parse([]byte(tt.payload))
			if err != nil {
				t.Fatal(err)
			}
			if chunk.Reasoning != "private trace" {
				t.Fatalf("legacy reasoning = %q, want raw compatibility value", chunk.Reasoning)
			}
			want := ReasoningDisclosure{
				Visibility: ReasoningNotExposed,
				Source:     "openai_compatible",
				Dialect:    tt.dialect,
			}
			if chunk.ReasoningDisclosure == nil || *chunk.ReasoningDisclosure != want {
				t.Fatalf("reasoning disclosure = %#v, want %#v", chunk.ReasoningDisclosure, want)
			}
			if chunk.ReasoningDisclosure.Provider != "" || chunk.ReasoningDisclosure.Model != "" {
				t.Fatalf("untrusted response body supplied route provenance: %#v", chunk.ReasoningDisclosure)
			}
		})
	}
}

func TestOpenAIParserReasoningDisclosureRequiresExplicitPublicRouteEvidence(t *testing.T) {
	parser := &OpenAIParser{ReasoningEvidence: ReasoningDisclosureEvidence{
		ExplicitlyPublic: true,
		Provider:         "openai-compatible-test",
		Model:            "frozen-route-model",
	}}
	chunk, err := parser.Parse([]byte(
		`{"model":"untrusted-body-model","choices":[{"delta":{"reasoning_content":"public summary"}}]}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	want := ReasoningDisclosure{
		Visibility: ReasoningVisible,
		Source:     "openai_compatible",
		Dialect:    "delta.reasoning_content",
		Provider:   "openai-compatible-test",
		Model:      "frozen-route-model",
	}
	if chunk.ReasoningDisclosure == nil || *chunk.ReasoningDisclosure != want {
		t.Fatalf("reasoning disclosure = %#v, want %#v", chunk.ReasoningDisclosure, want)
	}
}

func TestOpenAIParserUnknownAndIncompleteEvidenceRemainNotExposed(t *testing.T) {
	unknown, err := (&OpenAIParser{ReasoningEvidence: ReasoningDisclosureEvidence{
		ExplicitlyPublic: true,
		Provider:         "openai-compatible-test",
		Model:            "frozen-route-model",
	}}).Parse([]byte(`{"choices":[{"delta":{"thinking":"reasoning-shaped private text"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if unknown.ReasoningDisclosure != nil || unknown.Reasoning != "" {
		t.Fatalf("unknown dialect inferred as disclosure: %#v", unknown)
	}

	incomplete, err := (&OpenAIParser{ReasoningEvidence: ReasoningDisclosureEvidence{
		ExplicitlyPublic: true,
		Provider:         "openai-compatible-test",
	}}).Parse([]byte(`{"choices":[{"delta":{"reasoning":"private trace"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if incomplete.ReasoningDisclosure == nil ||
		incomplete.ReasoningDisclosure.Visibility != ReasoningNotExposed {
		t.Fatalf("incomplete provenance became displayable: %#v", incomplete.ReasoningDisclosure)
	}
}
