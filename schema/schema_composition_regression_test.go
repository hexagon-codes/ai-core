package schema

import (
	"encoding/json"
	"testing"
)

func TestSchemaJSON_AnyOfOmitsEmptyType(t *testing.T) {
	s := &Schema{
		AnyOf: []*Schema{
			{
				Type:                 "object",
				Required:             []string{"path", "position", "body"},
				AdditionalProperties: false,
			},
			{
				Type:                 "object",
				Required:             []string{"path", "line", "body"},
				AdditionalProperties: false,
			},
		},
	}

	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if typ, exists := got["type"]; exists {
		t.Fatalf("composition-only schema must omit type, got %#v", typ)
	}
	branches, ok := got["anyOf"].([]any)
	if !ok || len(branches) != 2 {
		t.Fatalf("anyOf branches = %#v, want 2", got["anyOf"])
	}
	for i, branch := range branches {
		m, ok := branch.(map[string]any)
		if !ok {
			t.Fatalf("anyOf[%d] = %T, want object", i, branch)
		}
		if m["type"] != "object" {
			t.Fatalf("anyOf[%d].type = %#v, want object", i, m["type"])
		}
		if m["additionalProperties"] != false {
			t.Fatalf("anyOf[%d].additionalProperties = %#v, want false", i, m["additionalProperties"])
		}
	}
}

func TestSchemaJSON_CombinationKeywords(t *testing.T) {
	s := &Schema{
		OneOf: []*Schema{{Type: "string"}, {Type: "integer"}},
		AllOf: []*Schema{{Type: "object"}, {Type: "object"}},
	}

	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(got["oneOf"].([]any)) != 2 || len(got["allOf"].([]any)) != 2 {
		t.Fatalf("combination keywords were not preserved: %s", data)
	}
}
