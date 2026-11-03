package schema

import "testing"

func TestNumericStringMarkerAppliesToRefSiblings(t *testing.T) {
	validator, err := NewValidator(map[string]any{
		"$defs": map[string]any{"base": map[string]any{"type": "object"}},
		"$ref":  "#/$defs/base",
		"type":  "object",
		"properties": map[string]any{
			"n": map[string]any{"type": "string", "minimum": 10, numericStringKeyword: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"5"}`)); err == nil {
		t.Fatal("expected sibling numeric-string constraint to reject")
	}
	if err := validator.ValidateJSON([]byte(`{"n":"12"}`)); err != nil {
		t.Fatalf("valid value rejected: %v", err)
	}
}

func TestNumericStringMarkerAppliesToDynamicRefSiblings(t *testing.T) {
	validator, err := NewValidator(map[string]any{
		"$defs":       map[string]any{"base": map[string]any{"$anchor": "base", "type": "object"}},
		"$dynamicRef": "#base",
		"type":        "object",
		"properties": map[string]any{
			"n": map[string]any{"type": "string", "minimum": 10, numericStringKeyword: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"5"}`)); err == nil {
		t.Fatal("expected sibling numeric-string constraint to reject")
	}
	if err := validator.ValidateJSON([]byte(`{"n":"12"}`)); err != nil {
		t.Fatalf("valid value rejected: %v", err)
	}
}

func TestNumericStringMarkerAppliesToAnyOfRefOutsideDefs(t *testing.T) {
	validator, err := NewValidator(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"n": map[string]any{
				"anyOf": []any{
					map[string]any{"$ref": "#/properties/tagged"},
					map[string]any{"type": "number"},
				},
			},
			"tagged": map[string]any{"type": "string", "minimum": 10, numericStringKeyword: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"5"}`)); err == nil {
		t.Fatal("expected anyOf branch ref outside $defs to enforce numeric-string constraint")
	}
	if err := validator.ValidateJSON([]byte(`{"n":"12"}`)); err != nil {
		t.Fatalf("valid value rejected: %v", err)
	}
}

func TestNumericStringMarkerAppliesToOneOfRefOutsideDefs(t *testing.T) {
	validator, err := NewValidator(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"n": map[string]any{
				"oneOf": []any{
					map[string]any{"$ref": "#/properties/tagged"},
					map[string]any{"type": "number"},
				},
			},
			"tagged": map[string]any{"type": "string", "minimum": 10, numericStringKeyword: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"5"}`)); err == nil {
		t.Fatal("expected oneOf branch ref outside $defs to enforce numeric-string constraint")
	}
	if err := validator.ValidateJSON([]byte(`{"n":"12"}`)); err != nil {
		t.Fatalf("valid value rejected: %v", err)
	}
}

func TestNumericStringMarkerAnyOfAlternativeStillPasses(t *testing.T) {
	validator, err := NewValidator(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"n": map[string]any{
				"anyOf": []any{
					map[string]any{"$ref": "#/properties/tagged"},
					map[string]any{"type": "string"},
				},
			},
			"tagged": map[string]any{"type": "string", "minimum": 10, numericStringKeyword: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"5"}`)); err != nil {
		t.Fatalf("valid alternative branch rejected: %v", err)
	}
}
