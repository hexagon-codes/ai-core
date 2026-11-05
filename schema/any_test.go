package schema

import (
	"encoding/json"
	"testing"
)

func TestInterfaceFieldAllowsAnyJSONValue(t *testing.T) {
	t.Parallel()

	type input struct {
		Data any `json:"data"`
	}
	generated := Of[input]()
	property := generated.Properties["data"]
	if property == nil {
		t.Fatalf("data property missing: %s", generated.String())
	}
	if property.Type != "" {
		t.Fatalf("any property type = %q, want unconstrained", property.Type)
	}

	validator, err := NewValidator(generated)
	if err != nil {
		t.Fatalf("NewValidator() error = %v", err)
	}
	for _, value := range []any{
		map[string]any{"data": 1},
		map[string]any{"data": "text"},
		map[string]any{"data": []any{1, 2}},
		map[string]any{"data": nil},
	} {
		payload, err := json.Marshal(value)
		if err != nil {
			t.Fatalf("json.Marshal() error = %v", err)
		}
		if err := validator.ValidateJSON(payload); err != nil {
			t.Errorf("ValidateJSON(%s) error = %v", payload, err)
		}
	}
}
