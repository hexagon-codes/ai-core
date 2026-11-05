package tool

import (
	"context"
	"testing"
)

type validatedInput struct {
	Mode  string `json:"mode" enum:"fast,slow" required:"true"`
	Count int    `json:"count" min:"1" max:"3"`
}

func TestFuncToolValidateUsesSchemaConstraints(t *testing.T) {
	t.Parallel()

	called := false
	tool := NewFunc("validated", "validated tool", func(context.Context, validatedInput) (string, error) {
		called = true
		return "ok", nil
	})

	if err := tool.Validate(map[string]any{"mode": "invalid", "count": 1.0}); err == nil {
		t.Fatal("Validate() accepted an out-of-enum value")
	}
	if err := tool.Validate(map[string]any{"mode": "fast", "count": 4.0}); err == nil {
		t.Fatal("Validate() accepted a value above max")
	}
	if err := tool.Validate(map[string]any{"count": 1.0}); err == nil {
		t.Fatal("Validate() accepted a missing required field")
	}
	if err := tool.Validate(map[string]any{"mode": "fast", "count": 2.0}); err != nil {
		t.Fatalf("Validate() rejected valid input: %v", err)
	}

	result, err := tool.Execute(context.Background(), map[string]any{"mode": "invalid", "count": 1.0})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if result.Success {
		t.Fatal("Execute() succeeded with invalid arguments")
	}
	if called {
		t.Fatal("Execute() called the function with invalid arguments")
	}
}
