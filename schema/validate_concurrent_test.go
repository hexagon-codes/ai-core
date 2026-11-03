package schema

import (
	"sync"
	"testing"
)

func TestValidatorConcurrentBranchMatching(t *testing.T) {
	document := map[string]any{
		"anyOf": []any{
			map[string]any{"type": "string", "minimum": 10, numericStringKeyword: true},
			map[string]any{"type": "string", "const": "5"},
		},
	}
	validator, err := NewValidator(document)
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for iteration := 0; iteration < 50; iteration++ {
				if err := validator.ValidateJSON([]byte(`"5"`)); err != nil {
					t.Errorf("concurrent validation: %v", err)
					return
				}
			}
		}()
	}
	wait.Wait()
}
