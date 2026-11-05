package schema

import (
	"testing"
)

type numericRecursiveNode struct {
	N    int64                 `json:"n,string" min:"10"`
	Next *numericRecursiveNode `json:"next,omitempty"`
}

func TestNumericStringConstraintsApplyThroughRecursiveRef(t *testing.T) {
	t.Parallel()

	validator, err := NewValidator(Of[numericRecursiveNode]())
	if err != nil {
		t.Fatalf("NewValidator() error = %v", err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"10","next":{"n":"10","next":{"n":"9"}}}`)); err == nil {
		t.Fatal("nested string-encoded value below minimum was accepted")
	}
	if err := validator.ValidateJSON([]byte(`{"n":"10","next":{"n":"10","next":{"n":"11"}}}`)); err != nil {
		t.Fatalf("valid nested value rejected: %v", err)
	}
}
