package schema

import (
	"encoding/json"
	"testing"
)

func TestValidatorChecksDynamicSchemaAndOriginalFields(t *testing.T) {
	validator, err := NewValidator(json.RawMessage(`{"type":"object","properties":{"count":{"type":"integer","minimum":1,"maximum":3},"kind":{"enum":["clip"]}},"required":["count","kind"],"additionalProperties":false}`))
	if err != nil {
		t.Fatal(err)
	}
	for name, document := range map[string]string{
		"missing": `{"count":2}`,
		"type":    `{"count":"2","kind":"clip"}`,
		"range":   `{"count":4,"kind":"clip"}`,
		"enum":    `{"count":2,"kind":"other"}`,
		"unknown": `{"count":2,"kind":"clip","hidden":true}`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := validator.ValidateJSON([]byte(document)); err == nil {
				t.Fatal("invalid instance accepted")
			}
		})
	}
	if err := validator.ValidateJSON([]byte(`{"count":2,"kind":"clip"}`)); err != nil {
		t.Fatal(err)
	}
}

func TestValidatorResolvesGeneratedRecursiveSchema(t *testing.T) {
	validator, err := NewValidator(Of[recursiveNode]())
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"name":"root","children":[{"name":"child"}]}`)); err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"name":"root","children":[{"name":12}]}`)); err == nil {
		t.Fatal("recursive child type was not validated")
	}
}

func TestValidatorPreservesLargeIntegerPrecision(t *testing.T) {
	validator, err := NewValidator(json.RawMessage(`{"const":9007199254740993}`))
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`9007199254740993`)); err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`9007199254740992`)); err == nil {
		t.Fatal("distinct large integers became equal")
	}
}
