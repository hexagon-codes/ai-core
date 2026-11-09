package schema

import (
	"encoding/json"
	"testing"
)

func TestStringEncodedFloatBoundsKeepGoSyntax(t *testing.T) {
	type input struct {
		N float64 `json:"n,string" min:"1" max:"2" required:"true"`
	}
	validator, err := NewValidator(Of[input]())
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"01.5", "+1.5", "1.5", "1.5e0"} {
		t.Run(text, func(t *testing.T) {
			raw, err := json.Marshal(map[string]string{"n": text})
			if err != nil {
				t.Fatal(err)
			}
			var decoded input
			if err := json.Unmarshal(raw, &decoded); err != nil || decoded.N != 1.5 {
				t.Fatalf("Go fixture is not the expected value: %+v, %v", decoded, err)
			}
			if err := validator.ValidateJSON(raw); err != nil {
				t.Fatalf("valid Go value rejected: %v", err)
			}
		})
	}
	for _, raw := range []string{`{"n":"+2.5"}`, `{"n":"00.5"}`, `{"n":"not-a-number"}`} {
		if validator.ValidateJSON([]byte(raw)) == nil {
			t.Fatalf("out-of-range or invalid value accepted: %s", raw)
		}
	}
}

func TestNumericStringBoundsParticipateInOneOfCounting(t *testing.T) {
	type lower struct {
		N int64 `json:"n,string" min:"10" max:"20" required:"true"`
	}
	type upper struct {
		N int64 `json:"n,string" min:"30" max:"40" required:"true"`
	}
	validator, err := NewValidator(&Schema{OneOf: []*Schema{Of[lower](), Of[upper]()}})
	if err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{`{"n":"15"}`, `{"n":"35"}`} {
		if err := validator.ValidateJSON([]byte(raw)); err != nil {
			t.Fatalf("exactly one range matches %s: %v", raw, err)
		}
	}
	if validator.ValidateJSON([]byte(`{"n":"25"}`)) == nil {
		t.Fatal("value matching no range accepted")
	}
	overlap, err := NewValidator(&Schema{OneOf: []*Schema{Of[lower](), Of[lower]()}})
	if err != nil {
		t.Fatal(err)
	}
	if overlap.ValidateJSON([]byte(`{"n":"15"}`)) == nil {
		t.Fatal("value matching two ranges accepted")
	}
}

func TestInternalSchemaResourceReferencesUseCompiledScope(t *testing.T) {
	for _, test := range []struct {
		name, document, valid, invalid string
	}{
		{"absolute", `{"$id":"urn:resource:root","$defs":{"leaf":{"$id":"urn:resource:leaf","type":"string"}},"$ref":"urn:resource:leaf"}`, `"valid"`, `15`},
		{"relative", `{"$id":"https://schema.example.invalid/root.json","$defs":{"leaf":{"$id":"leaf.json","type":"string"}},"$ref":"leaf.json"}`, `"valid"`, `15`},
		{"nested pointer", `{"type":"object","properties":{"n":{"$id":"urn:resource:n","$defs":{"leaf":{"type":"integer"}},"$ref":"#/$defs/leaf"}}}`, `{"n":15}`, `{"n":"invalid"}`},
		{"marked resource", `{"$id":"urn:resource:root","$defs":{"leaf":{"$id":"urn:resource:leaf","type":"string","minimum":10,"x-hexagon-string-number":true}},"$ref":"urn:resource:leaf"}`, `"10"`, `"9"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			validator, err := NewValidator(json.RawMessage(test.document))
			if err != nil {
				t.Fatal(err)
			}
			if err := validator.ValidateJSON([]byte(test.valid)); err != nil {
				t.Fatalf("valid internal resource rejected: %v", err)
			}
			if validator.ValidateJSON([]byte(test.invalid)) == nil {
				t.Fatal("invalid internal resource accepted")
			}
		})
	}
}
