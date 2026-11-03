package schema

import (
	"encoding/json"
	"testing"
)

func TestSchemaRoundTripPreservesExactConstraints(t *testing.T) {
	cases := []struct{ name, raw, good, bad string }{
		{"enum", `{"type":"integer","enum":[9007199254740993]}`, `9007199254740993`, `9007199254740992`},
		{"minimum", `{"type":"integer","minimum":9007199254740993}`, `9007199254740993`, `9007199254740992`},
		{"maximum", `{"type":"integer","maximum":9007199254740995}`, `9007199254740995`, `9007199254740996`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var decoded Schema
			if err := json.Unmarshal([]byte(c.raw), &decoded); err != nil {
				t.Fatal(err)
			}
			v, err := NewValidator(decoded)
			if err != nil {
				t.Fatal(err)
			}
			if err := v.ValidateJSON([]byte(c.good)); err != nil {
				t.Fatalf("valid value rejected: %v", err)
			}
			if v.ValidateJSON([]byte(c.bad)) == nil {
				t.Fatal("out-of-contract value accepted")
			}
		})
	}
	var empty Schema
	if err := json.Unmarshal([]byte(`{"enum":[]}`), &empty); err != nil {
		t.Fatal(err)
	}
	v, err := NewValidator(empty)
	if err != nil {
		t.Fatal(err)
	}
	if v.ValidateJSON([]byte(`1`)) == nil {
		t.Fatal("empty enum was discarded")
	}
	var invalid Schema
	if err := json.Unmarshal([]byte(`{"anyOf":[]}`), &invalid); err != nil {
		t.Fatal(err)
	}
	if _, err := NewValidator(invalid); err == nil {
		t.Fatal("invalid empty anyOf was discarded")
	}
}

func TestStaticSchemaPreservesNumericTags(t *testing.T) {
	type minimum struct {
		N int64 `json:"n" min:"9007199254740993"`
	}
	type maximum struct {
		N int64 `json:"n" max:"9007199254740995"`
	}
	type unsigned struct {
		N uint64 `json:"n" enum:"18446744073709551615"`
	}
	cases := []struct {
		name      string
		s         *Schema
		good, bad string
	}{
		{"minimum", Of[minimum](), `{"n":9007199254740993}`, `{"n":9007199254740992}`},
		{"maximum", Of[maximum](), `{"n":9007199254740995}`, `{"n":9007199254740996}`},
		{"unsigned enum", Of[unsigned](), `{"n":18446744073709551615}`, `{"n":1}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v, err := NewValidator(c.s)
			if err != nil {
				t.Fatal(err)
			}
			if err := v.ValidateJSON([]byte(c.good)); err != nil {
				t.Fatalf("valid tag value rejected: %v", err)
			}
			if v.ValidateJSON([]byte(c.bad)) == nil {
				t.Fatal("invalid tag value accepted")
			}
		})
	}
}
