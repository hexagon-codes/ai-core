package schema

import (
	"encoding/json"
	"testing"
)

func TestStringEncodedIntegerSyntaxAcrossSchemaForms(t *testing.T) {
	type input struct {
		N int64 `json:"n,string" min:"-20" max:"20" required:"true"`
	}
	s := Of[input]()
	forms := map[string]any{
		"direct": s,
		"anyOf":  map[string]any{"anyOf": []any{s}},
		"oneOf":  map[string]any{"oneOf": []any{s}},
		"ref":    map[string]any{"$defs": map[string]any{"input": s}, "$ref": "#/$defs/input"},
	}
	for name, document := range forms {
		t.Run(name, func(t *testing.T) {
			validator, err := NewValidator(document)
			if err != nil {
				t.Fatal(err)
			}
			for _, text := range []string{"15", "-15", "015", "-015", "-0", "15.5", "1.5e1", " 15", "15 "} {
				raw, err := json.Marshal(map[string]string{"n": text})
				if err != nil {
					t.Fatal(err)
				}
				var decoded input
				decodeErr := json.Unmarshal(raw, &decoded)
				validateErr := validator.ValidateJSON(raw)
				if (validateErr == nil) != (decodeErr == nil) {
					t.Errorf("value %q: validation = %v, Go decoding = %v", text, validateErr, decodeErr)
				}
			}
			if err := validator.ValidateJSON([]byte(`{"n":"+15"}`)); err == nil {
				t.Error("leading plus is outside the minimum Go JSON string-number syntax")
			}
		})
	}
}

func TestStringEncodedIntegerSyntaxWithoutBoundsAndWithPattern(t *testing.T) {
	type input struct {
		Signed   *int64  `json:"signed,string" desc:"Signed integer" default:"15" format:"int64"`
		Unsigned uint64  `json:"unsigned,string"`
		Pattern  int64   `json:"pattern,string" pattern:"^1"`
		Float    float64 `json:"float,string"`
		Choice   *int64  `json:"choice,string,omitempty" enum:"12,15"`
	}
	document := Of[input]()
	if field := document.Properties["signed"]; field.Description != "Signed integer" || field.Default != "15" || field.Format != "int64" {
		t.Fatalf("nullable field annotations moved away from property: %+v", field)
	}
	if field := document.Properties["choice"]; len(field.Enum) != 2 || field.Enum[0] != "12" || field.Enum[1] != "15" {
		t.Fatalf("nullable field enum moved away from property: %+v", field)
	}
	validator, err := NewValidator(document)
	if err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{
		`{"signed":"-015","unsigned":"18446744073709551615","pattern":"15","float":"1.5e1"}`,
		`{"signed":null,"unsigned":"0","pattern":"12","float":"1.5"}`,
	} {
		var decoded input
		if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
			t.Fatalf("invalid Go fixture: %v", err)
		}
		if err := validator.ValidateJSON([]byte(raw)); err != nil {
			t.Errorf("valid fixture rejected: %s: %v", raw, err)
		}
	}
	for _, raw := range []string{
		`{"signed":"1.5","unsigned":"0","pattern":"15","float":"1.5"}`,
		`{"signed":"1e1","unsigned":"0","pattern":"15","float":"1.5"}`,
		`{"signed":"1","unsigned":"-0","pattern":"15","float":"1.5"}`,
		`{"signed":"1","unsigned":"0","pattern":"1.5","float":"1.5"}`,
		`{"signed":"1","unsigned":"0","pattern":"2","float":"1.5"}`,
		`{"signed":"1","unsigned":"0","pattern":"15","float":"1.5","choice":null}`,
	} {
		if err := validator.ValidateJSON([]byte(raw)); err == nil {
			t.Errorf("invalid fixture accepted: %s", raw)
		}
	}
}
