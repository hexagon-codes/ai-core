package schema

import (
	"encoding/json"
	"testing"
)

func verifyWireValue[T any](t *testing.T, value T) {
	t.Helper()
	raw, e := json.Marshal(value)
	if e != nil {
		t.Fatal(e)
	}
	v, e := NewValidator(Of[T]())
	if e != nil {
		t.Fatal(e)
	}
	if e = v.ValidateJSON(raw); e != nil {
		t.Fatalf("JSON=%s error=%v", raw, e)
	}
}
func TestStaticSchemaMatchesStandardJSONTypes(t *testing.T) {
	type values struct {
		Bytes  []byte          `json:"bytes"`
		Number json.Number     `json:"number"`
		Raw    json.RawMessage `json:"raw"`
		ID     int64           `json:"id,string"`
		Flag   bool            `json:"flag,string"`
		Text   string          `json:"text,string"`
	}
	verifyWireValue(t, values{[]byte("hello"), "9007199254740993", json.RawMessage(`{"ok":true}`), 9007199254740993, true, "hello"})
	verifyWireValue(t, values{nil, "0", nil, 0, false, ""})
}
func TestStringEncodedTagsKeepWireConstraints(t *testing.T) {
	type input struct {
		ID   int64  `json:"id,string" enum:"1,2" default:"1"`
		Text string `json:"text,string" enum:"hello"`
	}
	v, e := NewValidator(Of[input]())
	if e != nil {
		t.Fatal(e)
	}
	if e = v.ValidateJSON([]byte(`{"id":"1","text":"\"hello\""}`)); e != nil {
		t.Fatal(e)
	}
	if v.ValidateJSON([]byte(`{"id":1,"text":"hello"}`)) == nil {
		t.Fatal("unquoted representation accepted")
	}
}
func TestOutOfRangeIntegerTagKeepsNumericRepresentation(t *testing.T) {
	type input struct {
		N uint64 `json:"n" enum:"18446744073709551616" default:"18446744073709551616"`
	}
	s := Of[input]()
	p := s.Properties["n"]
	if p.Enum[0] != json.Number("18446744073709551616") || p.Default != json.Number("18446744073709551616") {
		t.Fatalf("numeric literal changed type: %+v", p)
	}
}

func TestOutOfRangeNumberTagKeepsNumericRepresentation(t *testing.T) {
	type input struct {
		N float64 `json:"n" enum:"1e400" default:"1e400"`
	}
	s := Of[input]()
	if s.Properties["n"].Enum[0] != json.Number("1e400") || s.Properties["n"].Default != json.Number("1e400") {
		t.Fatal("overflowed numeric literal became a string")
	}
	raw, e := json.Marshal(s)
	if e != nil {
		t.Fatal(e)
	}
	var roundtrip Schema
	if e = json.Unmarshal(raw, &roundtrip); e != nil {
		t.Fatal(e)
	}
	if roundtrip.Properties["n"].Enum[0] != json.Number("1e400") {
		t.Fatal("literal lost in JSON roundtrip")
	}
}

func TestStringEncodedNumericBoundsKeepRangeSemantics(t *testing.T) {
	type input struct {
		ID   int64  `json:"id,string" min:"10" max:"20"`
		Name string `json:"name" min:"2" max:"5"`
	}
	s := Of[input]()
	id := s.Properties["id"]
	if id.Type != "string" {
		t.Fatalf("id type=%q", id.Type)
	}
	if id.MinLength != nil || id.MaxLength != nil {
		t.Fatalf("numeric bounds became string lengths: %+v", id)
	}
	if id.Minimum == nil || *id.Minimum != 10 || id.MinimumNumber != "10" {
		t.Fatalf("minimum lost: %+v", id)
	}
	if id.Maximum == nil || *id.Maximum != 20 || id.MaximumNumber != "20" {
		t.Fatalf("maximum lost: %+v", id)
	}
	name := s.Properties["name"]
	if name.MinLength == nil || *name.MinLength != 2 || name.MaxLength == nil || *name.MaxLength != 5 {
		t.Fatalf("string length bounds changed: %+v", name)
	}
	validator, err := NewValidator(s)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"10", "20"} {
		if err := validator.ValidateJSON([]byte(`{"id":"` + value + `","name":"ab"}`)); err != nil {
			t.Fatalf("valid string-encoded number %q rejected: %v", value, err)
		}
	}
}

func TestStringEncodedNumericBoundsRejectInvalidValues(t *testing.T) {
	type input struct {
		N int64   `json:"n,string" min:"10" max:"20" required:"true"`
		F float64 `json:"f,string" min:"1.5" max:"2.5" required:"true"`
	}
	validator, err := NewValidator(Of[input]())
	if err != nil {
		t.Fatal(err)
	}
	valid := []string{
		`{"n":"10","f":"1.5"}`,
		`{"n":"20","f":"2.5"}`,
		`{"n":"15","f":"2"}`,
	}
	for _, value := range valid {
		if err := validator.ValidateJSON([]byte(value)); err != nil {
			t.Fatalf("valid value rejected: %s: %v", value, err)
		}
	}
	invalid := []string{
		`{"n":"9","f":"2"}`,
		`{"n":"21","f":"2"}`,
		`{"n":"abc","f":"2"}`,
		`{"n":"15","f":"1.2"}`,
		`{"n":"15","f":"2.6"}`,
		`{"n":"15","f":"x"}`,
	}
	for _, value := range invalid {
		if err := validator.ValidateJSON([]byte(value)); err == nil {
			t.Fatalf("invalid value accepted: %s", value)
		}
	}
}

func TestStandardStringSchemasKeepNumericMinimumAsAnnotation(t *testing.T) {
	type tagged struct {
		N int64 `json:"n,string" min:"10" max:"20" required:"true"`
	}
	if !hasNumericStringMarker(map[string]any{
		"type":               "string",
		"minimum":            json.Number("10"),
		numericStringKeyword: true,
	}) {
		t.Fatal("generated numeric-string marker missing")
	}
	cases := []struct {
		name   string
		schema any
		value  string
	}{
		{"string", map[string]any{"type": "string", "minimum": 10}, `"ordinary text"`},
		{"union", map[string]any{"type": []any{"string", "number"}, "minimum": 10}, `"ordinary text"`},
		{"no-type", map[string]any{"minimum": 10}, `"ordinary text"`},
		{"nested", map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"minimum": 10}}}, `{"n":"ordinary text"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			validator, err := NewValidator(tc.schema)
			if err != nil {
				t.Fatal(err)
			}
			if err := validator.ValidateJSON([]byte(tc.value)); err != nil {
				t.Fatalf("standard string value rejected: %v", err)
			}
		})
	}
}

func TestNumericStringMarkerDoesNotOverrideAlternativeAnyOfBranch(t *testing.T) {
	document := map[string]any{
		"anyOf": []any{
			map[string]any{
				"type":               "string",
				"minimum":            10,
				numericStringKeyword: true,
			},
			map[string]any{"type": "string", "const": "5"},
		},
	}
	validator, err := NewValidator(document)
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`"5"`)); err != nil {
		t.Fatalf("alternative anyOf branch was rejected by inactive marker: %v", err)
	}
}

func TestNumericStringMarkerAppliesThroughConditionalThen(t *testing.T) {
	type tagged struct {
		N int64 `json:"n,string" min:"10" max:"20" required:"true"`
	}
	document := map[string]any{
		"if":   map[string]any{"type": "object", "required": []any{"n"}},
		"then": Of[tagged](),
	}
	validator, err := NewValidator(document)
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"n":"9"}`)); err == nil {
		t.Fatal("conditional then marker was not enforced")
	}
}

func TestStringEncodedNumericBoundsApplyThroughCompositionsAndAnchors(t *testing.T) {
	type tagged struct {
		N int64 `json:"n,string" min:"10" max:"20" required:"true"`
	}
	taggedSchema := Of[tagged]()
	for _, tc := range []struct {
		name   string
		schema any
		value  string
	}{
		{"direct", taggedSchema, `{"n":"9"}`},
		{"anyOf", map[string]any{"anyOf": []any{taggedSchema, map[string]any{"type": "null"}}}, `{"n":"9"}`},
		{"oneOf", map[string]any{"oneOf": []any{taggedSchema, map[string]any{"type": "null"}}}, `{"n":"21"}`},
		{"prefixItems", map[string]any{"type": "array", "prefixItems": []any{taggedSchema}}, `[{"n":"abc"}]`},
		{"anchor", map[string]any{"$defs": map[string]any{"item": map[string]any{"$anchor": "item", "allOf": []any{taggedSchema}}}, "$ref": "#item"}, `{"n":"9"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			validator, err := NewValidator(tc.schema)
			if err != nil {
				t.Fatal(err)
			}
			if err := validator.ValidateJSON([]byte(tc.value)); err == nil {
				t.Fatalf("invalid string-encoded numeric value accepted: %s", tc.value)
			}
		})
	}
}

func TestStringEncodedNumericBoundsApplyThroughRefsAndArrays(t *testing.T) {
	type node struct {
		N    int64 `json:"n,string" min:"10" max:"20" required:"true"`
		Next *node `json:"next,omitempty"`
	}
	type wrapper struct {
		Items []node `json:"items" required:"true"`
	}
	validator, err := NewValidator(Of[wrapper]())
	if err != nil {
		t.Fatal(err)
	}
	if err := validator.ValidateJSON([]byte(`{"items":[{"n":"10","next":{"n":"20"}}]}`)); err != nil {
		t.Fatalf("valid nested value rejected: %v", err)
	}
	if err := validator.ValidateJSON([]byte(`{"items":[{"n":"10","next":{"n":"9"}}]}`)); err == nil {
		t.Fatal("nested out-of-range value accepted")
	}
	if err := validator.ValidateJSON([]byte(`{"items":[{"n":"21"}]}`)); err == nil {
		t.Fatal("array out-of-range value accepted")
	}
}
