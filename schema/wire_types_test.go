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
