package schema

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEnumTagsPreserveFieldTypes(t *testing.T) {
	type input struct {
		Count   int     `json:"count" enum:"1,2"`
		Ratio   float64 `json:"ratio" enum:"0.5,1.5"`
		Enabled bool    `json:"enabled" enum:"true,false"`
		Label   string  `json:"label" enum:"one,two"`
	}
	encoded, err := json.Marshal(Of[input]())
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Properties map[string]struct {
			Enum []any `json:"enum"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	want := map[string][]any{
		"count":   {float64(1), float64(2)},
		"ratio":   {0.5, 1.5},
		"enabled": {true, false},
		"label":   {"one", "two"},
	}
	for name, expected := range want {
		if !reflect.DeepEqual(got.Properties[name].Enum, expected) {
			t.Errorf("%s enum=%#v, want %#v", name, got.Properties[name].Enum, expected)
		}
	}
}
