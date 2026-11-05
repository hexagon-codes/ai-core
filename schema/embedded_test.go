package schema

import (
	"encoding/json"
	"reflect"
	"testing"
)

type embeddedBase struct {
	ID string `json:"id" required:"true"`
}

type embeddedExported struct {
	Name string `json:"name"`
}

type embeddedInput struct {
	embeddedBase
	embeddedExported
	Email string `json:"email"`
}

type namedEmbeddedInput struct {
	embeddedExported `json:"profile"`
	Email            string `json:"email"`
}

type ConflictA struct {
	ValueA string `json:"value" required:"true"`
}

type ConflictB struct {
	ValueB string `json:"value"`
}

func TestOfFlattensAnonymousEmbeddedStructs(t *testing.T) {
	t.Parallel()

	generated := Of[embeddedInput]()
	for _, name := range []string{"id", "name", "email"} {
		if _, ok := generated.Properties[name]; !ok {
			t.Errorf("properties missing %q: %s", name, generated.String())
		}
	}
	if _, ok := generated.Properties["embeddedBase"]; ok {
		t.Errorf("unexported embedded type should not become a property: %s", generated.String())
	}
	if _, ok := generated.Properties["embeddedExported"]; ok {
		t.Errorf("anonymous embedded type should not become a property: %s", generated.String())
	}

	required := make(map[string]bool, len(generated.Required))
	for _, name := range generated.Required {
		required[name] = true
	}
	if !required["id"] {
		t.Errorf("embedded required field was not merged: %v", generated.Required)
	}

	payload, err := json.Marshal(embeddedInput{})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var instance map[string]any
	if err := json.Unmarshal(payload, &instance); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	for name := range instance {
		if _, ok := generated.Properties[name]; !ok {
			t.Errorf("actual JSON field %q is not declared in generated schema: %s", name, generated.String())
		}
	}
}

func TestOfKeepsExplicitlyNamedEmbeddedField(t *testing.T) {
	t.Parallel()

	generated := Of[namedEmbeddedInput]()
	if _, ok := generated.Properties["profile"]; !ok {
		t.Fatalf("named embedded field should be a property: %s", generated.String())
	}
	if _, ok := generated.Properties["name"]; ok {
		t.Fatalf("named embedded field should not be flattened: %s", generated.String())
	}
}

func TestOfOmitsConflictingEmbeddedFields(t *testing.T) {
	t.Parallel()

	conflictType := reflect.StructOf([]reflect.StructField{
		{Name: "ConflictA", Type: reflect.TypeOf(ConflictA{}), Anonymous: true},
		{Name: "ConflictB", Type: reflect.TypeOf(ConflictB{}), Anonymous: true},
		{Name: "Name", Type: reflect.TypeOf(""), Tag: `json:"name"`},
	})
	generated := FromType(conflictType)
	if _, ok := generated.Properties["value"]; ok {
		t.Fatalf("ambiguous embedded field should be omitted: %s", generated.String())
	}
	for _, name := range generated.Required {
		if name == "value" {
			t.Fatalf("ambiguous embedded field should not be required: %v", generated.Required)
		}
	}
	if _, ok := generated.Properties["name"]; !ok {
		t.Fatalf("non-conflicting fields should remain: %s", generated.String())
	}

	payload, err := json.Marshal(reflect.New(conflictType).Interface())
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var instance map[string]any
	if err := json.Unmarshal(payload, &instance); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if _, ok := instance["value"]; ok {
		t.Fatalf("encoding/json unexpectedly emitted ambiguous field: %s", payload)
	}
}
