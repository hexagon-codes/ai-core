package schema

import (
	"encoding/json"
	"strings"
	"testing"
)

type recursiveNode struct {
	Name     string          `json:"name"`
	Children []recursiveNode `json:"children,omitempty"`
}

func TestRecursiveSchemaEmitsResolvableReference(t *testing.T) {
	s := Of[recursiveNode]()
	encoded, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	ref := s.Properties["children"].Items.Ref
	if !strings.HasPrefix(ref, "#/$defs/") {
		t.Fatalf("recursive child has no local definition reference: %s", encoded)
	}
	definition := s.Defs[strings.TrimPrefix(ref, "#/$defs/")]
	if definition == nil || definition.Properties["name"].Type != "string" {
		t.Fatalf("recursive reference does not resolve: %s", encoded)
	}
}

type mutualA struct {
	Child *mutualB `json:"child"`
}
type mutualB struct {
	Parent *mutualA `json:"parent"`
}

func TestMutuallyRecursiveSchemaSerializes(t *testing.T) {
	s := Of[mutualA]()
	if _, err := json.Marshal(s); err != nil {
		t.Fatal(err)
	}
	if len(s.Defs) == 0 {
		t.Fatal("mutual recursion has no finite reference definition")
	}
}
