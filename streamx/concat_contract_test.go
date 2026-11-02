package streamx

import (
	"errors"
	"testing"
)

func TestConcatPreservesAllDeltas(t *testing.T) {
	value, err := ConcatItems([]string{"a", "b", "c"})
	if err != nil || value != "abc" {
		t.Fatalf("value=%q error=%v", value, err)
	}
	type opaque struct{ Value int }
	if _, err := ConcatItems([]opaque{{1}, {2}}); !errors.Is(err, ErrNoConcatFunc) {
		t.Fatalf("unregistered merge silently discarded data: %v", err)
	}
}

func TestConcatRegistrySeparatesConcreteTypes(t *testing.T) {
	type first struct{ Text string }
	type second struct{ Text string }
	RegisterConcatFunc(func(items []first) (first, error) { return first{items[0].Text + items[1].Text}, nil })
	RegisterConcatFunc(func(items []second) (second, error) { return second{items[0].Text + ":" + items[1].Text}, nil })
	a, err := ConcatItems([]first{{"a"}, {"b"}})
	if err != nil || a.Text != "ab" {
		t.Fatalf("first result=%v error=%v", a, err)
	}
	b, err := ConcatItems([]second{{"a"}, {"b"}})
	if err != nil || b.Text != "a:b" {
		t.Fatalf("second result=%v error=%v", b, err)
	}
}
