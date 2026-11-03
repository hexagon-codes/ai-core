package streamx

import (
	"context"
	"reflect"
	"testing"
	"time"
)

func TestMergeAdaptersKeepFollowingSource(t *testing.T) {
	adapters := []struct {
		name string
		wrap func(*StreamReader[string]) *StreamReader[string]
	}{
		{"buffer", func(r *StreamReader[string]) *StreamReader[string] { return Buffer(r, 2) }},
		{"timeout", func(r *StreamReader[string]) *StreamReader[string] { return Timeout(r, time.Second) }},
	}
	for _, a := range adapters {
		t.Run(a.name, func(t *testing.T) {
			r := a.wrap(Merge(FromSlice([]string{}), FromSlice([]string{"tail"})))
			defer r.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			values, err := r.Collect(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(values, []string{"tail"}) {
				t.Fatalf("values=%v, want tail", values)
			}
		})
	}
}
