package streamx

import (
	"errors"
	"io"
	"testing"
	"time"
)

type jsonTerminalValue struct {
	Value int `json:"value"`
}

func TestJSONParserTerminalContract(t *testing.T) {
	parsers := []struct {
		name  string
		input string
		parse func(*StreamReader[string]) *StreamReader[jsonTerminalValue]
	}{
		{"object", `{"value":42}`, func(source *StreamReader[string]) *StreamReader[jsonTerminalValue] {
			return ParseJSON[jsonTerminalValue](source)
		}},
		{"array", `[{"value":42}]`, func(source *StreamReader[string]) *StreamReader[jsonTerminalValue] {
			return ParseJSONArray[jsonTerminalValue](source)
		}},
		{"structured", `{"value":42}`, func(source *StreamReader[string]) *StreamReader[jsonTerminalValue] {
			return ParseStructured[jsonTerminalValue](source, nil)
		}},
	}
	for _, parser := range parsers {
		t.Run(parser.name+"/value_then_eof", func(t *testing.T) {
			reader := parser.parse(FromSlice([]string{parser.input}))
			defer reader.Close()
			value, err := reader.Recv()
			if err != nil || value.Value != 42 {
				t.Fatalf("value=%+v error=%v, want value 42", value, err)
			}
			if _, err := reader.Recv(); !errors.Is(err, io.EOF) {
				t.Fatalf("terminal error=%v, want EOF", err)
			}
		})
		t.Run(parser.name+"/source_error", func(t *testing.T) {
			source, writer := Pipe[string](0)
			reader := parser.parse(source)
			defer reader.Close()
			cause := errors.New("source unavailable")
			if err := writer.CloseWithError(cause); err != nil {
				t.Fatal(err)
			}
			if _, err := reader.Recv(); !errors.Is(err, cause) {
				t.Fatalf("terminal error=%v, want original source error", err)
			}
		})
		t.Run(parser.name+"/close_unblocks_source", func(t *testing.T) {
			source, writer := Pipe[string](0)
			reader := parser.parse(source)
			if err := reader.Close(); err != nil {
				t.Fatal(err)
			}
			if err := reader.Close(); err != nil {
				t.Fatal(err)
			}
			finished := make(chan error, 1)
			go func() { finished <- writer.Send(parser.input) }()
			select {
			case err := <-finished:
				if !errors.Is(err, ErrStreamClosed) {
					t.Fatalf("send after consumer close=%v, want closed stream", err)
				}
			case <-time.After(time.Second):
				t.Fatal("source writer remained blocked after parsed stream closed")
			}
		})
	}
}

func TestParseJSONBufferLimitReturnsError(t *testing.T) {
	reader := ParseJSON[jsonTerminalValue](FromSlice([]string{`{"value":42}`}), &JSONParseConfig{MaxBufferSize: 4})
	defer reader.Close()
	if _, err := reader.Recv(); err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("buffer limit returned %v, want an explicit error", err)
	}
}
