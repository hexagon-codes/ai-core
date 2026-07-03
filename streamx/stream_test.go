package streamx

import (
	"context"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestStream_OpenAI(t *testing.T) {
	input := `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}

	if result.Content != "Hello World" {
		t.Errorf("expected 'Hello World', got '%s'", result.Content)
	}
	if result.ID != "chatcmpl-123" {
		t.Errorf("expected ID 'chatcmpl-123', got '%s'", result.ID)
	}
	if result.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got '%s'", result.Model)
	}
	if result.FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got '%s'", result.FinishReason)
	}
}

func TestStream_Chunks(t *testing.T) {
	input := `data: {"id":"1","choices":[{"index":0,"delta":{"content":"A"}}]}

data: {"id":"1","choices":[{"index":0,"delta":{"content":"B"}}]}

data: {"id":"1","choices":[{"index":0,"delta":{"content":"C"}}]}

data: [DONE]

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)

	var chunks []*Chunk
	for chunk := range stream.Chunks() {
		chunks = append(chunks, chunk)
	}

	if len(chunks) != 3 {
		t.Errorf("expected 3 chunks, got %d", len(chunks))
	}

	contents := ""
	for _, c := range chunks {
		contents += c.Content
	}
	if contents != "ABC" {
		t.Errorf("expected 'ABC', got '%s'", contents)
	}
}

func TestStream_Callbacks(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" World"}}]}

data: [DONE]

`
	var chunkCount int
	var doneResult *Result
	var gotError error

	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	stream.OnChunk(func(c *Chunk) {
		chunkCount++
	}).OnDone(func(r *Result) {
		doneResult = r
	}).OnError(func(err error) {
		gotError = err
	}).Start()

	// 等待完成
	<-stream.Done()

	if chunkCount != 2 {
		t.Errorf("expected 2 chunks, got %d", chunkCount)
	}
	if doneResult == nil {
		t.Error("expected done callback to be called")
	}
	if doneResult.Content != "Hello World" {
		t.Errorf("expected 'Hello World', got '%s'", doneResult.Content)
	}
	if gotError != nil {
		t.Errorf("unexpected error: %v", gotError)
	}
}

func TestStream_Context(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Start"}}]}

data: [DONE]

`
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	stream := NewStreamWithContext(ctx, strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()

	// 应该成功完成（在超时前）
	if err != nil {
		t.Logf("got error (may be expected): %v", err)
	}

	if result.Content != "Start" {
		t.Errorf("expected 'Start', got '%s'", result.Content)
	}
}

func TestStream_Close(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Test"}}]}

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	stream.Start()

	// 读取一些数据
	<-stream.Chunks()

	// 关闭
	err := stream.Close()
	if err != nil {
		t.Errorf("close error: %v", err)
	}

	// 再次关闭应该无错误
	err = stream.Close()
	if err != nil {
		t.Errorf("second close error: %v", err)
	}
}

func TestStreamCloseUnblocksBlockedRead(t *testing.T) {
	r := newBlockingReadCloser()
	stream := NewStream(r, OpenAIFormat)
	stream.Start()
	<-r.readStarted

	closed := make(chan error, 1)
	go func() {
		closed <- stream.Close()
	}()

	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close returned error: %v", err)
		}
	case <-time.After(150 * time.Millisecond):
		_ = r.Close()
		t.Fatal("Close blocked while processLoop was waiting for input")
	}
}

type blockingReadCloser struct {
	readStarted chan struct{}
	closed      chan struct{}
	once        sync.Once
}

func newBlockingReadCloser() *blockingReadCloser {
	return &blockingReadCloser{
		readStarted: make(chan struct{}),
		closed:      make(chan struct{}),
	}
}

func (r *blockingReadCloser) Read([]byte) (int, error) {
	r.once.Do(func() {
		close(r.readStarted)
	})
	<-r.closed
	return 0, io.ErrClosedPipe
}

func (r *blockingReadCloser) Close() error {
	select {
	case <-r.closed:
	default:
		close(r.closed)
	}
	return nil
}

func TestStream_ToolCalls(t *testing.T) {
	input := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"arguments":"{\"location\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"arguments":"\"Beijing\"}"}}]}}]}

data: [DONE]

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("collect error: %v", err)
	}

	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}

	tc := result.ToolCalls[0]
	if tc.ID != "call_123" {
		t.Errorf("expected ID 'call_123', got '%s'", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got '%s'", tc.Name)
	}
	if tc.Arguments != `{"location":"Beijing"}` {
		t.Errorf("expected arguments '{\"location\":\"Beijing\"}', got '%s'", tc.Arguments)
	}
}

func TestCollectContent(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" World"}}]}

data: [DONE]

`
	content, err := CollectContent(strings.NewReader(input), OpenAIFormat)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if content != "Hello World" {
		t.Errorf("expected 'Hello World', got '%s'", content)
	}
}

func TestProcessStream(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"A"}}]}

data: {"choices":[{"delta":{"content":"B"}}]}

data: {"choices":[{"delta":{"content":"C"}}]}

data: [DONE]

`
	var contents []string
	err := ProcessStream(strings.NewReader(input), OpenAIFormat, func(c *Chunk) error {
		contents = append(contents, c.Content)
		return nil
	})

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(contents) != 3 {
		t.Errorf("expected 3 contents, got %d", len(contents))
	}
}

func TestOpenAIParser(t *testing.T) {
	parser := &OpenAIParser{}

	data := `{"id":"chatcmpl-123","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`

	chunk, err := parser.Parse([]byte(data))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if chunk.ID != "chatcmpl-123" {
		t.Errorf("expected ID 'chatcmpl-123', got '%s'", chunk.ID)
	}
	if chunk.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got '%s'", chunk.Model)
	}
	if chunk.Role != "assistant" {
		t.Errorf("expected role 'assistant', got '%s'", chunk.Role)
	}
	if chunk.Content != "Hello" {
		t.Errorf("expected content 'Hello', got '%s'", chunk.Content)
	}
}

func TestOpenAIParser_IsDone(t *testing.T) {
	parser := &OpenAIParser{}

	if !parser.IsDone([]byte("[DONE]")) {
		t.Error("expected true for [DONE]")
	}
	if parser.IsDone([]byte(`{"choices":[]}`)) {
		t.Error("expected false for normal data")
	}
}

func TestClaudeParser(t *testing.T) {
	parser := &ClaudeParser{}

	// message_start
	data1 := `{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3"}}`
	chunk1, err := parser.Parse([]byte(data1))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if chunk1.ID != "msg_123" {
		t.Errorf("expected ID 'msg_123', got '%s'", chunk1.ID)
	}
	if chunk1.Role != "assistant" {
		t.Errorf("expected role 'assistant', got '%s'", chunk1.Role)
	}

	// content_block_delta
	data2 := `{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}`
	chunk2, err := parser.Parse([]byte(data2))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if chunk2.Content != "Hello" {
		t.Errorf("expected content 'Hello', got '%s'", chunk2.Content)
	}

	// message_stop
	data3 := `{"type":"message_stop"}`
	if !parser.IsDone([]byte(data3)) {
		t.Error("expected true for message_stop")
	}
}

func TestGeminiParser(t *testing.T) {
	parser := &GeminiParser{}

	data := `{"candidates":[{"content":{"parts":[{"text":"Hello World"}],"role":"model"},"finishReason":"STOP"}]}`

	chunk, err := parser.Parse([]byte(data))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if chunk.Content != "Hello World" {
		t.Errorf("expected content 'Hello World', got '%s'", chunk.Content)
	}
	if chunk.Role != "model" {
		t.Errorf("expected role 'model', got '%s'", chunk.Role)
	}
	if chunk.FinishReason != "STOP" {
		t.Errorf("expected finish_reason 'STOP', got '%s'", chunk.FinishReason)
	}
}

func TestJSONParser(t *testing.T) {
	parser := &JSONParser{
		ContentPath: "choices.0.delta.content",
		DoneValue:   "[DONE]",
	}

	data := `{"choices":[{"delta":{"content":"Test"}}]}`
	chunk, err := parser.Parse([]byte(data))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if chunk.Content != "Test" {
		t.Errorf("expected content 'Test', got '%s'", chunk.Content)
	}

	if !parser.IsDone([]byte("[DONE]")) {
		t.Error("expected true for [DONE]")
	}
}

func TestStream_Result(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"Hi"}}]}

data: [DONE]

`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)
	stream.Start()

	result := stream.Result()
	if result.Content != "Hi" {
		t.Errorf("expected 'Hi', got '%s'", result.Content)
	}
}

func TestNewStreamWithParser(t *testing.T) {
	parser := &JSONParser{
		ContentPath: "text",
		DoneValue:   "END",
	}

	input := `data: {"text":"Custom"}

data: END

`
	stream := NewStreamWithParser(strings.NewReader(input), parser)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if result.Content != "Custom" {
		t.Errorf("expected 'Custom', got '%s'", result.Content)
	}
}

func TestNewStreamWithParser_CustomRawJSONLines(t *testing.T) {
	parser := &JSONParser{
		ContentPath: "text",
		DoneValue:   "END",
	}

	input := "{\"text\":\"Raw\"}\nEND\n"
	stream := NewStreamWithParser(strings.NewReader(input), parser)
	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if result.Content != "Raw" {
		t.Errorf("expected 'Raw', got '%s'", result.Content)
	}
}
