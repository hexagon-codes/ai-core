package streamx

import (
	"io"
	"strings"
	"sync"
	"testing"
)

type trackingReadCloser struct {
	io.Reader
	mu     sync.Mutex
	closed bool
}

func (r *trackingReadCloser) Close() error {
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
	return nil
}

func (r *trackingReadCloser) isClosed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.closed
}

func TestCollectClosesUnderlyingBody(t *testing.T) {
	t.Parallel()

	body := &trackingReadCloser{Reader: strings.NewReader(
		"data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n",
	)}
	stream := NewStream(body, OpenAIFormat)
	defer stream.Close()

	if _, err := stream.Collect(); err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if !body.isClosed() {
		t.Fatal("Collect() did not close the underlying body")
	}
}
