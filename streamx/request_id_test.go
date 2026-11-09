package streamx

import (
	"strings"
	"testing"
)

func TestSetRequestIDIsExposedOnResult(t *testing.T) {
	stream := NewStream(strings.NewReader(
		"data: {\"id\":\"resp\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: [DONE]\n\n",
	), OpenAIFormat)
	stream.SetRequestID("req-stream-1")
	if _, err := stream.Collect(); err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if got := stream.Result().RequestID; got != "req-stream-1" {
		t.Fatalf("Result().RequestID = %q, want req-stream-1", got)
	}
}
