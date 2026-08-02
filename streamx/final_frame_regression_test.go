package streamx

import (
	"strings"
	"testing"
)

func TestCollectProcessesFinalSSEFrameWithoutTrailingNewline(t *testing.T) {
	t.Parallel()

	input := `data: {"id":"answer-1","choices":[{"delta":{"content":"final token"},"finish_reason":"stop"}]}`
	stream := NewStream(strings.NewReader(input), OpenAIFormat)

	result, err := stream.Collect()
	if err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if result.Content != "final token" {
		t.Fatalf("Content = %q, want final frame content", result.Content)
	}
	if result.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", result.FinishReason)
	}
}
