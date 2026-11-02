package streamx

import (
	"fmt"
	"strings"
	"testing"
)

func TestClaudeFinalFramePreservesActualTermination(t *testing.T) {
	for _, reason := range []string{"end_turn", "stop_sequence", "tool_use", "max_tokens", "refusal", "pause_turn", ""} {
		t.Run(reason, func(t *testing.T) {
			input := "data: {\"type\":\"message_start\",\"message\":{\"id\":\"message\"}}\n\n"
			if reason != "" {
				input += fmt.Sprintf("data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":%q}}\n\n", reason)
			}
			input += "data: {\"type\":\"message_stop\"}\n\n"
			response, err := NewStream(strings.NewReader(input), ClaudeFormat).Collect()
			if err != nil {
				t.Fatal(err)
			}
			want := reason
			if want == "" {
				want = "unknown"
			}
			if response.FinishReason != want {
				t.Fatalf("finish reason=%q, want %q", response.FinishReason, want)
			}
		})
	}
}
