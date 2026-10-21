package streamx

import (
	"strings"
	"testing"
)

// usage 是上游累计快照，重复帧不累加，空 choices 的最终用量也不能丢失。
func TestOpenAIStreamRetainsUsageSnapshots(t *testing.T) {
	stream := NewStream(strings.NewReader(
		"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4,\"total_tokens\":13}}\n\n"+
			"data: [DONE]\n\n"), OpenAIFormat)
	defer stream.Close()
	for range stream.Chunks() {
	}
	result := stream.Result()
	if result.Content != "ok" || result.Usage.PromptTokens != 9 || result.Usage.CompletionTokens != 4 || result.Usage.TotalTokens != 13 {
		t.Fatalf("reported usage lost or accumulated twice: %+v", result)
	}
}
