package openai

import "testing"

// TestParseResponse_CachedTokens 验证 OpenAI 响应的
// prompt_tokens_details.cached_tokens 被映射到 llm.Usage.CacheReadTokens。
// OpenAI 无缓存写入概念，故 CacheCreationTokens 恒为 0。
func TestParseResponse_CachedTokens(t *testing.T) {
	p := New("test")

	resp := &openAIResponse{ID: "id_1", Model: "gpt-test"}
	resp.Usage.PromptTokens = 200
	resp.Usage.CompletionTokens = 80
	resp.Usage.TotalTokens = 280
	resp.Usage.PromptTokensDetails.CachedTokens = 64

	got := p.parseResponse(resp)

	if got.Usage.PromptTokens != 200 {
		t.Errorf("PromptTokens = %d, want 200", got.Usage.PromptTokens)
	}
	if got.Usage.CompletionTokens != 80 {
		t.Errorf("CompletionTokens = %d, want 80", got.Usage.CompletionTokens)
	}
	if got.Usage.TotalTokens != 280 {
		t.Errorf("TotalTokens = %d, want 280", got.Usage.TotalTokens)
	}
	if got.Usage.CacheReadTokens != 64 {
		t.Errorf("CacheReadTokens = %d, want 64", got.Usage.CacheReadTokens)
	}
	if got.Usage.CacheCreationTokens != 0 {
		t.Errorf("CacheCreationTokens = %d, want 0 (OpenAI 无缓存写入概念)", got.Usage.CacheCreationTokens)
	}
}

// TestParseResponse_NoCachedTokens 验证响应无缓存细分时 CacheReadTokens 为零值。
func TestParseResponse_NoCachedTokens(t *testing.T) {
	p := New("test")

	resp := &openAIResponse{ID: "id_2", Model: "gpt-test"}
	resp.Usage.PromptTokens = 10
	resp.Usage.CompletionTokens = 5
	resp.Usage.TotalTokens = 15

	got := p.parseResponse(resp)

	if got.Usage.CacheReadTokens != 0 {
		t.Errorf("无缓存命中时 CacheReadTokens 应为 0, got %d", got.Usage.CacheReadTokens)
	}
}
