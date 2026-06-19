package anthropic

import "testing"

// TestParseResponse_CacheTokens 验证 Anthropic 响应中的缓存 Token 字段
// 被正确映射到 llm.Usage 的四维度，且 TotalTokens 语义保持
// "非缓存输入 + 输出" 不变。
func TestParseResponse_CacheTokens(t *testing.T) {
	p := New("test")

	resp := &anthropicResponse{ID: "msg_1", Model: "claude-test"}
	resp.Usage.InputTokens = 100
	resp.Usage.OutputTokens = 50
	resp.Usage.CacheCreationInputTokens = 20
	resp.Usage.CacheReadInputTokens = 30

	got := p.parseResponse(resp, "")

	if got.Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", got.Usage.PromptTokens)
	}
	if got.Usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", got.Usage.CompletionTokens)
	}
	// TotalTokens 不把缓存 Token 计入，仍为 input+output
	if got.Usage.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150 (input+output, 缓存维度不计入)", got.Usage.TotalTokens)
	}
	if got.Usage.CacheCreationTokens != 20 {
		t.Errorf("CacheCreationTokens = %d, want 20", got.Usage.CacheCreationTokens)
	}
	if got.Usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", got.Usage.CacheReadTokens)
	}
}

// TestParseResponse_NoCacheTokens 验证未启用提示词缓存时（响应无缓存字段），
// 两个缓存维度为零值，行为与改动前一致。
func TestParseResponse_NoCacheTokens(t *testing.T) {
	p := New("test")

	resp := &anthropicResponse{ID: "msg_2", Model: "claude-test"}
	resp.Usage.InputTokens = 80
	resp.Usage.OutputTokens = 40

	got := p.parseResponse(resp, "")

	if got.Usage.CacheCreationTokens != 0 || got.Usage.CacheReadTokens != 0 {
		t.Errorf("无缓存时缓存维度应为 0, got creation=%d read=%d",
			got.Usage.CacheCreationTokens, got.Usage.CacheReadTokens)
	}
	if got.Usage.TotalTokens != 120 {
		t.Errorf("TotalTokens = %d, want 120", got.Usage.TotalTokens)
	}
}
