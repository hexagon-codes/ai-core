package ollama

// BUG-20260710：ollama 客户端不下发 keep_alive → 受 Ollama 服务端默认 5 分钟摆布。
//
// 真机取证（hexclaw 桌面 · qwen3.5:9b 纯 CPU）：应用一次请求 prompt≈7943 token，
// prefill 23 tok/s → 冷路径 344s；KV 前缀缓存命中后热路径 46s。但空闲 5 分钟模型
// 即卸载、KV 缓存全丢 → 用户每次隔几分钟再聊都踩冷路径（体感「走应用永远 6 分钟」）。
//
// 修复契约（本测试断言正确行为，未修复时 FAIL 即证明缺口）：
//   1. 请求体默认携带 keep_alive（默认 30m，让模型 + KV 缓存驻留），每次请求刷新驻留窗口；
//   2. 可经 WithKeepAlive 选项覆盖；
//   3. metadata["keep_alive"]（string）可按请求覆盖。

import (
	"encoding/json"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func keepAliveOf(t *testing.T, p *Provider, req llm.CompletionRequest) any {
	t.Helper()
	body, err := p.buildRequestBody(req, false)
	if err != nil {
		t.Fatalf("buildRequestBody: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	v, ok := payload["keep_alive"]
	if !ok {
		t.Fatalf("请求体缺 keep_alive（Ollama 默认 5m 卸载模型 → KV 缓存丢失，CPU 冷路径 344s 反复重演）；payload keys=%v", keysOf(payload))
	}
	return v
}

func keysOf(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestBUG20260710_KeepAlive_DefaultPresent(t *testing.T) {
	p := New(WithModel("qwen3.5:9b"))
	got := keepAliveOf(t, p, llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if got != "30m" {
		t.Fatalf("默认 keep_alive 应为 30m，got %v", got)
	}
}

func TestBUG20260710_KeepAlive_OptionOverride(t *testing.T) {
	p := New(WithModel("qwen3.5:9b"), WithKeepAlive("2h"))
	got := keepAliveOf(t, p, llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if got != "2h" {
		t.Fatalf("WithKeepAlive 应覆盖默认值，got %v", got)
	}
}

func TestBUG20260710_KeepAlive_MetadataOverride(t *testing.T) {
	p := New(WithModel("qwen3.5:9b"))
	got := keepAliveOf(t, p, llm.CompletionRequest{
		Model:    "qwen3.5:9b",
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
		Metadata: map[string]any{"keep_alive": "1h"},
	})
	if got != "1h" {
		t.Fatalf("metadata.keep_alive 应按请求覆盖，got %v", got)
	}
}
