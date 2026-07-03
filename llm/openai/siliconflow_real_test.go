package openai

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestRealSiliconFlowOpenAICompatibleProvider(t *testing.T) {
	if os.Getenv("AICORE_REAL_LLM_EVAL") != "1" {
		t.Skip("set AICORE_REAL_LLM_EVAL=1 to run the real SiliconFlow OpenAI-compatible provider test")
	}
	key := strings.TrimSpace(os.Getenv("AICORE_SILICONFLOW_API_KEY"))
	if key == "" {
		t.Skip("set AICORE_SILICONFLOW_API_KEY to run the real SiliconFlow OpenAI-compatible provider test")
	}

	baseURL := envOr("AICORE_SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
	model := envOr("AICORE_SILICONFLOW_MODEL", "Qwen/Qwen3.6-35B-A3B")
	p := New(key,
		WithBaseURL(baseURL),
		WithModel(model),
		WithRequestTimeout(120*time.Second),
		WithStreamIdleTimeout(90*time.Second),
	)

	req := llm.CompletionRequest{
		Model:     model,
		MaxTokens: 32,
		Metadata:  map[string]any{"enable_thinking": false},
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a protocol conformance test. Reply with the requested marker only."},
			{Role: llm.RoleUser, Content: "Reply exactly: AICORE_REAL_OK"},
		},
	}

	t.Run("complete", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
		defer cancel()

		resp, err := p.Complete(ctx, req)
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		assertRealMarker(t, resp.Content)
		if resp.Model == "" {
			t.Fatalf("expected response model to be populated")
		}
	})

	t.Run("stream", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
		defer cancel()

		stream, err := p.Stream(ctx, req)
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		result, err := stream.Collect()
		if err != nil {
			t.Fatalf("Collect: %v", err)
		}
		assertRealMarker(t, result.Content)
	})
}

func assertRealMarker(t *testing.T, content string) {
	t.Helper()
	content = strings.TrimSpace(content)
	if content == "" {
		t.Fatalf("real provider returned empty content")
	}
	if !strings.Contains(content, "AICORE_REAL_OK") {
		t.Fatalf("real provider content %q does not contain expected marker", content)
	}
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
