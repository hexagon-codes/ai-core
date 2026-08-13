package llm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/llm/anthropic"
	"github.com/hexagon-codes/ai-core/llm/ark"
	"github.com/hexagon-codes/ai-core/llm/compatible"
	"github.com/hexagon-codes/ai-core/llm/deepseek"
	"github.com/hexagon-codes/ai-core/llm/ernie"
	"github.com/hexagon-codes/ai-core/llm/gemini"
	"github.com/hexagon-codes/ai-core/llm/ollama"
	"github.com/hexagon-codes/ai-core/llm/openai"
	"github.com/hexagon-codes/ai-core/llm/qwen"
)

// TestOfficialProviders_CountTokensContextContract 锁定官方 Provider 的可取消计数契约。
func TestOfficialProviders_CountTokensContextContract(t *testing.T) {
	providers := map[string]llm.TokenCounter{
		"anthropic": anthropic.New("key"),
		"ark":       ark.New("key"),
		"compatible": compatible.New(
			"compatible",
			"key",
		),
		"deepseek": deepseek.New("key"),
		"ernie":    ernie.New("key", "secret"),
		"gemini":   gemini.New("key"),
		"ollama":   ollama.New(),
		"openai":   openai.New("key"),
		"qwen":     qwen.New("key"),
	}
	messages := []llm.Message{llm.UserMessage("context-aware token counting")}

	for name, provider := range providers {
		t.Run(name, func(t *testing.T) {
			counter, ok := provider.(llm.ContextTokenCounter)
			if !ok {
				t.Fatalf("%T does not implement llm.ContextTokenCounter", provider)
			}

			want, err := provider.CountTokens(messages)
			if err != nil {
				t.Fatalf("CountTokens() error = %v", err)
			}
			got, err := counter.CountTokensContext(context.Background(), messages)
			if err != nil || got != want {
				t.Fatalf("CountTokensContext() = (%d, %v), want (%d, nil)", got, err, want)
			}

			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			got, err = counter.CountTokensContext(ctx, messages)
			if got != 0 || !errors.Is(err, context.Canceled) {
				t.Fatalf("CountTokensContext(canceled) = (%d, %v), want (0, context.Canceled)", got, err)
			}
		})
	}
}
