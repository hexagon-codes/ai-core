package llm

import (
	"context"
	"testing"
)

// completerOnly 只实现 Complete，用于验证消费方可只依赖窄接口 Completer。
type completerOnly struct{ reply string }

func (c *completerOnly) Complete(_ context.Context, _ CompletionRequest) (*CompletionResponse, error) {
	return &CompletionResponse{Content: c.reply}, nil
}

// useCompleter 是只依赖 Completer 窄接口的消费方。
func useCompleter(ctx context.Context, c Completer, q string) (string, error) {
	resp, err := c.Complete(ctx, CompletionRequest{Messages: []Message{UserMessage(q)}})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// TestCompleter_NarrowDependency 验证只实现 Complete 的类型即可被只依赖 Completer 的消费方使用。
func TestCompleter_NarrowDependency(t *testing.T) {
	got, err := useCompleter(context.Background(), &completerOnly{reply: "ok"}, "hi")
	if err != nil {
		t.Fatalf("useCompleter: %v", err)
	}
	if got != "ok" {
		t.Errorf("got %q, want ok", got)
	}
}
