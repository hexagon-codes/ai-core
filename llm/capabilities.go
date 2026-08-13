package llm

// 本文件按接口隔离原则（ISP）把完整 Provider 拆成若干窄能力接口。
//
// 完整 Provider 是一个胖接口（Complete/Stream/Models/CountTokens），但许多消费方
// 只用其中一两个能力（如结构化输出只需 Complete）。提供窄接口后，这些消费方可只
// 依赖所需能力，而非被迫接受整个 Provider。完整 Provider 是各窄接口的超集，既有
// Provider 实现无需任何改动即可满足这些窄接口（加法、零行为变更）。

import (
	"context"
	"errors"
)

// ErrContextTokenCountingUnsupported 表示包装层无法向底层传递可取消的 Token 计数。
var ErrContextTokenCountingUnsupported = errors.New("context-aware token counting is not supported")

// Completer 仅提供非流式补全能力。
type Completer interface {
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
}

// Streamer 仅提供流式补全能力。
type Streamer interface {
	Stream(ctx context.Context, req CompletionRequest) (*Stream, error)
}

// ChatProvider 提供聊天能力（补全 + 流式），不含模型列举 / Token 计数 / 嵌入。
type ChatProvider interface {
	Completer
	Streamer
}

// TokenCounter 仅提供 Token 计数能力。
//
// 该接口保留无 context 的既有签名，以兼容已发布的 Provider 实现。
// 需要在计数期间响应取消或截止时间的实现可额外实现 ContextTokenCounter。
type TokenCounter interface {
	CountTokens(messages []Message) (int, error)
}

// ContextTokenCounter 是 TokenCounter 的可取消扩展能力。
//
// 实现应在计数过程中检查 ctx，并在取消或超时时返回 ctx.Err()。
// 这是可选接口，不会改变既有 Provider 或 TokenCounter 的方法集。
type ContextTokenCounter interface {
	TokenCounter
	CountTokensContext(ctx context.Context, messages []Message) (int, error)
}

// CountTokensContext 使用 context-aware 能力计数，并兼容旧版 TokenCounter。
//
// 若 counter 实现 ContextTokenCounter，则原样传递 ctx；否则先检查 ctx，
// 再同步调用旧 CountTokens。旧调用一旦开始便无法被该函数强制取消，需在调用期间
// 响应取消的实现应实现 ContextTokenCounter。该函数不会用后台 goroutine 模拟取消。
func CountTokensContext(
	ctx context.Context,
	counter TokenCounter,
	messages []Message,
) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if contextCounter, ok := counter.(ContextTokenCounter); ok {
		return contextCounter.CountTokensContext(ctx, messages)
	}
	return counter.CountTokens(messages)
}

// ModelLister 仅提供模型列举能力。
type ModelLister interface {
	Models() []ModelInfo
}

// Named 仅提供名称。
type Named interface {
	Name() string
}

// 编译期断言：完整 Provider 是各窄能力接口的超集，
// 任意 Provider 实现都自动满足这些窄接口。
var (
	_ Completer    = (Provider)(nil)
	_ Streamer     = (Provider)(nil)
	_ ChatProvider = (Provider)(nil)
	_ TokenCounter = (Provider)(nil)
	_ TokenCounter = ContextTokenCounter(nil)
	_ ModelLister  = (Provider)(nil)
	_ Named        = (Provider)(nil)
	_ Completer    = (EmbeddingProvider)(nil)
)
