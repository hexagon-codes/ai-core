package llm

// 本文件按接口隔离原则（ISP）把完整 Provider 拆成若干窄能力接口。
//
// 完整 Provider 是一个胖接口（Complete/Stream/Models/CountTokens），但许多消费方
// 只用其中一两个能力（如结构化输出只需 Complete）。提供窄接口后，这些消费方可只
// 依赖所需能力，而非被迫接受整个 Provider。完整 Provider 是各窄接口的超集，既有
// Provider 实现无需任何改动即可满足这些窄接口（加法、零行为变更）。

import "context"

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
type TokenCounter interface {
	CountTokens(messages []Message) (int, error)
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
	_ ModelLister  = (Provider)(nil)
	_ Named        = (Provider)(nil)
	_ Completer    = (EmbeddingProvider)(nil)
)
