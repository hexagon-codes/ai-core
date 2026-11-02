package llm

import (
	"context"
	"fmt"
)

// CallProvider 是调用所需的最小模型端口，不要求模型目录或其他管理能力。
type CallProvider interface {
	Name() string
	Complete(context.Context, CompletionRequest) (*CompletionResponse, error)
	Stream(context.Context, CompletionRequest) (*Stream, error)
}

// CompletionInvoker 由上层运行时绑定实际调用、预算与用量；必须使用传入的 Provider。
type CompletionInvoker interface {
	InvokeComplete(context.Context, CallProvider, CompletionRequest, string) (*CompletionResponse, error)
}

type CompletionInvokerFunc func(context.Context, CallProvider, CompletionRequest, string) (*CompletionResponse, error)

func (f CompletionInvokerFunc) InvokeComplete(ctx context.Context, provider CallProvider, request CompletionRequest, purpose string) (*CompletionResponse, error) {
	return f(ctx, provider, request, purpose)
}

type completionInvokerKey struct{}

func WithCompletionInvoker(ctx context.Context, invoker CompletionInvoker) context.Context {
	return context.WithValue(ctx, completionInvokerKey{}, invoker)
}

// InvokeComplete 在有运行宿主时进入统一调用边界；独立库调用保留直接 Provider 行为。
// purpose 用于区分摘要、结构化修正等事实，不改变模型请求协议。
func InvokeComplete(ctx context.Context, provider CallProvider, request CompletionRequest, purpose string) (*CompletionResponse, error) {
	if provider == nil {
		return nil, fmt.Errorf("LLM provider is nil")
	}
	if invoker, ok := ctx.Value(completionInvokerKey{}).(CompletionInvoker); ok && invoker != nil {
		return invoker.InvokeComplete(ctx, provider, request, purpose)
	}
	return provider.Complete(ctx, request)
}
