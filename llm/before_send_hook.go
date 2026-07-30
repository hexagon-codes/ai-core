package llm

import (
	"context"

	"github.com/hexagon-codes/ai-core/transport"
)

// BeforeSendHook is invoked by the shared provider transport immediately
// before one physical HTTP attempt.
type BeforeSendHook = transport.BeforeSendHook

// WithBeforeSendHook installs an out-of-band durable authorization hook. It
// does not add fields to CompletionRequest or provider wire payloads.
func WithBeforeSendHook(
	ctx context.Context,
	hook BeforeSendHook,
) context.Context {
	return transport.WithBeforeSendHook(ctx, hook)
}

// WithBeforeSendHookForAction limits the hook to the named shared-transport
// action so provider-internal auxiliary requests cannot consume it.
func WithBeforeSendHookForAction(
	ctx context.Context,
	action string,
	hook BeforeSendHook,
) context.Context {
	return transport.WithBeforeSendHookForAction(ctx, action, hook)
}

func hasBeforeSendHook(ctx context.Context) bool {
	return transport.HasBeforeSendHook(ctx)
}
