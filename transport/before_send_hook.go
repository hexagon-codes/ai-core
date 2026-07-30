package transport

import "context"

// BeforeSendHook is the final durable authorization boundary for one physical
// HTTP attempt. Do invokes it after all local request construction and network
// policy checks have succeeded, immediately before http.Client.Do.
type BeforeSendHook func(context.Context) error

type beforeSendHookContextKey struct{}
type beforeSendHookBinding struct {
	action string
	hook   BeforeSendHook
}

// WithBeforeSendHook attaches an attempt-level authorization hook without
// changing the provider request payload. Returning an error from the hook
// prevents the physical HTTP attempt.
func WithBeforeSendHook(
	ctx context.Context,
	hook BeforeSendHook,
) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if hook == nil {
		return ctx
	}
	return context.WithValue(
		ctx,
		beforeSendHookContextKey{},
		beforeSendHookBinding{hook: hook},
	)
}

// WithBeforeSendHookForAction limits the authorization hook to one provider
// transport action. Auxiliary requests that reuse the same context (for
// example OAuth token refresh before a completion) do not consume the hook.
func WithBeforeSendHookForAction(
	ctx context.Context,
	action string,
	hook BeforeSendHook,
) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if hook == nil {
		return ctx
	}
	return context.WithValue(
		ctx,
		beforeSendHookContextKey{},
		beforeSendHookBinding{action: action, hook: hook},
	)
}

func beforeSendHookBindingFromContext(
	ctx context.Context,
) beforeSendHookBinding {
	if ctx == nil {
		return beforeSendHookBinding{}
	}
	binding, _ := ctx.Value(
		beforeSendHookContextKey{},
	).(beforeSendHookBinding)
	return binding
}

func beforeSendHookForAction(
	ctx context.Context,
	action string,
) BeforeSendHook {
	binding := beforeSendHookBindingFromContext(ctx)
	if binding.hook == nil ||
		binding.action != "" && binding.action != action {
		return nil
	}
	return binding.hook
}

// HasBeforeSendHook reports whether a caller requires a physical transport
// boundary. Middleware that could satisfy a request without reaching transport
// must bypass that shortcut when this returns true.
func HasBeforeSendHook(ctx context.Context) bool {
	return beforeSendHookBindingFromContext(ctx).hook != nil
}
