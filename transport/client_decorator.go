package transport

import (
	"context"
	"errors"
	"net/http"
)

// ClientDecorator 在既有网络策略装配完成后观察实际 HTTP 交换。
// 返回的客户端必须保留原客户端行为及受保护 RoundTripper；默认不安装装饰器。
type ClientDecorator func(context.Context, *http.Client, Request) (*http.Client, error)
type clientDecoratorKey struct{}

func WithClientDecorator(ctx context.Context, decorator ClientDecorator) context.Context {
	if decorator == nil {
		return ctx
	}
	if previous, ok := ctx.Value(clientDecoratorKey{}).(ClientDecorator); ok {
		combined := decorator
		decorator = func(ctx context.Context, client *http.Client, request Request) (*http.Client, error) {
			client, err := previous(ctx, client, request)
			if err != nil {
				return nil, err
			}
			return combined(ctx, client, request)
		}
	}
	return context.WithValue(ctx, clientDecoratorKey{}, decorator)
}

// AttemptRejectedError 表示宿主在 HTTP 尝试前拒绝派发，不能当网络故障自动重试。
type AttemptRejectedError struct{ Cause error }

func (e *AttemptRejectedError) Error() string { return "HTTP attempt rejected: " + e.Cause.Error() }
func (e *AttemptRejectedError) Unwrap() error { return e.Cause }

func attemptRejected(err error) bool {
	var rejected *AttemptRejectedError
	return errors.As(err, &rejected)
}
