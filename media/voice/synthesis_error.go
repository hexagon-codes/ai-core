package voice

import (
	"context"
	"errors"
	"strings"

	"github.com/hexagon-codes/ai-core/transport"
)

// IsDefinitiveSynthesisFailure 判断语音合成是否已明确未被接纳。
// 请求前校验、明确的供应商拒绝以及全部子错误均明确失败的聚合返回 true。
// 取消、超时、未知错误返回 false；包装错误和实现 DefinitiveFailure() bool
// 的适配器错误可保留原语义，调用方无需依赖具体错误类型。
func IsDefinitiveSynthesisFailure(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if failure, ok := err.(interface{ DefinitiveFailure() bool }); ok {
		return failure.DefinitiveFailure()
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !IsDefinitiveSynthesisFailure(cause) {
				return false
			}
		}
		return true
	}
	if upstream, ok := err.(*transport.ProviderError); ok {
		switch upstream.StatusCode {
		case 400, 401, 403, 404, 422, 429:
			return true
		default:
			return false
		}
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return IsDefinitiveSynthesisFailure(wrapped.Unwrap())
	}
	return false
}

type definitiveSynthesisError struct{ cause error }

func (e *definitiveSynthesisError) Error() string { return e.cause.Error() }
func (e *definitiveSynthesisError) Unwrap() error { return e.cause }
func (e *definitiveSynthesisError) DefinitiveFailure() bool {
	return !errors.Is(e.cause, context.Canceled) && !errors.Is(e.cause, context.DeadlineExceeded)
}

type synthesisFailures struct{ causes []error }

func (e *synthesisFailures) Error() string {
	messages := make([]string, len(e.causes))
	for i, cause := range e.causes {
		messages[i] = cause.Error()
	}
	return "chained-tts: all providers failed: " + strings.Join(messages, "; ")
}

func (e *synthesisFailures) Unwrap() []error { return e.causes }

func (e *synthesisFailures) DefinitiveFailure() bool {
	if len(e.causes) == 0 {
		return false
	}
	for _, cause := range e.causes {
		if !IsDefinitiveSynthesisFailure(cause) {
			return false
		}
	}
	return true
}
