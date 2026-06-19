package llm

import (
	"context"
	"fmt"
)

// GenerateFunc 产生一次候选输出。
//
// feedback 是上一次校验失败的反馈（首次调用为空字符串）。实现方通常把 feedback
// 注入到 prompt，让模型针对上次的问题自我修正。返回的 error 视为基础设施错误
// （如网络/Provider 失败），会立即终止循环——它不同于"校验不通过"。
type GenerateFunc[T any] func(ctx context.Context, feedback string) (T, error)

// ValidateFunc 校验候选输出。
//
// 返回 nil 表示通过；返回非 nil 错误表示不通过，错误信息会作为反馈喂给下一次
// GenerateFunc，引导模型修正。校验可以是任意逻辑：JSON schema、SQL 可解析性、
// 代码可编译性、引用一致性、业务规则等。
type ValidateFunc[T any] func(ctx context.Context, candidate T) error

// RefineConfig 配置生成→校验→重试循环。
type RefineConfig struct {
	// MaxRetries 校验失败后的最大重试次数（总尝试数 = MaxRetries + 1）。
	// 0 表示只生成校验一次、不重试。
	MaxRetries int
}

// RefineError 聚合"生成→校验→重试"在耗尽重试后仍未通过的失败信息。
type RefineError struct {
	// Attempts 实际尝试次数。
	Attempts int
	// Last 最后一次校验失败的错误，可经 errors.Is/As 进一步判定。
	Last error
}

func (e *RefineError) Error() string {
	return fmt.Sprintf("generate-validate-retry: 经过 %d 次尝试仍未通过校验: %v", e.Attempts, e.Last)
}

// Unwrap 返回最后一次校验错误，支持 errors.Is/As 链式判断。
func (e *RefineError) Unwrap() error { return e.Last }

// GenerateValidateRetry 是通用的「生成 → 校验 → 重试」原语。
//
// 它反复调用 gen 产生候选、用 validate 校验；校验失败时把错误信息作为反馈喂给
// 下一次 gen，直到校验通过或耗尽重试。与 structured.Generate（JSON 专用：自动
// schema + 解析）不同，本原语不绑定任何输出形态，可用于任意「产出 → 校验 → 修正」
// 场景：SQL 生成 + 可解析校验、代码生成 + 编译校验、计划生成 + 合法性校验、
// 带引用回答 + 引用一致性校验等。
//
// 错误语义：
//   - gen 返回错误（基础设施/Provider 失败）→ 立即返回该错误，不重试；
//   - validate 返回错误（产出不达标）→ 作为反馈进入下一轮，直到耗尽重试；
//   - 耗尽重试仍未通过 → 返回 *RefineError（Unwrap 为最后一次校验错误）；
//   - ctx 取消 → 返回 ctx.Err()。
func GenerateValidateRetry[T any](
	ctx context.Context,
	gen GenerateFunc[T],
	validate ValidateFunc[T],
	cfg RefineConfig,
) (T, error) {
	var zero T

	total := cfg.MaxRetries + 1
	if total < 1 {
		total = 1
	}

	feedback := ""
	var lastErr error
	attempts := 0

	for i := 0; i < total; i++ {
		if err := ctx.Err(); err != nil {
			return zero, err
		}

		candidate, err := gen(ctx, feedback)
		if err != nil {
			// 基础设施错误：立即返回，不重试（反馈无意义）。
			return zero, fmt.Errorf("generate-validate-retry: 生成失败: %w", err)
		}

		attempts++

		verr := validate(ctx, candidate)
		if verr == nil {
			return candidate, nil
		}

		lastErr = verr
		feedback = verr.Error()
	}

	return zero, &RefineError{Attempts: attempts, Last: lastErr}
}
