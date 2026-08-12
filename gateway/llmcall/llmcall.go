// Package llmcall 把单次 LLM 调用收敛到一个入口，附带 transient error 自动 retry
// 与可选 progress 回调，避免每个调用方自己写 retry / 进度上报代码。
//
// 提供：
//   - CallWithProgress: 单次 LLM 调用 + 可选 progress callback
//   - 自动 retry on transient error（5xx / timeout / rate limit）— 指数退避
//   - 失败时保留 LLM 原文 error（编译类调用需要给用户排错）
//
// 仅依赖 ai-core/llm.Provider 接口，不引入更高层依赖。
//
// 不在范围内：
//   - prompt caching
//   - 跨 provider failover
//   - 流式 token-by-token（本 gateway 服务的是单次完整调用）
//   - cost ledger 写表
package llmcall

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/hexagon-codes/toolkit/util/retry"

	"github.com/hexagon-codes/ai-core/llm"
)

// ProgressStage 是 LLM 调用的粗粒度阶段。
type ProgressStage string

const (
	StageStarting  ProgressStage = "starting"  // 即将发起请求
	StageRetrying  ProgressStage = "retrying"  // 上次失败，退避后重试
	StageCompleted ProgressStage = "completed" // 收到响应
)

// Progress 单个进度事件。
type Progress struct {
	Stage    ProgressStage
	Attempt  int    // 第几次尝试（1-indexed）
	Reason   string // retry/error 原因
	Provider string // 实际用的 provider 名
	Model    string
}

// ProgressFunc 接收进度事件回调。
type ProgressFunc func(p Progress)

// Request 单次 LLM 调用的入参。
//
// Provider 用 llm.Provider 接口而非 hexagon.Provider —— 保持 ai-core 不向上依赖。
// hexagon.Provider 是 llm.Provider 的别名（参考 hexagon library），自然兼容。
type Request struct {
	Provider     llm.Provider
	ProviderName string // 仅用于日志 / progress 标签
	Req          llm.CompletionRequest
	MaxRetries   int           // 0 表示用默认（3 次）
	RetryBackoff time.Duration // 0 表示用默认（500ms × 2^attempt）
}

// Response gateway 返回值（直接转发 LLM response + 调用 metadata）。
type Response struct {
	*llm.CompletionResponse
	Attempts int // 实际尝试次数（含成功那次）
	Elapsed  time.Duration
}

// CallWithProgress 同步调一次 LLM，失败按 transient 类别退避重试。
//
// onProgress 可为 nil（不需要进度反馈时）。
func CallWithProgress(ctx context.Context, req Request, onProgress ProgressFunc) (*Response, error) {
	if req.Provider == nil {
		return nil, errors.New("llmcall: provider 为 nil")
	}
	maxRetries := req.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}
	baseBackoff := req.RetryBackoff
	if baseBackoff <= 0 {
		baseBackoff = 500 * time.Millisecond
	}

	emit := func(stage ProgressStage, attempt int, reason string) {
		if onProgress == nil {
			return
		}
		onProgress(Progress{
			Stage: stage, Attempt: attempt, Reason: reason,
			Provider: req.ProviderName, Model: req.Req.Model,
		})
	}

	start := time.Now()
	emit(StageStarting, 1, "")

	// 委托给 toolkit/util/retry.DoWithContext（参考 ai-core/llm/middleware.go、
	// ai-core/llm/router/router.go 的用法）。
	//   - retry.Attempts(maxRetries): 总尝试次数（含首次），与旧 for 1..maxRetries 一致。
	//   - retry.Delay(baseBackoff) + retry.Multiplier(2) + 显式指数策略：指数退避序列
	//     baseBackoff*2^(n-1)（500ms、1s、2s…），与旧 baseBackoff*(1<<(attempt-1)) 逐项相同。
	//   - retry.If(isTransient): 仅瞬时错误重试，非瞬时错误立即停止（等价旧 break）。
	//   - retry.OnRetry: 一次失败且将重试时回调，n 为一基已尝试序号，
	//     对应旧代码"以 attempt=n+1 发 StageRetrying"。
	// ctx 取消/超时由 DoWithContext 在循环顶部与退避计时器中处理，直接返回 ctx.Err()。
	var (
		resp     *llm.CompletionResponse
		lastErr  error
		attempts int
	)
	err := retry.DoWithContext(ctx, func() error {
		attempts++
		r, e := req.Provider.Complete(ctx, req.Req)
		if e != nil {
			lastErr = e
			return e
		}
		resp = r
		return nil
	},
		retry.Attempts(maxRetries),
		retry.Delay(baseBackoff),
		retry.Multiplier(2),
		retry.DelayType(retry.ExponentialBackoff),
		// 旧实现退避无上限，故将 MaxDelay 设为实质无界（toolkit 默认 30s 会截断，
		// 与旧 baseBackoff*2^(n-1) 不一致）；仅保留 toolkit 内部对 time.Duration 溢出的兜底。
		retry.MaxDelay(time.Duration(math.MaxInt64)),
		retry.If(isTransient),
		retry.OnRetry(func(n int, e error) {
			emit(StageRetrying, n+1, e.Error())
		}),
	)
	if err == nil {
		emit(StageCompleted, attempts, "")
		return &Response{
			CompletionResponse: resp,
			Attempts:           attempts,
			Elapsed:            time.Since(start),
		}, nil
	}

	// ctx 取消/超时：透传原始 ctx.Err()（区别于 Provider 业务错误），与旧行为一致。
	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}

	// 失败路径统一返回 (attempts=maxRetries) 包装错误，%w 保留 LLM 原文 error。
	// 注意：无论是重试耗尽还是非瞬时错误提前停止，旧代码均落到此分支，故沿用 maxRetries。
	return nil, fmt.Errorf("llmcall: 调用失败 (attempts=%d): %w", maxRetries, lastErr)
}

// Call 不需要进度反馈时的快捷入口。
func Call(ctx context.Context, req Request) (*Response, error) {
	return CallWithProgress(ctx, req, nil)
}

// isTransient 简化版瞬时错误判别（参考各 provider 的 5xx + rate limit + connection error）。
//
// 真实生产环境应该看 HTTP status / provider error code（参考 ai-core/llm/*/error.go），
// 这里 MVP 仅看关键字 — 后续可挂 provider-specific classifier。
func isTransient(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	for _, k := range []string{
		"timeout", "deadline", "connection", "eof", "503", "502", "504", "529",
		"rate limit", "too many requests", "overloaded",
	} {
		if strings.Contains(msg, k) {
			return true
		}
	}
	return false
}
