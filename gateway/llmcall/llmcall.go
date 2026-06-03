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
	"strings"
	"time"

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
	Attempts int           // 实际尝试次数（含成功那次）
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
	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if attempt == 1 {
			emit(StageStarting, attempt, "")
		} else {
			emit(StageRetrying, attempt, lastErr.Error())
		}

		resp, err := req.Provider.Complete(ctx, req.Req)
		if err == nil {
			emit(StageCompleted, attempt, "")
			return &Response{
				CompletionResponse: resp,
				Attempts:           attempt,
				Elapsed:            time.Since(start),
			}, nil
		}
		lastErr = err
		if !isTransient(err) {
			break
		}
		if attempt < maxRetries {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(baseBackoff * (1 << (attempt - 1))):
			}
		}
	}
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
