package llm

import (
	"context"
	"errors"
	"strings"
)

// FailoverReason 是 LLM 调用错误的分类：不同分类对应不同的降级/重试策略。
//
// 自 hexclaw engine 下沉到框架层（A9），使所有消费者共享统一的错误分类，避免各自重造。
//
// 设计原则：
//   - 按 HTTP 状态码 + error 特征分类；
//   - 每类有明确处理策略（见 FailoverAction / HandleFailover）；
//   - 未识别 → FailUnknown（默认重试 1 次，再失败上报）。
//
// 为什么要分类：统一布尔 isTransient 会让"402 余额不足/400 context 超长/401 无效 Key"
// 等不可重试错误也走退避重试，无意义。按原因分类后才能做精确降级（换凭证/压缩/切 Provider/直接报错）。
type FailoverReason int

const (
	FailNone           FailoverReason = iota // 无错误
	FailRateLimit                            // 429 - 指数退避重试
	FailQuotaExceeded                        // 402 - 换凭证
	FailContextTooLong                       // 400 + context length - 压缩后重试
	FailProviderDown                         // 503/504/超时 - 切 Provider
	FailInvalidKey                           // 401/403 - 报错给用户
	FailModelNotFound                        // 404 - 报错
	FailUnknown                              // 其他 - 重试 1 次
)

// String 返回稳定的低基数字符串（便于日志 / metrics / 测试断言）。
func (r FailoverReason) String() string {
	switch r {
	case FailNone:
		return "none"
	case FailRateLimit:
		return "rate_limit"
	case FailQuotaExceeded:
		return "quota_exceeded"
	case FailContextTooLong:
		return "context_too_long"
	case FailProviderDown:
		return "provider_down"
	case FailInvalidKey:
		return "invalid_key"
	case FailModelNotFound:
		return "model_not_found"
	default:
		return "unknown"
	}
}

// ClassifyError 把原始 error + HTTP 状态码 + 响应体分类为 FailoverReason。
//
//   - err: 原始错误（网络超时 / JSON 解析失败 / 上游 HTTP error 等）；
//   - httpStatus: 上游 HTTP 响应码（没有传 0）；
//   - body: 响应体文本（用于识别 context length / quota 等细节，没有传空串）。
//
// 优先级：httpStatus 精确分类 → err 消息匹配（上游库可能没暴露 status）→ FailUnknown。
//
// 本函数不给 llm.Provider 接口加任何方法，纯函数对外部错误做分类（A9 验收要求）。
func ClassifyError(err error, httpStatus int, body string) FailoverReason {
	if err == nil && httpStatus == 0 {
		return FailNone
	}

	// 1. HTTP 状态码优先分类
	switch httpStatus {
	case 429:
		return FailRateLimit
	case 402:
		return FailQuotaExceeded
	case 400:
		// 400 可能携带 context 超长 / 配额耗尽 等不同语义，依据响应体细分。
		if isContextLengthError(body) {
			return FailContextTooLong
		}
		if isQuotaBillingError(body) {
			return FailQuotaExceeded
		}
		return FailUnknown
	case 401, 403:
		return FailInvalidKey
	case 404:
		return FailModelNotFound
	case 500, 502, 503, 504:
		return FailProviderDown
	}

	// 2. ctx 超时/取消 → ProviderDown
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return FailProviderDown
		}

		// 3. error 消息模式匹配（上游库可能没暴露 httpStatus）。优先英文关键词，中文兜底。
		// 仅匹配语义关键词，不匹配裸状态码数字：裸 "402"/"404" 等会误命中 request id /
		// 模型名（如 "req_1402..."、"gpt-404-turbo"）。
		// 无效凭证先于配额判定：配额关键词（insufficient/balance）常与凭证错误同现，
		// 误判为配额会触发重试+换凭证，而无效 Key 应直接快速失败。
		msg := strings.ToLower(err.Error())
		switch {
		case strings.Contains(msg, "too many requests") ||
			strings.Contains(msg, "rate limit") || strings.Contains(msg, "请求过于频繁"):
			return FailRateLimit
		case strings.Contains(msg, "invalid api key") || strings.Contains(msg, "invalid_api_key") ||
			strings.Contains(msg, "incorrect api key") || strings.Contains(msg, "unauthorized") ||
			strings.Contains(msg, "authentication") || strings.Contains(msg, "无效"):
			return FailInvalidKey
		case isQuotaBillingError(msg):
			return FailQuotaExceeded
		case isContextLengthError(msg):
			return FailContextTooLong
		case strings.Contains(msg, "model not found") || strings.Contains(msg, "does not exist"):
			return FailModelNotFound
		case strings.Contains(msg, "unavailable") || strings.Contains(msg, "bad gateway") ||
			strings.Contains(msg, "gateway timeout") || strings.Contains(msg, "internal server error") ||
			strings.Contains(msg, "overloaded") || strings.Contains(msg, "server error") ||
			strings.Contains(msg, "timeout") || strings.Contains(msg, "connection") ||
			strings.Contains(msg, "deadline") || strings.Contains(msg, "temporary") ||
			strings.Contains(msg, "transient"):
			return FailProviderDown
		}
	}

	return FailUnknown
}

// isQuotaBillingError 模糊判定余额不足 / 配额耗尽 / 计费类错误（各 Provider message 不统一）。
func isQuotaBillingError(text string) bool {
	s := strings.ToLower(text)
	markers := []string{
		"insufficient_quota", "insufficient quota", "insufficient funds",
		"insufficient balance", "exceeded your current quota", "out of credit",
		"quota", "billing", "余额", "欠费",
	}
	for _, m := range markers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

// isContextLengthError 模糊判定上下文超长错误（各 Provider message 不统一）。
func isContextLengthError(text string) bool {
	s := strings.ToLower(text)
	markers := []string{
		"context length", "context_length",
		"maximum context", "max context",
		"context window", "context limit",
		"too many tokens", "token limit",
		"exceeds the model",
	}
	for _, m := range markers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

// FailoverAction 描述分类后推荐采取的行动；消费者据此决定重试 / 切 Provider / 换凭证 / 压缩 / 上报。
type FailoverAction struct {
	Retry            bool   // 是否值得重试
	SwitchProvider   bool   // 是否应换 Provider
	SwitchCredential bool   // 是否应换凭证
	CompressContext  bool   // 是否应压缩 context 再试
	BackoffSeconds   int    // 重试前等待秒数（0 = 立即）
	MaxRetries       int    // 最大重试次数
	UserFacing       string // 给用户看的简短原因
}

// HandleFailover 查表：FailoverReason → 推荐 FailoverAction。
func HandleFailover(reason FailoverReason) FailoverAction {
	switch reason {
	case FailRateLimit:
		return FailoverAction{Retry: true, BackoffSeconds: 4, MaxRetries: 3, UserFacing: "服务繁忙，稍后重试"}
	case FailQuotaExceeded:
		return FailoverAction{Retry: true, SwitchCredential: true, MaxRetries: 1, UserFacing: "账号余额不足或达到配额，尝试切换凭证"}
	case FailContextTooLong:
		return FailoverAction{Retry: true, CompressContext: true, MaxRetries: 1, UserFacing: "对话过长，已压缩后重试"}
	case FailProviderDown:
		return FailoverAction{Retry: true, SwitchProvider: true, MaxRetries: 2, UserFacing: "模型服务暂时不可用，切换其他 Provider"}
	case FailInvalidKey:
		return FailoverAction{Retry: false, UserFacing: "API 凭证无效，请检查设置"}
	case FailModelNotFound:
		return FailoverAction{Retry: false, UserFacing: "指定模型不存在或已下线"}
	case FailUnknown:
		return FailoverAction{Retry: true, MaxRetries: 1, UserFacing: "未知错误，重试一次"}
	default:
		return FailoverAction{Retry: false}
	}
}
