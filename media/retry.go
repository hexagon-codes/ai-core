package media

import "github.com/hexagon-codes/ai-core/transport"

// SubmitRetryPolicy 返回计费型「任务创建/提交」类 POST 的 transport 重试策略。
//
// 未向上游透传幂等键时，5xx 等二义性响应下任务可能已创建并计费，
// 自动重试会重复建任务/重复计费，故禁用重试（MaxAttempts=1）；
// 已透传幂等键时上游可据键去重，维持 transport 默认重试。
// 轮询/查询类 GET 无此风险，保持默认重试即可，不应使用本策略。
func SubmitRetryPolicy(idempotencyKey string) transport.RetryPolicy {
	if idempotencyKey == "" {
		return transport.RetryPolicy{MaxAttempts: 1}
	}
	return transport.RetryPolicy{}
}
