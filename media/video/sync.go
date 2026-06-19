package video

import (
	"context"
	"time"

	"github.com/hexagon-codes/ai-core/media"
)

// SubmitAndWait 是视频异步生成的同步包装。
//
// 它先 Submit 提交任务，再用 media.WaitFor 按 pollInterval 轮询 Poll
// 直到任务到达终态（成功或失败）、出错或 ctx 取消，返回最终任务状态。
// 适合需要阻塞式结果、不想自己写轮询循环的调用方；前端流式轮询场景仍可
// 直接用 Submit + Poll。
//
// pollInterval <= 0 时使用 media.DefaultPollInterval。
// 返回的 TaskStatus 在出错时为最后一次成功获取的状态（可能为零值）。
func (s *Service) SubmitAndWait(ctx context.Context, providerName string, req Request, pollInterval time.Duration) (TaskStatus, error) {
	taskID, err := s.Submit(ctx, providerName, req)
	if err != nil {
		return TaskStatus{}, err
	}

	var last TaskStatus
	waitErr := media.WaitFor(ctx, pollInterval, func(ctx context.Context) (bool, error) {
		st, perr := s.Poll(ctx, taskID)
		if perr != nil {
			return false, perr
		}
		last = st
		return st.Done, nil
	})
	return last, waitErr
}
