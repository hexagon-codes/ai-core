package image

import (
	"context"
	"time"

	"github.com/hexagon-codes/ai-core/media"
)

// AsyncImageProvider 是异步文生图 Provider 接口（Submit → Poll）。
//
// 同步即时返回的 Provider 用 Provider 接口即可；本接口面向 job-based 的异步
// 图像生成 API（排队渲染、超高分辨率、批量任务等），与 video.Provider 同构。
// 需要阻塞式结果时用 SubmitAndWait 同步包装，无需自己写轮询循环。
type AsyncImageProvider interface {
	// Name 返回 Provider 名称。
	Name() string
	// Submit 提交生成任务，返回任务 ID。
	Submit(ctx context.Context, req Request) (taskID string, err error)
	// Poll 查询任务状态；State 到达终态时 Result 应可用。
	Poll(ctx context.Context, taskID string) (TaskStatus, error)
	// SupportedModels 返回支持的模型 ID 列表。
	SupportedModels() []string
}

// TaskStatus 异步图像任务状态。
type TaskStatus struct {
	TaskID    string          `json:"task_id"`
	State     media.TaskState `json:"state"`
	Result    *Result         `json:"result,omitempty"`
	Error     string          `json:"error,omitempty"`
	RequestID string          `json:"request_id,omitempty"`
	Billed    *bool           `json:"billed,omitempty"`
	Progress  float64         `json:"progress,omitempty"`
	RawStatus string          `json:"raw_status,omitempty"`
}

// SubmitAndWait 是异步文生图的同步包装：先 Submit，再用 media.WaitFor 按
// interval 轮询 Poll 直到任务到达终态、出错或 ctx 取消，返回最终状态。
//
// interval <= 0 时使用 media.DefaultPollInterval。出错时返回最后一次成功
// 获取到的状态（可能为零值）。
func SubmitAndWait(ctx context.Context, p AsyncImageProvider, req Request, interval time.Duration) (TaskStatus, error) {
	taskID, err := p.Submit(ctx, req)
	if err != nil {
		return TaskStatus{}, err
	}

	var last TaskStatus
	waitErr := media.WaitFor(ctx, interval, func(ctx context.Context) (bool, error) {
		st, perr := p.Poll(ctx, taskID)
		if perr != nil {
			return false, perr
		}
		last = st
		return st.State.Terminal(), nil
	})
	return last, waitErr
}
