package voice

import (
	"context"
	"time"

	"github.com/hexagon-codes/ai-core/media"
)

// AudioProvider 是异步音频生成 Provider 接口（Submit → Poll）。
//
// 与同步的 TTSProvider 互补：TTSProvider 面向即时短文本合成（一次调用返回音频）；
// AudioProvider 面向异步、长时音频生成——长篇朗读、音乐 / 音效生成、批量合成等
// job-based 上游。需要阻塞式结果时用 SubmitAndWaitAudio 同步包装。
type AudioProvider interface {
	// Name 返回 Provider 名称。
	Name() string
	// Submit 提交音频生成任务，返回任务 ID。
	Submit(ctx context.Context, req AudioRequest) (taskID string, err error)
	// Poll 查询任务状态；State 到达终态时 Result 应可用。
	Poll(ctx context.Context, taskID string) (AudioTaskStatus, error)
}

// AudioRequest 异步音频生成请求。
type AudioRequest struct {
	Model  string      `json:"model,omitempty"`  // 模型 ID
	Text   string      `json:"text"`             // 要合成 / 生成的文本或提示词
	Voice  string      `json:"voice,omitempty"`  // 音色（合成场景）
	Format AudioFormat `json:"format,omitempty"` // 输出格式，空则 Provider 默认
	// Extra 承载 Provider 特有参数（音乐风格、时长、情感等），避免字段膨胀。
	Extra map[string]any `json:"extra,omitempty"`
}

// AudioTaskStatus 异步音频任务状态。
type AudioTaskStatus struct {
	TaskID   string            `json:"task_id"`
	State    media.TaskState   `json:"state"`
	Result   *SynthesizeResult `json:"result,omitempty"`
	Error    string            `json:"error,omitempty"`
	Progress float64           `json:"progress,omitempty"`
}

// SubmitAndWaitAudio 是异步音频生成的同步包装：先 Submit，再用 media.WaitFor
// 按 interval 轮询 Poll 直到任务到达终态、出错或 ctx 取消，返回最终状态。
//
// interval <= 0 时使用 media.DefaultPollInterval。
func SubmitAndWaitAudio(ctx context.Context, p AudioProvider, req AudioRequest, interval time.Duration) (AudioTaskStatus, error) {
	taskID, err := p.Submit(ctx, req)
	if err != nil {
		return AudioTaskStatus{}, err
	}

	var last AudioTaskStatus
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
