// Package media 提供 ai-core 的统一媒体生成子域。
//
// 媒体生成涵盖图像（同步返回）、视频（异步 Submit→Poll 长任务）、
// 语音 STT/TTS、语音对话等。不同上游的关键差异在于"同步返回 vs 长任务轮询"，
// 本包用统一的任务模型（Submit→Poll）+ 同步包装（WaitFor）抹平这一差异：
//
//   - 异步 Provider（视频 / Sora / Veo 等长任务）实现 Submit→Poll；
//   - 调用方需要阻塞式结果时用 WaitFor 做同步包装，无需自己写轮询循环；
//   - 同步 Provider（DALL-E 等即时返回）直接返回结果，无需任务。
//
// 子域（各自独立 Provider + Service 路由）：
//
//   - media/image：文生图（同步 Generate）
//   - media/video：文生视频（异步 Submit→Poll + SubmitAndWait 同步包装）
//   - media/voice：语音 STT/TTS（后续阶段下沉）
//   - media/voicechat：语音对话（后续阶段下沉）
//
// 设计原则：媒体作为独立内聚包，不塞进 ai-core/llm；
// 媒体的路由权重 / 计费 / 成片落盘等应用层逻辑留在上层（hexclaw / hexeye），
// 本包只做"纯上游 Provider 调用 + 任务轮询"抽象。
package media

import (
	"context"
	"strings"
	"time"
)

// DefaultPollInterval 是 WaitFor 在未指定轮询间隔时使用的默认值。
const DefaultPollInterval = 2 * time.Second

// TaskState 表示媒体异步生成任务的状态。
//
// 不同上游用各自的状态字符串（如智谱 PROCESSING/SUCCESS/FAIL），
// Provider 实现负责映射到本枚举，给上层一个统一的状态机视图。
type TaskState string

const (
	// TaskQueued 任务已提交、排队中（尚未开始生成）。
	TaskQueued TaskState = "queued"
	// TaskRunning 任务生成中。
	TaskRunning TaskState = "running"
	// TaskSucceeded 任务成功，结果可用。
	TaskSucceeded TaskState = "succeeded"
	// TaskFailed 任务失败。
	TaskFailed TaskState = "failed"
	// TaskCancelled 任务取消。
	TaskCancelled TaskState = "cancelled"
)

// Terminal 报告状态是否为终态（成功或失败）。
//
// 轮询循环应在终态时停止。
func (s TaskState) Terminal() bool {
	return s == TaskSucceeded || s == TaskFailed || s == TaskCancelled
}

// NormalizeTaskState 把上游各异的任务状态字符串映射到统一的 TaskState。
//
// 这是 image / video 子域共享的单一映射源：以前两包各持一份 switch，
// 曾因 video 版漏掉 content_moderated 而让被审核拦截的视频任务落入
// default→TaskRunning、WaitFor 永不终止（bug-20260702）。合并为一份后
// 词表就不会再漂移。词表取两包并集，各词映射语义在两包间本就一致：
//   - *_moderated：request_moderated 仍在送审（进行中）；content_moderated
//     已被拦截（失败终态）。
//   - 未知非空状态按进行中处理（继续轮询），空状态按排队处理。
func NormalizeTaskState(status string) TaskState {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "queued", "queueing", "submitted", "pending", "created":
		return TaskQueued
	case "running", "processing", "generating", "in_progress", "request_moderated":
		return TaskRunning
	case "succeeded", "success", "succeed", "completed", "done", "ready":
		return TaskSucceeded
	case "failed", "fail", "error", "content_moderated":
		return TaskFailed
	case "cancelled", "canceled":
		return TaskCancelled
	default:
		if status == "" {
			return TaskQueued
		}
		return TaskRunning
	}
}

// PollFunc 轮询一次异步任务，返回任务是否到达终态及可能的错误。
//
// done=true 表示任务已结束（成功或失败均算 done，失败细节由调用方自行从
// 具体 Provider 的状态结构中读取）；err 非空表示轮询过程本身出错。
type PollFunc func(ctx context.Context) (done bool, err error)

// WaitFor 是"提交 → 轮询"异步 Provider 的统一同步包装。
//
// 它按 interval 周期性调用 poll，直到任务到达终态、poll 返回错误、
// 或 ctx 被取消为止。这样需要阻塞式结果的调用方无需自己实现轮询循环。
//
// 行为细节：
//   - interval <= 0 时使用 DefaultPollInterval；
//   - 立即先轮询一次，避免无谓等待一个 interval；
//   - ctx 取消时返回 ctx.Err()。
func WaitFor(ctx context.Context, interval time.Duration, poll PollFunc) error {
	if interval <= 0 {
		interval = DefaultPollInterval
	}
	timer := time.NewTimer(interval)
	defer timer.Stop()

	for {
		// 先轮询一次再等待，避免提交后白等一个周期。
		done, err := poll(ctx)
		if err != nil {
			return err
		}
		if done {
			return nil
		}

		// 重置定时器进入下一轮等待。
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(interval)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
	}
}
