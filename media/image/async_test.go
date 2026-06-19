package image

import (
	"context"
	"errors"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

// fakeAsyncImage 是一个测试用异步 Provider：Poll 前 pollsTillDone 次返回 running，
// 之后返回 succeeded 并带结果。submitErr 非空时 Submit 直接失败。
type fakeAsyncImage struct {
	submitErr     error
	pollsTillDone int
	calls         int
}

func (f *fakeAsyncImage) Name() string              { return "fake-async-image" }
func (f *fakeAsyncImage) SupportedModels() []string { return []string{"fake-x"} }

func (f *fakeAsyncImage) Submit(ctx context.Context, req Request) (string, error) {
	if f.submitErr != nil {
		return "", f.submitErr
	}
	return "task-1", nil
}

func (f *fakeAsyncImage) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	f.calls++
	if f.calls <= f.pollsTillDone {
		return TaskStatus{TaskID: taskID, State: media.TaskRunning, Progress: 0.5}, nil
	}
	return TaskStatus{
		TaskID: taskID,
		State:  media.TaskSucceeded,
		Result: &Result{Provider: "fake-async-image", Images: []Image{{URL: "https://x/a.png"}}},
	}, nil
}

func TestAsyncImage_SubmitAndWait(t *testing.T) {
	p := &fakeAsyncImage{pollsTillDone: 2}
	// 用极小轮询间隔避免测试拖慢
	st, err := SubmitAndWait(context.Background(), p, Request{Prompt: "cat"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if st.State != media.TaskSucceeded {
		t.Errorf("State = %q, want succeeded", st.State)
	}
	if st.Result == nil || len(st.Result.Images) != 1 {
		t.Errorf("应返回带图像的结果, got %+v", st.Result)
	}
}

func TestAsyncImage_SubmitError(t *testing.T) {
	p := &fakeAsyncImage{submitErr: errors.New("boom")}
	if _, err := SubmitAndWait(context.Background(), p, Request{Prompt: "x"}, 1); err == nil {
		t.Error("Submit 失败应传播错误")
	}
}
