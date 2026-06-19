package voice

import (
	"context"
	"errors"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

// fakeAudio 是测试用异步音频 Provider。
type fakeAudio struct {
	submitErr     error
	pollsTillDone int
	calls         int
}

func (f *fakeAudio) Name() string { return "fake-async-audio" }

func (f *fakeAudio) Submit(ctx context.Context, req AudioRequest) (string, error) {
	if f.submitErr != nil {
		return "", f.submitErr
	}
	return "audio-task-1", nil
}

func (f *fakeAudio) Poll(ctx context.Context, taskID string) (AudioTaskStatus, error) {
	f.calls++
	if f.calls <= f.pollsTillDone {
		return AudioTaskStatus{TaskID: taskID, State: media.TaskRunning}, nil
	}
	return AudioTaskStatus{
		TaskID: taskID,
		State:  media.TaskSucceeded,
		Result: &SynthesizeResult{Format: FormatMP3, Audio: []byte("audio-bytes"), Size: 11},
	}, nil
}

func TestAudioProvider_SubmitAndWait(t *testing.T) {
	p := &fakeAudio{pollsTillDone: 1}
	st, err := SubmitAndWaitAudio(context.Background(), p, AudioRequest{Text: "你好"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if st.State != media.TaskSucceeded {
		t.Errorf("State = %q, want succeeded", st.State)
	}
	if st.Result == nil || st.Result.Size != 11 {
		t.Errorf("应返回带音频的结果, got %+v", st.Result)
	}
}

func TestAudioProvider_SubmitError(t *testing.T) {
	p := &fakeAudio{submitErr: errors.New("boom")}
	if _, err := SubmitAndWaitAudio(context.Background(), p, AudioRequest{Text: "x"}, 1); err == nil {
		t.Error("Submit 失败应传播错误")
	}
}
