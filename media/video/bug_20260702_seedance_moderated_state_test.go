package video

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/hexagon-codes/ai-core/media"
)

// bug-20260702：seedance normalizeTaskState 缺 content_moderated 分支，
// 网关对视频返回 content_moderated 时任务落 default→TaskRunning（非终态），
// media.WaitFor 永不终止直到 ctx 超时。此测试锁定「审核拦截=失败终态」。
func TestSeedancePollContentModeratedIsTerminalFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"content_moderated","error":{"message":"blocked by moderation"}}`))
	}))
	defer srv.Close()

	p := NewSeedanceCN("key", SeedanceWithBaseURL(srv.URL))
	st, err := p.Poll(context.Background(), "task_123")
	if err != nil {
		t.Fatal(err)
	}
	if st.State != media.TaskFailed {
		t.Fatalf("content_moderated must map to TaskFailed, got %q", st.State)
	}
	if !st.Done {
		t.Fatal("content_moderated must be a terminal state so WaitFor stops polling")
	}
}

// request_moderated（送审中）应仍为进行中状态，与 image 侧语义一致。
func TestSeedancePollRequestModeratedIsRunning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"request_moderated"}`))
	}))
	defer srv.Close()

	p := NewSeedanceCN("key", SeedanceWithBaseURL(srv.URL))
	st, err := p.Poll(context.Background(), "task_123")
	if err != nil {
		t.Fatal(err)
	}
	if st.State != media.TaskRunning {
		t.Fatalf("request_moderated must map to TaskRunning, got %q", st.State)
	}
	if st.Done {
		t.Fatal("request_moderated must not be terminal")
	}
}
