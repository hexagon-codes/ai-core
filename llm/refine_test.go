package llm

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestRefine_SuccessFirstTry 校验首次即通过，无重试。
func TestRefine_SuccessFirstTry(t *testing.T) {
	calls := 0
	gen := func(ctx context.Context, feedback string) (string, error) {
		calls++
		return "ok", nil
	}
	validate := func(ctx context.Context, c string) error { return nil }

	out, err := GenerateValidateRetry(context.Background(), gen, validate, RefineConfig{MaxRetries: 3})
	if err != nil || out != "ok" {
		t.Fatalf("got (%q,%v), want (ok,nil)", out, err)
	}
	if calls != 1 {
		t.Errorf("应只生成 1 次, got %d", calls)
	}
}

// TestRefine_RetryThenSucceed 首次校验失败、反馈进入第二次生成后通过。
// 验证 feedback 被正确传递（第二次 gen 依据 feedback 产出合格结果）。
func TestRefine_RetryThenSucceed(t *testing.T) {
	var gotFeedback string
	gen := func(ctx context.Context, feedback string) (string, error) {
		if feedback == "" {
			return "bad", nil // 首次产出不合格
		}
		gotFeedback = feedback
		return "good", nil // 拿到反馈后修正
	}
	validate := func(ctx context.Context, c string) error {
		if c != "good" {
			return errors.New("must be good")
		}
		return nil
	}

	out, err := GenerateValidateRetry(context.Background(), gen, validate, RefineConfig{MaxRetries: 2})
	if err != nil || out != "good" {
		t.Fatalf("got (%q,%v), want (good,nil)", out, err)
	}
	if !strings.Contains(gotFeedback, "must be good") {
		t.Errorf("第二次生成应收到首次校验错误作为反馈, got %q", gotFeedback)
	}
}

// TestRefine_ExhaustRetries 始终校验失败，耗尽重试返回 *RefineError，Unwrap 为末次校验错误。
func TestRefine_ExhaustRetries(t *testing.T) {
	sentinel := errors.New("always invalid")
	gen := func(ctx context.Context, feedback string) (string, error) { return "x", nil }
	validate := func(ctx context.Context, c string) error { return sentinel }

	_, err := GenerateValidateRetry(context.Background(), gen, validate, RefineConfig{MaxRetries: 2})
	var re *RefineError
	if !errors.As(err, &re) {
		t.Fatalf("应返回 *RefineError, got %T", err)
	}
	if re.Attempts != 3 { // 1 + 2 retries
		t.Errorf("Attempts = %d, want 3", re.Attempts)
	}
	if !errors.Is(err, sentinel) {
		t.Error("Unwrap 应可链到末次校验错误")
	}
}

// TestRefine_GenerateErrorImmediate 生成（基础设施）错误立即返回、不重试。
func TestRefine_GenerateErrorImmediate(t *testing.T) {
	calls := 0
	gen := func(ctx context.Context, feedback string) (string, error) {
		calls++
		return "", errors.New("network down")
	}
	validate := func(ctx context.Context, c string) error { return nil }

	_, err := GenerateValidateRetry(context.Background(), gen, validate, RefineConfig{MaxRetries: 5})
	if err == nil {
		t.Fatal("生成错误应返回")
	}
	if calls != 1 {
		t.Errorf("生成错误不应重试, got %d 次", calls)
	}
}

// TestRefine_CtxCancelled 已取消 ctx 立即返回错误。
func TestRefine_CtxCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	gen := func(ctx context.Context, feedback string) (string, error) { return "x", nil }
	validate := func(ctx context.Context, c string) error { return nil }

	if _, err := GenerateValidateRetry(ctx, gen, validate, RefineConfig{MaxRetries: 3}); err == nil {
		t.Error("已取消 ctx 应返回错误")
	}
}
