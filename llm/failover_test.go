package llm

import (
	"context"
	"errors"
	"testing"
)

func TestClassifyError_ByHTTPStatus(t *testing.T) {
	cases := []struct {
		status int
		body   string
		want   FailoverReason
	}{
		{429, "", FailRateLimit},
		{402, "", FailQuotaExceeded},
		{400, "maximum context length exceeded", FailContextTooLong},
		{400, "bad request", FailUnknown},
		{401, "", FailInvalidKey},
		{403, "", FailInvalidKey},
		{404, "", FailModelNotFound},
		{503, "", FailProviderDown},
		{500, "", FailProviderDown},
	}
	for _, c := range cases {
		if got := ClassifyError(errors.New("x"), c.status, c.body); got != c.want {
			t.Errorf("ClassifyError(status=%d, body=%q) = %s, want %s", c.status, c.body, got, c.want)
		}
	}
}

func TestClassifyError_ProviderError(t *testing.T) {
	err := &ProviderError{
		Provider:   "openai",
		StatusCode: 429,
		Status:     "429 Too Many Requests",
		Body:       `{"error":"rate limit"}`,
	}
	if got := ClassifyError(err, 0, ""); got != FailRateLimit {
		t.Fatalf("ClassifyError(ProviderError 429) = %s, want %s", got, FailRateLimit)
	}

	err = &ProviderError{
		Provider:   "openai",
		StatusCode: 400,
		Status:     "400 Bad Request",
		Body:       `{"error":"context length exceeded"}`,
	}
	if got := ClassifyError(err, 0, ""); got != FailContextTooLong {
		t.Fatalf("ClassifyError(ProviderError 400 context) = %s, want %s", got, FailContextTooLong)
	}
}

func TestClassifyError_ByMessage(t *testing.T) {
	cases := []struct {
		msg  string
		want FailoverReason
	}{
		{"HTTP 429 too many requests", FailRateLimit},
		{"rate limit reached", FailRateLimit},
		{"insufficient balance", FailQuotaExceeded},
		{"billing quota exceeded", FailQuotaExceeded},
		{"this model's maximum context length is 8192 tokens", FailContextTooLong},
		{"invalid api key", FailInvalidKey},
		{"model not found", FailModelNotFound},
		{"503 service unavailable", FailProviderDown},
		{"connection timeout", FailProviderDown},
		{"something weird", FailUnknown},
	}
	for _, c := range cases {
		if got := ClassifyError(errors.New(c.msg), 0, ""); got != c.want {
			t.Errorf("ClassifyError(%q) = %s, want %s", c.msg, got, c.want)
		}
	}
}

func TestClassifyError_CtxAndNil(t *testing.T) {
	if got := ClassifyError(context.DeadlineExceeded, 0, ""); got != FailProviderDown {
		t.Errorf("ctx deadline = %s, want provider_down", got)
	}
	if got := ClassifyError(nil, 0, ""); got != FailNone {
		t.Errorf("nil err = %s, want none", got)
	}
}

func TestHandleFailover_Strategies(t *testing.T) {
	if a := HandleFailover(FailRateLimit); !a.Retry || a.BackoffSeconds == 0 {
		t.Errorf("rate_limit 应退避重试: %+v", a)
	}
	if a := HandleFailover(FailQuotaExceeded); !a.SwitchCredential {
		t.Errorf("quota 应换凭证: %+v", a)
	}
	if a := HandleFailover(FailContextTooLong); !a.CompressContext {
		t.Errorf("context_too_long 应压缩: %+v", a)
	}
	if a := HandleFailover(FailProviderDown); !a.SwitchProvider {
		t.Errorf("provider_down 应切 Provider: %+v", a)
	}
	if a := HandleFailover(FailInvalidKey); a.Retry {
		t.Errorf("invalid_key 不应重试: %+v", a)
	}
	if a := HandleFailover(FailModelNotFound); a.Retry {
		t.Errorf("model_not_found 不应重试: %+v", a)
	}
}
