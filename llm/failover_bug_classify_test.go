package llm

import (
	"errors"
	"testing"
)

// TestClassifyError_Bug_SubstringStatusMatches verifies that bare HTTP status
// codes embedded in request IDs or model names do not trigger false-positive
// classification, and that invalid-key beats quota, and that a 400 with a quota
// body is classified as quota rather than unknown.
func TestClassifyError_Bug_SubstringStatusMatches(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		status int
		body   string
		want   FailoverReason
	}{
		{
			name: "request id containing 402 is not quota",
			err:  errors.New("upstream error req_1402abcd processing failed"),
			want: FailUnknown,
		},
		{
			name: "503 unavailable message triggers provider failover",
			err:  errors.New("503 unavailable"),
			want: FailProviderDown,
		},
		{
			name: "model name containing 404 is not model_not_found",
			err:  errors.New("call to gpt-404-turbo returned malformed json"),
			want: FailUnknown,
		},
		{
			name: "request id containing 529 is not classified by substring",
			err:  errors.New("trace req_529xyz parse failure"),
			want: FailUnknown,
		},
		{
			name: "invalid api key beats quota when both keywords present",
			err:  errors.New("invalid api key; account has insufficient quota balance"),
			want: FailInvalidKey,
		},
		{
			name:   "401 status with quota body stays invalid key",
			status: 401,
			body:   "insufficient_quota: your account balance is too low",
			want:   FailInvalidKey,
		},
		{
			name:   "400 status with insufficient_quota body is quota",
			status: 400,
			body:   `{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}`,
			want:   FailQuotaExceeded,
		},
		{
			name:   "400 status with billing body is quota",
			status: 400,
			body:   "billing hard limit reached",
			want:   FailQuotaExceeded,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ClassifyError(c.err, c.status, c.body)
			if got != c.want {
				t.Errorf("ClassifyError(%v, %d, %q) = %v, want %v",
					c.err, c.status, c.body, got, c.want)
			}
		})
	}
}
