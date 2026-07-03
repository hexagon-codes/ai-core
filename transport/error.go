package transport

import (
	"fmt"
	"time"
)

// ProviderError carries structured upstream failure details.
//
// It lives in the transport layer because it is produced by the shared HTTP
// layer; the llm package re-exports it as a type alias for backward
// compatibility. Error() intentionally keeps the old
// "provider api error: status, body: ..." shape so existing callers that
// classify by message continue to work.
type ProviderError struct {
	Provider string
	Action   string
	Method   string
	URL      string

	StatusCode int
	Status     string

	Body          string
	BodyTruncated bool

	RequestPreview          string
	RequestPreviewTruncated bool
	RequestHeadersBlocked   []string

	ResponseHeaders map[string]string
	RequestID       string
	RetryAfter      time.Duration
	Elapsed         time.Duration

	Cause error
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}

	label := e.Provider
	if label == "" {
		label = "provider"
	}

	if e.Status != "" {
		if e.Body != "" {
			return fmt.Sprintf("%s api error: %s, body: %s", label, e.Status, e.Body)
		}
		return fmt.Sprintf("%s api error: %s", label, e.Status)
	}

	if e.Cause != nil {
		if e.Action != "" {
			return fmt.Sprintf("%s %s request failed: %v", label, e.Action, e.Cause)
		}
		return fmt.Sprintf("%s request failed: %v", label, e.Cause)
	}

	if e.Action != "" {
		return fmt.Sprintf("%s %s request failed", label, e.Action)
	}
	return fmt.Sprintf("%s request failed", label)
}

func (e *ProviderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}
