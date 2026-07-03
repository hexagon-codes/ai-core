package transport

import (
	"context"
	"io"
	"sync"
	"time"
)

type idleTimeoutReadCloser struct {
	body     io.ReadCloser
	timeout  time.Duration
	provider string
	action   string
	once     sync.Once
}

// NewIdleTimeoutReadCloser guards stream reads against silent idle stalls.
func NewIdleTimeoutReadCloser(body io.ReadCloser, timeout time.Duration, provider, action string) io.ReadCloser {
	if body == nil || timeout <= 0 {
		return body
	}
	return &idleTimeoutReadCloser{
		body:     body,
		timeout:  timeout,
		provider: provider,
		action:   action,
	}
}

func (r *idleTimeoutReadCloser) Read(p []byte) (int, error) {
	type readResult struct {
		data []byte
		n    int
		err  error
	}
	ch := make(chan readResult, 1)
	go func() {
		buf := make([]byte, len(p))
		n, err := r.body.Read(buf)
		ch <- readResult{data: buf, n: n, err: err}
	}()

	startedAt := time.Now()
	timer := time.NewTimer(r.timeout)
	defer timer.Stop()

	select {
	case result := <-ch:
		if result.n > 0 {
			copy(p, result.data[:result.n])
		}
		return result.n, result.err
	case <-timer.C:
		r.once.Do(func() {
			_ = r.body.Close()
		})
		return 0, &ProviderError{
			Provider: r.provider,
			Action:   r.action + ".stream_read",
			Elapsed:  time.Since(startedAt),
			Cause:    context.DeadlineExceeded,
		}
	}
}

func (r *idleTimeoutReadCloser) Close() error {
	var err error
	r.once.Do(func() {
		err = r.body.Close()
	})
	return err
}
