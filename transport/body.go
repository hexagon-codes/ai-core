package transport

import (
	"io"
	"sync"
)

type cancelOnCloseReadCloser struct {
	body       io.ReadCloser
	cancel     func()
	cancelOnce sync.Once
	closeOnce  sync.Once
}

// NewCancelOnCloseReadCloser cancels the owning request context when the body
// reaches EOF/error or is explicitly closed.
func NewCancelOnCloseReadCloser(body io.ReadCloser, cancel func()) io.ReadCloser {
	if body == nil || cancel == nil {
		return body
	}
	return &cancelOnCloseReadCloser{
		body:   body,
		cancel: cancel,
	}
}

func (r *cancelOnCloseReadCloser) Read(p []byte) (int, error) {
	n, err := r.body.Read(p)
	if err != nil {
		r.cancelOnce.Do(r.cancel)
	}
	return n, err
}

func (r *cancelOnCloseReadCloser) Close() error {
	var err error
	r.closeOnce.Do(func() {
		err = r.body.Close()
		r.cancelOnce.Do(r.cancel)
	})
	return err
}
