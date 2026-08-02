package qdrant

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/hexagon-codes/ai-core/store/vector"
	"github.com/hexagon-codes/toolkit/util/retry"
)

// ErrInvalidBatchConfig is returned before launching workers. It prevents
// zero-sized loops, zero-capacity semaphore deadlocks, and invalid retry
// policies from turning caller mistakes into hangs or panics.
var ErrInvalidBatchConfig = errors.New("qdrant: invalid batch configuration")

// BatchConfig 批量操作配置
type BatchConfig struct {
	// BatchSize 每批大小
	BatchSize int

	// Concurrency 并发数
	Concurrency int

	// RetryCount 重试次数
	RetryCount int

	// RetryDelay 重试延迟
	RetryDelay time.Duration

	// OnProgress 进度回调
	OnProgress func(processed, total int)

	// OnError 错误回调（返回 true 继续，false 停止）
	OnError func(err error, batch int) bool
}

// DefaultBatchConfig 默认批量配置
var DefaultBatchConfig = BatchConfig{
	BatchSize:   100,
	Concurrency: 4,
	RetryCount:  3,
	RetryDelay:  time.Second,
}

// BatchOption 批量操作选项
type BatchOption func(*BatchConfig)

// WithBatchSize 设置批量大小
func WithBatchSize(size int) BatchOption {
	return func(c *BatchConfig) {
		c.BatchSize = size
	}
}

// WithConcurrency 设置并发数
func WithConcurrency(n int) BatchOption {
	return func(c *BatchConfig) {
		c.Concurrency = n
	}
}

// WithRetry 设置重试次数和延迟
func WithRetry(count int, delay time.Duration) BatchOption {
	return func(c *BatchConfig) {
		c.RetryCount = count
		c.RetryDelay = delay
	}
}

// WithOnProgress 设置进度回调
func WithOnProgress(fn func(processed, total int)) BatchOption {
	return func(c *BatchConfig) {
		c.OnProgress = fn
	}
}

// WithOnError 设置错误回调
func WithOnError(fn func(err error, batch int) bool) BatchOption {
	return func(c *BatchConfig) {
		c.OnError = fn
	}
}

// AddBatch 批量添加文档（带并发和重试）
func (s *Store) AddBatch(ctx context.Context, docs []vector.Document, opts ...BatchOption) error {
	cfg := DefaultBatchConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := validateBatchConfig(cfg); err != nil {
		return err
	}

	if len(docs) == 0 {
		return nil
	}

	// 分批
	var batches [][]vector.Document
	for i := 0; i < len(docs); i += cfg.BatchSize {
		end := i + cfg.BatchSize
		if end > len(docs) {
			end = len(docs)
		}
		batches = append(batches, docs[i:end])
	}

	// 并发处理
	var wg sync.WaitGroup
	sem := make(chan struct{}, cfg.Concurrency)
	errCh := make(chan error, len(batches))
	processed := 0
	var mu sync.Mutex

	for batchIdx, batch := range batches {
		wg.Add(1)
		go func(idx int, batch []vector.Document) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
			defer func() { <-sem }()

			// 使用 toolkit retry 进行重试
			err := retry.DoWithContext(ctx, func() error {
				return s.Add(ctx, batch)
			}, retry.Attempts(cfg.RetryCount+1),
				retry.Delay(cfg.RetryDelay),
				retry.DelayType(retry.LinearBackoff))

			if err != nil {
				if cfg.OnError != nil {
					if !cfg.OnError(err, idx) {
						errCh <- err
						return
					}
				} else {
					errCh <- fmt.Errorf("batch %d failed: %w", idx, err)
					return
				}
			}

			// 更新进度
			mu.Lock()
			processed += len(batch)
			if cfg.OnProgress != nil {
				cfg.OnProgress(processed, len(docs))
			}
			mu.Unlock()
		}(batchIdx, batch)
	}

	wg.Wait()
	close(errCh)

	// 收集错误
	for err := range errCh {
		if err != nil {
			return err
		}
	}

	return nil
}

// DeleteBatch 批量删除文档
func (s *Store) DeleteBatch(ctx context.Context, ids []string, opts ...BatchOption) error {
	cfg := DefaultBatchConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := validateBatchConfig(cfg); err != nil {
		return err
	}

	if len(ids) == 0 {
		return nil
	}

	// 分批删除
	for i := 0; i < len(ids); i += cfg.BatchSize {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		end := i + cfg.BatchSize
		if end > len(ids) {
			end = len(ids)
		}

		batch := ids[i:end]

		// 使用 toolkit retry 进行重试
		err := retry.DoWithContext(ctx, func() error {
			return s.Delete(ctx, batch)
		}, retry.Attempts(cfg.RetryCount+1),
			retry.Delay(cfg.RetryDelay),
			retry.DelayType(retry.LinearBackoff))

		if err != nil {
			return fmt.Errorf("delete batch failed: %w", err)
		}

		if cfg.OnProgress != nil {
			cfg.OnProgress(end, len(ids))
		}
	}

	return nil
}

func validateBatchConfig(cfg BatchConfig) error {
	switch {
	case cfg.BatchSize <= 0:
		return fmt.Errorf("%w: batch size must be positive", ErrInvalidBatchConfig)
	case cfg.Concurrency <= 0:
		return fmt.Errorf("%w: concurrency must be positive", ErrInvalidBatchConfig)
	case cfg.RetryCount < 0:
		return fmt.Errorf("%w: retry count must not be negative", ErrInvalidBatchConfig)
	case cfg.RetryDelay < 0:
		return fmt.Errorf("%w: retry delay must not be negative", ErrInvalidBatchConfig)
	default:
		return nil
	}
}

// Scroll 滚动获取所有文档
func (s *Store) Scroll(ctx context.Context, batchSize int, fn func(docs []vector.Document) error) error {
	if batchSize <= 0 {
		batchSize = 100
	}

	var offset any

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		req := map[string]any{
			"limit":        batchSize,
			"with_payload": true,
			"with_vector":  true,
		}
		if offset != nil {
			req["offset"] = offset
		}

		resp, err := s.doRequest(ctx, "POST", "/collections/"+s.config.Collection+"/points/scroll", req)
		if err != nil {
			return fmt.Errorf("scroll failed: %w", err)
		}

		var scrollResp scrollResponse
		if err := json.Unmarshal(resp, &scrollResp); err != nil {
			return fmt.Errorf("failed to parse scroll response: %w", err)
		}

		if len(scrollResp.Result.Points) == 0 {
			break
		}

		// 转换文档
		docs := make([]vector.Document, len(scrollResp.Result.Points))
		for i, point := range scrollResp.Result.Points {
			doc := vector.Document{
				ID:        s.fromPointID(point.ID, point.Payload),
				Embedding: point.Vector,
			}

			if point.Payload != nil {
				if content, ok := point.Payload["content"].(string); ok {
					doc.Content = content
				}
				doc.Metadata = make(map[string]any)
				for k, v := range point.Payload {
					if k != "content" && k != "created_at" && k != "_original_id" {
						doc.Metadata[k] = v
					}
				}
			}

			docs[i] = doc
		}

		if err := fn(docs); err != nil {
			return err
		}

		offset = scrollResp.Result.NextPageOffset
		if offset == nil {
			break
		}
	}

	return nil
}

type scrollResponse struct {
	Result struct {
		Points         []scrollPoint `json:"points"`
		NextPageOffset any           `json:"next_page_offset"`
	} `json:"result"`
}

type scrollPoint struct {
	ID      any            `json:"id"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector"`
}
