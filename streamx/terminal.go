package streamx

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
)

// TerminalValidatingParser 为底层协议解析器补充终态校验。
//
// 只有底层解析器实际报告了终态（finish_reason、done、is_end 等）后，
// 普通 EOF 才被视为正常结束；否则返回 io.ErrUnexpectedEOF，避免上游
// 断流或代理截断被调用方误认为完整模型输出。
type TerminalValidatingParser struct {
	base      ChunkParser
	completed atomic.Bool
	observed  atomic.Bool
}

// NewTerminalValidatingParser 包装一个协议解析器并增加终态校验。
func NewTerminalValidatingParser(base ChunkParser) *TerminalValidatingParser {
	if base == nil {
		base = &OpenAIParser{}
	}
	return &TerminalValidatingParser{base: base}
}

// Parse 透传底层解析结果，并记录已观察到的终态。
func (p *TerminalValidatingParser) Parse(data []byte) (*Chunk, error) {
	// 任何被交给解析器的数据帧都算“已观察到响应”，即使解析失败，
	// 也不能在后续 EOF 时被当成完全空流而吞掉错误。
	p.observed.Store(true)
	chunk, err := p.base.Parse(data)
	if p.base.IsDone(data) {
		p.completed.Store(true)
		// [DONE] 等哨兵可能不是合法 JSON；底层解析器的错误在这里不代表失败。
		if err != nil {
			return nil, nil
		}
		return chunk, nil
	}
	if err == nil && chunk != nil && chunk.FinishReason != "" {
		p.completed.Store(true)
	}
	return chunk, err
}

// IsDone 透传底层解析器的结束标记判断。
func (p *TerminalValidatingParser) IsDone(data []byte) bool {
	return p.base.IsDone(data)
}

// NormalizeTerminalError 将无终态的普通 EOF 归一为 io.ErrUnexpectedEOF。
func (p *TerminalValidatingParser) NormalizeTerminalError(err error) error {
	if p.completed.Load() {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil
		}
		return err
	}
	// 完全没有任何数据帧的空流保持既有“空响应成功”语义；只有已经产生
	// 过内容却未报告终态时才视为异常截断。
	if !p.observed.Load() {
		return nil
	}
	if err == nil || errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}
