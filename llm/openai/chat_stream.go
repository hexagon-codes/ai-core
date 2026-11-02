package openai

import (
	"errors"
	"io"
	"sync/atomic"

	"github.com/hexagon-codes/ai-core/streamx"
)

var errChatStreamDone = errors.New("chat completion stream done")

// chatCompletionStreamParser 仅约束 Chat Completions 的完成证据，不改变其他协议的 EOF 合同。
type chatCompletionStreamParser struct {
	base      streamx.ChunkParser
	completed atomic.Bool
}

func (p *chatCompletionStreamParser) Parse(data []byte) (*streamx.Chunk, error) {
	if p.base.IsDone(data) {
		p.completed.Store(true)
		return nil, errChatStreamDone
	}
	chunk, err := p.base.Parse(data)
	if err == nil && chunk != nil && chunk.FinishReason != "" {
		p.completed.Store(true)
	}
	return chunk, err
}

func (p *chatCompletionStreamParser) IsDone(data []byte) bool { return p.base.IsDone(data) }

func (p *chatCompletionStreamParser) NormalizeTerminalError(err error) error {
	if errors.Is(err, errChatStreamDone) {
		return nil
	}
	if err == nil && !p.completed.Load() {
		return io.ErrUnexpectedEOF
	}
	return err
}
