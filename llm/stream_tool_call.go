package llm

import (
	"context"
	"errors"
	"io"

	"github.com/hexagon-codes/ai-core/streamx"
)

// StreamToolCallChecker 判断一段流式响应中模型是否决定发起工具调用。
//
// 不同模型在「何时暴露工具调用意图」上差异显著：
//   - OpenAI：通常首个 delta 即带 tool_calls；
//   - Claude：可能先输出一段文本再进入 tool_use 块，**只看首个 chunk 会漏判**。
//
// 因此 Checker 通常需要向前预读直到出现工具调用增量或流结束。Checker 会消费
// 传入的 sr，调用方若仍需完整流，应先 sr.Copy(2) 复制一路给 Checker：
//
//	sr := provider.Stream(ctx, req).Reader()
//	cp := sr.Copy(2)
//	has, err := llm.DefaultStreamToolCallChecker(ctx, cp[0])
//	// cp[1] 仍是完整流，按 has 决定走工具执行还是直接产出
type StreamToolCallChecker func(ctx context.Context, sr *streamx.StreamReader[*streamx.Chunk]) (bool, error)

// 编译期断言：默认实现的签名与 StreamToolCallChecker 类型一致。
var _ StreamToolCallChecker = DefaultStreamToolCallChecker

// DefaultStreamToolCallChecker 是 Claude-aware 的默认实现。
//
// 它持续读取 chunk，一旦遇到任意非空 ToolCalls（含 Claude 的 tool_use 起始块
// 与 input_json_delta 参数增量，经 ClaudeParser 解析后体现在 Chunk.ToolCalls）
// 即判定为「有工具调用」；读到流结束（io.EOF）仍未见则判定为「无工具调用」。
//
// 该实现对「工具调用前先输出文本」的模型是安全的——不会因首个文本 chunk 误判。
// 代价是当确无工具调用时需读完整个流；若需更早返回，可针对特定模型自定义 Checker
// （例如某些模型一旦出现文本即可确定不会再有工具调用）。
func DefaultStreamToolCallChecker(ctx context.Context, sr *streamx.StreamReader[*streamx.Chunk]) (bool, error) {
	for {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		default:
		}

		chunk, err := sr.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return false, nil
			}
			return false, err
		}
		if chunk != nil && len(chunk.ToolCalls) > 0 {
			return true, nil
		}
	}
}
