package tokenizer

import (
	"encoding/json"
	"fmt"

	"github.com/hexagon-codes/ai-core/llm"
)

// RequestEstimate 是当前请求的上下文估算，不是已发生调用的 usage。
// ToolArgumentTokens 已包含在 MessageTokens 内；各类别不能重复相加。
// 媒体、私有续传与 Provider 扩展参数的实际上下文开销不由通用文本估算器假造。
type RequestEstimate struct {
	Tokens                int
	MessageTokens         int
	ToolArgumentTokens    int
	ToolSchemaTokens      int
	ResponseSchemaTokens  int
	ControlTokens         int
	UnestimatedMediaParts int
	OpaqueContinuations   int
	HasProviderMetadata   bool
}

// EstimateCompletionRequest 包含消息、工具定义/Schema、工具 arguments、输出 Schema 与输入控制文本。
// 多模态消息按 MultiContent 的优先级统计文本，不把图片 URL/base64 当作模型视觉 token。
// 未知部分通过字段显式声明；返回值始终是通用估算，精确值由支持完整请求的 Provider 计数器提供。
func EstimateCompletionRequest(request llm.CompletionRequest) (RequestEstimate, error) {
	estimate := RequestEstimate{MessageTokens: 3, HasProviderMetadata: len(request.Metadata) > 0}
	for _, message := range request.Messages {
		estimate.MessageTokens += 3 + CountGPT4(string(message.Role)) + CountGPT4(message.Name) + CountGPT4(message.ToolCallID)
		if len(message.MultiContent) == 0 {
			estimate.MessageTokens += CountGPT4(message.Content)
		} else {
			for _, part := range message.MultiContent {
				if part.Type == "text" {
					estimate.MessageTokens += CountGPT4(part.Text)
				} else {
					estimate.UnestimatedMediaParts++
				}
			}
		}
		for _, call := range message.ToolCalls {
			arguments := CountGPT4(call.Arguments)
			estimate.ToolArgumentTokens += arguments
			estimate.MessageTokens += CountGPT4(call.ID) + CountGPT4(call.Name) + arguments
		}
		if message.ProviderContinuation != nil {
			estimate.OpaqueContinuations++
		}
	}
	countJSON := func(value any) (int, error) {
		data, err := json.Marshal(value)
		if err != nil {
			return 0, fmt.Errorf("tokenizer: estimate request field: %w", err)
		}
		return CountGPT4(string(data)), nil
	}
	var err error
	if len(request.Tools) > 0 {
		estimate.ToolSchemaTokens, err = countJSON(request.Tools)
		if err != nil {
			return RequestEstimate{}, err
		}
	}
	if request.ResponseFormat != nil {
		estimate.ResponseSchemaTokens, err = countJSON(request.ResponseFormat)
		if err != nil {
			return RequestEstimate{}, err
		}
	}
	if request.ToolChoice != nil {
		estimate.ControlTokens, err = countJSON(request.ToolChoice)
		if err != nil {
			return RequestEstimate{}, err
		}
	}
	if len(request.Stop) > 0 {
		count, err := countJSON(request.Stop)
		if err != nil {
			return RequestEstimate{}, err
		}
		estimate.ControlTokens += count
	}
	estimate.Tokens = estimate.MessageTokens + estimate.ToolSchemaTokens + estimate.ResponseSchemaTokens + estimate.ControlTokens
	return estimate, nil
}
