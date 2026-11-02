package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
)

// GenerationOptions 是模型生成参数的纯协议表示，不包含消息、工具绑定或运行身份。
type GenerationOptions struct {
	ToolChoice           any
	MaxTokens            int
	Temperature          *float64
	TopP                 *float64
	Stop                 []string
	User                 string
	ReasoningPolicyScope ReasoningPolicyScope
	ResponseFormat       *ResponseFormat
}

// Clone 冻结参数中的可变数据；工具选择和 Schema 按实际 JSON 表示复制并保留数值精度。
func (o GenerationOptions) Clone() (GenerationOptions, error) {
	out := o
	if o.Temperature != nil {
		value := *o.Temperature
		out.Temperature = &value
	}
	if o.TopP != nil {
		value := *o.TopP
		out.TopP = &value
	}
	out.Stop = slices.Clone(o.Stop)
	var err error
	out.ToolChoice, err = cloneProtocolValue(o.ToolChoice)
	if err != nil {
		return GenerationOptions{}, fmt.Errorf("clone tool choice: %w", err)
	}
	out.ResponseFormat, err = cloneProtocolValue(o.ResponseFormat)
	if err != nil {
		return GenerationOptions{}, fmt.Errorf("clone response format: %w", err)
	}
	return out, nil
}

// ApplyTo 将已复制的参数应用到请求，消息、工具和 Metadata 保持各自的所有权合同。
func (o GenerationOptions) ApplyTo(request CompletionRequest) (CompletionRequest, error) {
	copy, err := o.Clone()
	if err != nil {
		return CompletionRequest{}, err
	}
	request.ToolChoice, request.MaxTokens, request.Temperature, request.TopP = copy.ToolChoice, copy.MaxTokens, copy.Temperature, copy.TopP
	request.Stop, request.User = copy.Stop, copy.User
	request.ReasoningPolicyScope, request.ResponseFormat = copy.ReasoningPolicyScope, copy.ResponseFormat
	return request, nil
}

// CloneToolDefinitions 冻结发送给模型的工具 Schema，不复制或持有工具执行资源。
func CloneToolDefinitions(tools []ToolDefinition) ([]ToolDefinition, error) {
	return cloneProtocolValue(tools)
}

func cloneProtocolValue[T any](value T) (T, error) {
	var out T
	data, err := json.Marshal(value)
	if err != nil {
		return out, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&out); err != nil {
		return out, err
	}
	return out, nil
}
