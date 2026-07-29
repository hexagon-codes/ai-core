package streamx

import (
	"encoding/json"
	"strings"
)

// ============== OpenAI 解析器 ==============

// OpenAIParser 实现 OpenAI API 流式响应格式的解析
// OpenAI 使用 Server-Sent Events (SSE) 格式，每行以 "data: " 开头
// 数据为 JSON 格式，结构遵循 Chat Completions API 规范
// 流结束标记为 "data: [DONE]"
type OpenAIParser struct {
	// ReasoningEvidence must come from the trusted, frozen provider route.
	// Its zero value preserves legacy parsing and fails closed.
	ReasoningEvidence ReasoningDisclosureEvidence
}

// openAIChunk 是 OpenAI 流式响应的 JSON 结构
// 对应 Chat Completions API 的 chunk 格式
type openAIChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role             string `json:"role,omitempty"`
			Content          string `json:"content,omitempty"`
			Reasoning        string `json:"reasoning,omitempty"`         // Ollama (Qwen3, DeepSeek-R1)
			ReasoningContent string `json:"reasoning_content,omitempty"` // OpenAI o1/o3, DeepSeek API
			ToolCalls        []struct {
				Index    int    `json:"index"`
				ID       string `json:"id,omitempty"`
				Type     string `json:"type,omitempty"`
				Function struct {
					Name      string `json:"name,omitempty"`
					Arguments string `json:"arguments,omitempty"`
				} `json:"function,omitempty"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}

// Parse 解析 OpenAI 格式的 JSON 数据为 Chunk
// 提取 choices[0].delta 中的内容和工具调用信息
func (p *OpenAIParser) Parse(data []byte) (*Chunk, error) {
	var oai openAIChunk
	if err := json.Unmarshal(data, &oai); err != nil {
		return nil, err
	}

	chunk := &Chunk{
		ID:    oai.ID,
		Model: oai.Model,
		Raw:   data,
	}

	if len(oai.Choices) > 0 {
		choice := oai.Choices[0]
		chunk.Index = choice.Index
		chunk.Role = choice.Delta.Role
		chunk.Content = choice.Delta.Content
		chunk.FinishReason = choice.FinishReason

		// 提取推理内容（Qwen3 用 reasoning，DeepSeek/OpenAI 用 reasoning_content）
		if choice.Delta.Reasoning != "" {
			chunk.Reasoning = choice.Delta.Reasoning
			chunk.ReasoningDisclosure = NewReasoningDisclosure(
				"openai_compatible",
				"delta.reasoning",
				p.ReasoningEvidence,
			)
		} else if choice.Delta.ReasoningContent != "" {
			chunk.Reasoning = choice.Delta.ReasoningContent
			chunk.ReasoningDisclosure = NewReasoningDisclosure(
				"openai_compatible",
				"delta.reasoning_content",
				p.ReasoningEvidence,
			)
		}

		// 处理工具调用
		// OpenAI streaming 规范：仅首个 delta 携带完整 id/name/type，后续 delta 仅含 arguments 增量，
		// 用 Index 作为合并键（不是 ID），见 mergeToolCalls。
		for _, tc := range choice.Delta.ToolCalls {
			chunk.ToolCalls = append(chunk.ToolCalls, ToolCall{
				Index:     tc.Index,
				ID:        tc.ID,
				Type:      tc.Type,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}

	return chunk, nil
}

// IsDone 检查是否为 OpenAI 的流结束标记
// OpenAI 使用 "[DONE]" 作为流结束的特殊标记
func (p *OpenAIParser) IsDone(data []byte) bool {
	return strings.TrimSpace(string(data)) == "[DONE]"
}

// ============== Claude 解析器 ==============

// ClaudeParser 实现 Anthropic Claude API 流式响应格式的解析
// Claude 使用基于事件的 SSE 格式，包含多种事件类型：
//   - message_start: 消息开始，包含 ID、角色、模型信息
//   - content_block_start: 内容块开始
//   - content_block_delta: 内容增量
//   - message_delta: 消息级别的增量更新
//   - message_stop: 消息结束
type ClaudeParser struct{}

// claudeEvent 是 Claude 流式响应的事件结构
// type 字段标识事件类型，不同类型有不同的数据字段
type claudeEvent struct {
	Type         string         `json:"type"`
	Message      *claudeMessage `json:"message,omitempty"`
	Index        int            `json:"index,omitempty"`
	ContentBlock *struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
		// tool_use 块携带工具调用的 id 与 name（input 初始为空，参数经
		// 后续 input_json_delta 增量流式传输）
		ID   string `json:"id,omitempty"`
		Name string `json:"name,omitempty"`
	} `json:"content_block,omitempty"`
	Delta *struct {
		Type       string `json:"type,omitempty"`
		Text       string `json:"text,omitempty"`
		StopReason string `json:"stop_reason,omitempty"`
		// PartialJSON 是 input_json_delta 增量，拼接后构成工具调用参数 JSON
		PartialJSON string `json:"partial_json,omitempty"`
	} `json:"delta,omitempty"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

type claudeMessage struct {
	ID           string `json:"id"`
	Type         string `json:"type"`
	Role         string `json:"role"`
	Model        string `json:"model"`
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// Parse 解析 Claude 格式的事件数据为 Chunk
// 根据事件类型提取不同的信息：
//   - message_start: 提取 ID、角色、模型
//   - content_block_delta: 提取文本内容增量
//   - message_delta: 提取结束原因
func (p *ClaudeParser) Parse(data []byte) (*Chunk, error) {
	var evt claudeEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, err
	}

	chunk := &Chunk{
		Raw: data,
	}

	switch evt.Type {
	case "message_start":
		if evt.Message != nil {
			chunk.ID = evt.Message.ID
			chunk.Role = evt.Message.Role
			chunk.Model = evt.Message.Model
		}

	case "content_block_start":
		// 工具调用块起始：携带 id 与 name，参数随后由 input_json_delta 流式传入。
		// 文本块的 content_block_start 无增量内容，忽略即可。
		if evt.ContentBlock != nil && evt.ContentBlock.Type == "tool_use" {
			chunk.ToolCalls = []ToolCall{{
				Index: evt.Index,
				ID:    evt.ContentBlock.ID,
				Type:  "function",
				Name:  evt.ContentBlock.Name,
			}}
		}

	case "content_block_delta":
		if evt.Delta != nil {
			// input_json_delta 是工具参数增量；其余（text_delta 或未标注）为文本增量。
			// 以 content block 的 index 作为合并键（见 ToolCall.Index 说明）。
			if evt.Delta.Type == "input_json_delta" {
				chunk.ToolCalls = []ToolCall{{
					Index:     evt.Index,
					Arguments: evt.Delta.PartialJSON,
				}}
			} else {
				chunk.Content = evt.Delta.Text
			}
		}

	case "message_delta":
		if evt.Delta != nil {
			chunk.FinishReason = evt.Delta.StopReason
		}

	case "message_stop":
		chunk.FinishReason = "stop"
	}

	return chunk, nil
}

// IsDone 检查是否为 Claude 的流结束事件
// Claude 使用 message_stop 事件类型标识流结束
func (p *ClaudeParser) IsDone(data []byte) bool {
	var evt claudeEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		return false
	}
	return evt.Type == "message_stop"
}

// ============== Gemini 解析器 ==============

// GeminiParser 实现 Google Gemini API 流式响应格式的解析
// Gemini 的响应包含 candidates 数组，每个 candidate 包含 content.parts
// 文本内容在 parts[].text 中，可能有多个 part 需要拼接
// 流结束通过 finishReason 字段标识
type GeminiParser struct{}

// geminiChunk 是 Gemini 流式响应的 JSON 结构
// 主要关注 candidates[0].content.parts 中的文本内容
type geminiChunk struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
			Role string `json:"role"`
		} `json:"content"`
		FinishReason  string `json:"finishReason,omitempty"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings,omitempty"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata,omitempty"`
}

// Parse 解析 Gemini 格式的 JSON 数据为 Chunk
// 提取 candidates[0].content.parts 中的所有文本并拼接
func (p *GeminiParser) Parse(data []byte) (*Chunk, error) {
	var gem geminiChunk
	if err := json.Unmarshal(data, &gem); err != nil {
		return nil, err
	}

	chunk := &Chunk{
		Raw: data,
	}

	if len(gem.Candidates) > 0 {
		candidate := gem.Candidates[0]
		chunk.Role = candidate.Content.Role
		chunk.FinishReason = candidate.FinishReason

		// 合并所有文本部分
		var content strings.Builder
		for _, part := range candidate.Content.Parts {
			content.WriteString(part.Text)
		}
		chunk.Content = content.String()
	}

	return chunk, nil
}

// IsDone 检查是否为 Gemini 的流结束标记
// Gemini 通过 finishReason 字段标识流结束，非空表示结束
func (p *GeminiParser) IsDone(data []byte) bool {
	var gem geminiChunk
	if err := json.Unmarshal(data, &gem); err != nil {
		return false
	}
	if len(gem.Candidates) > 0 {
		return gem.Candidates[0].FinishReason != ""
	}
	return false
}

// ============== 通用 JSON 解析器 ==============

// JSONParser 提供可配置的通用 JSON 解析器
// 适用于非标准格式或需要自定义字段路径的场景
// 通过配置路径表达式来提取数据
type JSONParser struct {
	// ContentPath 内容字段的路径表达式
	// 使用点号分隔，数组用数字索引
	// 例如："choices.0.delta.content" 表示 data.choices[0].delta.content
	ContentPath string

	// DoneValue 结束标记的值
	// 当原始数据等于此值时表示流结束
	// 例如：OpenAI 的 "[DONE]"
	DoneValue string

	// DoneField 结束标记字段（暂未使用）
	DoneField string
}

// Parse 根据配置的路径解析 JSON 数据为 Chunk
// 使用 ContentPath 提取内容字段
func (p *JSONParser) Parse(data []byte) (*Chunk, error) {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	chunk := &Chunk{
		Raw: data,
	}

	// 提取内容
	if p.ContentPath != "" {
		if content := getNestedValue(raw, p.ContentPath); content != nil {
			if s, ok := content.(string); ok {
				chunk.Content = s
			}
		}
	}

	return chunk, nil
}

// IsDone 检查是否为流结束标记
// 通过比较原始数据与配置的 DoneValue 判断
func (p *JSONParser) IsDone(data []byte) bool {
	if p.DoneValue != "" {
		return strings.TrimSpace(string(data)) == p.DoneValue
	}
	return false
}

// getNestedValue 根据路径表达式获取嵌套的值
// 支持对象字段访问（如 "content"）和数组索引（如 "0"）
// 路径用点号分隔，如 "choices.0.delta.content"
func getNestedValue(data map[string]any, path string) any {
	parts := strings.Split(path, ".")
	var current any = data

	for _, part := range parts {
		switch v := current.(type) {
		case map[string]any:
			current = v[part]
		case []any:
			// 尝试解析数组索引
			idx := 0
			for _, c := range part {
				if c >= '0' && c <= '9' {
					idx = idx*10 + int(c-'0')
				} else {
					return nil
				}
			}
			if idx < len(v) {
				current = v[idx]
			} else {
				return nil
			}
		default:
			return nil
		}
	}

	return current
}
