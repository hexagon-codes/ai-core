package anthropic

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/hexagon-codes/ai-core/llm"
)

const ContinuationProtocol = "anthropic.messages"

func (p *Provider) continuationSource(model string) string {
	credential := sha256.Sum256([]byte(p.apiKey))
	identity, _ := json.Marshal([]string{strings.TrimRight(p.baseURL, "/") + "/messages", hex.EncodeToString(credential[:]), model})
	sum := sha256.Sum256(identity)
	return hex.EncodeToString(sum[:])
}

func decodeAnthropicJSON(data []byte, value any) error {
	if !json.Valid(data) {
		return fmt.Errorf("anthropic: invalid JSON content")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	return decoder.Decode(value)
}

func (c *anthropicContent) UnmarshalJSON(data []byte) error {
	type contentFields anthropicContent
	var content contentFields
	if err := decodeAnthropicJSON(data, &content); err != nil {
		return err
	}
	*c = anthropicContent(content)
	c.RawJSON = append(json.RawMessage(nil), data...)
	return nil
}

func anthropicResumableStop(reason string) bool {
	return reason == "tool_use" || reason == "end_turn" || reason == "stop_sequence"
}

func (p *Provider) continuation(model string, output []json.RawMessage) *llm.ProviderContinuation {
	if len(output) == 0 {
		return nil
	}
	state := &llm.ProviderContinuation{Protocol: ContinuationProtocol, Source: p.continuationSource(model), Model: model}
	for _, block := range output {
		state.Output = append(state.Output, append(json.RawMessage(nil), block...))
	}
	return state
}

// continuationContent 只恢复原来源及仍对应本消息的原生块，私有续传不授权公开推理披露。
func (p *Provider) continuationContent(model string, message llm.Message) ([]map[string]any, bool, error) {
	state := message.ProviderContinuation
	if message.Role != llm.RoleAssistant || state == nil || state.Protocol != ContinuationProtocol || state.Source != p.continuationSource(model) || state.Model != model || message.HasMultiContent() {
		return nil, false, nil
	}
	blocks := make([]map[string]any, 0, len(state.Output))
	var text strings.Builder
	var calls []llm.ToolCallRef
	for _, raw := range state.Output {
		var block map[string]any
		if err := decodeAnthropicJSON(raw, &block); err != nil {
			return nil, false, err
		}
		blocks = append(blocks, block)
		if block["type"] == "text" {
			value, _ := block["text"].(string)
			text.WriteString(value)
		}
		if block["type"] == "tool_use" {
			id, _ := block["id"].(string)
			name, _ := block["name"].(string)
			args, err := json.Marshal(block["input"])
			if err != nil {
				return nil, false, err
			}
			calls = append(calls, llm.ToolCallRef{ID: id, Name: name, Arguments: string(args)})
		}
	}
	if text.String() != message.Content || len(calls) != len(message.ToolCalls) {
		return nil, false, nil
	}
	for index, call := range calls {
		other := message.ToolCalls[index]
		if call.ID != other.ID || call.Name != other.Name {
			return nil, false, nil
		}
		var args any
		arguments := other.Arguments
		if arguments == "" {
			arguments = "{}"
		}
		if err := decodeAnthropicJSON([]byte(arguments), &args); err != nil {
			return nil, false, err
		}
		normalized, err := json.Marshal(args)
		if err != nil {
			return nil, false, err
		}
		if call.Arguments != string(normalized) {
			return nil, false, nil
		}
	}
	return blocks, true, nil
}

type anthropicNativeBlock struct {
	fields                                      map[string]any
	text, thinking, signature, args             strings.Builder
	hasText, hasThinking, hasSignature, hasArgs bool
}

type anthropicNativeStream struct {
	blocks     map[int]*anthropicNativeBlock
	stopReason string
}

func (s *anthropicNativeStream) observe(data []byte) error {
	var event struct {
		Type         string          `json:"type"`
		Index        int             `json:"index"`
		ContentBlock json.RawMessage `json:"content_block"`
		Delta        struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			Signature   string `json:"signature"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
		} `json:"delta"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return err
	}
	switch event.Type {
	case "content_block_start":
		if s.blocks == nil {
			s.blocks = make(map[int]*anthropicNativeBlock)
		}
		var fields map[string]any
		if err := decodeAnthropicJSON(event.ContentBlock, &fields); err != nil {
			return err
		}
		if fields == nil {
			return fmt.Errorf("anthropic: content block is missing")
		}
		block := &anthropicNativeBlock{fields: fields}
		if value, ok := fields["text"].(string); ok {
			block.hasText = true
			block.text.WriteString(value)
		}
		if value, ok := fields["thinking"].(string); ok {
			block.hasThinking = true
			block.thinking.WriteString(value)
		}
		if value, ok := fields["signature"].(string); ok {
			block.hasSignature = true
			block.signature.WriteString(value)
		}
		s.blocks[event.Index] = block
	case "content_block_delta":
		block := s.blocks[event.Index]
		if block == nil {
			return nil
		}
		switch event.Delta.Type {
		case "text_delta":
			block.hasText = true
			block.text.WriteString(event.Delta.Text)
		case "thinking_delta":
			block.hasThinking = true
			block.thinking.WriteString(event.Delta.Thinking)
		case "signature_delta":
			block.hasSignature = true
			block.signature.WriteString(event.Delta.Signature)
		case "input_json_delta":
			block.hasArgs = true
			block.args.WriteString(event.Delta.PartialJSON)
		}
	case "message_delta":
		if event.Delta.StopReason != "" {
			s.stopReason = event.Delta.StopReason
		}
	}
	return nil
}

func (s *anthropicNativeStream) output() ([]json.RawMessage, error) {
	indices := make([]int, 0, len(s.blocks))
	for index := range s.blocks {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	output := make([]json.RawMessage, 0, len(indices))
	for _, index := range indices {
		block := s.blocks[index]
		fields := block.fields
		if block.hasText {
			fields["text"] = block.text.String()
		}
		if block.hasThinking {
			fields["thinking"] = block.thinking.String()
		}
		if block.hasSignature {
			fields["signature"] = block.signature.String()
		}
		if block.hasArgs {
			var args any
			if err := decodeAnthropicJSON([]byte(block.args.String()), &args); err != nil {
				return nil, err
			}
			fields["input"] = args
		}
		data, err := json.Marshal(fields)
		if err != nil {
			return nil, err
		}
		output = append(output, data)
	}
	return output, nil
}
