package anthropic

import (
	"encoding/json"
	"fmt"
)

// anthropicToolChoice 将通用选择合同映射到 Messages API，不改变模型或推理设置。
func anthropicToolChoice(raw any) (map[string]any, error) {
	if raw == nil {
		return nil, nil
	}
	if value, ok := raw.(string); ok {
		switch value {
		case "":
			return nil, nil
		case "auto", "none":
			return map[string]any{"type": value}, nil
		case "required", "any":
			return map[string]any{"type": "any"}, nil
		default:
			return map[string]any{"type": "tool", "name": value}, nil
		}
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("anthropic: encode tool choice: %w", err)
	}
	if string(data) == "null" {
		return nil, nil
	}
	var choice struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
		DisableParallel *bool `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(data, &choice); err != nil {
		return nil, fmt.Errorf("anthropic: tool choice must be a string or object: %w", err)
	}
	result := map[string]any{}
	switch choice.Type {
	case "auto", "none":
		result["type"] = choice.Type
	case "required", "any":
		result["type"] = "any"
	case "function", "tool", "":
		name := choice.Name
		if choice.Function != nil {
			name = choice.Function.Name
		}
		if name == "" {
			return nil, fmt.Errorf("anthropic: named tool choice requires a name")
		}
		result["type"], result["name"] = "tool", name
	default:
		return nil, fmt.Errorf("anthropic: unsupported tool choice type %q", choice.Type)
	}
	if choice.DisableParallel != nil {
		result["disable_parallel_tool_use"] = *choice.DisableParallel
	}
	return result, nil
}
