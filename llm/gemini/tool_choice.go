package gemini

import (
	"encoding/json"
	"fmt"
	"strings"
)

// geminiToolChoice 使用 GenerateContent 的 FunctionCallingConfig 合同。
func geminiToolChoice(raw any) (map[string]any, error) {
	if raw == nil {
		return nil, nil
	}
	if value, ok := raw.(string); ok {
		switch value {
		case "":
			return nil, nil
		case "auto":
			return map[string]any{"mode": "AUTO"}, nil
		case "none":
			return map[string]any{"mode": "NONE"}, nil
		case "required", "any":
			return map[string]any{"mode": "ANY"}, nil
		default:
			return map[string]any{"mode": "ANY", "allowedFunctionNames": []string{value}}, nil
		}
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("gemini: encode tool choice: %w", err)
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
		Mode         string   `json:"mode"`
		AllowedNames []string `json:"allowedFunctionNames"`
	}
	if err := json.Unmarshal(data, &choice); err != nil {
		return nil, fmt.Errorf("gemini: tool choice must be a string or object: %w", err)
	}
	if choice.Mode != "" {
		mode := strings.ToUpper(choice.Mode)
		switch mode {
		case "AUTO", "NONE", "ANY", "VALIDATED":
		default:
			return nil, fmt.Errorf("gemini: unsupported function calling mode %q", choice.Mode)
		}
		result := map[string]any{"mode": mode}
		if len(choice.AllowedNames) > 0 {
			result["allowedFunctionNames"] = choice.AllowedNames
		}
		return result, nil
	}
	switch choice.Type {
	case "auto":
		return map[string]any{"mode": "AUTO"}, nil
	case "none":
		return map[string]any{"mode": "NONE"}, nil
	case "required", "any":
		return map[string]any{"mode": "ANY"}, nil
	case "function", "tool", "":
		name := choice.Name
		if choice.Function != nil {
			name = choice.Function.Name
		}
		if name == "" {
			return nil, fmt.Errorf("gemini: named tool choice requires a name")
		}
		return map[string]any{"mode": "ANY", "allowedFunctionNames": []string{name}}, nil
	default:
		return nil, fmt.Errorf("gemini: unsupported tool choice type %q", choice.Type)
	}
}
