package image

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

const maxImageJSONResponseBytes = 64 << 20

// decodeImageJSON 读取带大小上限的 JSON 响应，避免异常上游放大内存。
func decodeImageJSON(body io.Reader, target any) error {
	data, err := io.ReadAll(io.LimitReader(body, maxImageJSONResponseBytes+1))
	if err != nil {
		return err
	}
	if len(data) > maxImageJSONResponseBytes {
		return fmt.Errorf("image response exceeds %d bytes", maxImageJSONResponseBytes)
	}
	return json.Unmarshal(data, target)
}

func firstJSONPath(root map[string]any, paths ...string) string {
	for _, path := range paths {
		var current any = root
		ok := true
		for _, part := range strings.Split(path, ".") {
			switch node := current.(type) {
			case map[string]any:
				current, ok = node[part]
			case []any:
				index, err := strconv.Atoi(part)
				if err != nil || index < 0 || index >= len(node) {
					ok = false
					break
				}
				current = node[index]
			default:
				ok = false
			}
			if !ok {
				break
			}
		}
		if ok {
			if value := anyToString(current); value != "" {
				return value
			}
		}
	}
	return ""
}

func anyToString(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case json.Number:
		return v.String()
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case fmt.Stringer:
		return v.String()
	default:
		return ""
	}
}

func mergeExtra(dst map[string]any, extra map[string]any) {
	for k, v := range extra {
		if strings.TrimSpace(k) == "" {
			continue
		}
		dst[k] = v
	}
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
