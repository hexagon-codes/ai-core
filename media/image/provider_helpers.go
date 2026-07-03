package image

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

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
