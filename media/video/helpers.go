package video

import (
	"encoding/json"
	"fmt"
	"io"
)

const maxVideoJSONResponseBytes = 64 << 20

// decodeVideoJSON 读取带大小上限的 JSON 响应，避免异常上游放大内存。
func decodeVideoJSON(body io.Reader, target any) error {
	data, err := io.ReadAll(io.LimitReader(body, maxVideoJSONResponseBytes+1))
	if err != nil {
		return err
	}
	if len(data) > maxVideoJSONResponseBytes {
		return fmt.Errorf("video response exceeds %d bytes", maxVideoJSONResponseBytes)
	}
	return json.Unmarshal(data, target)
}
