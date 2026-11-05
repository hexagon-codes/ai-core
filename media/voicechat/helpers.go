package voicechat

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const maxVoicechatJSONResponseBytes = 32 << 20

func decodeVoicechatJSON(body io.Reader, target any) error {
	data, err := io.ReadAll(io.LimitReader(body, maxVoicechatJSONResponseBytes+1))
	if err != nil {
		return err
	}
	if len(data) > maxVoicechatJSONResponseBytes {
		return fmt.Errorf("voicechat response exceeds %d bytes", maxVoicechatJSONResponseBytes)
	}
	return json.Unmarshal(data, target)
}

func requestIDFromHeader(header http.Header) string {
	for _, key := range []string{
		"x-request-id",
		"request-id",
		"openai-request-id",
		"anthropic-request-id",
		"x-volc-request-id",
		"x-tt-logid",
		"trace-id",
	} {
		if value := header.Get(key); value != "" {
			return value
		}
	}
	return ""
}
