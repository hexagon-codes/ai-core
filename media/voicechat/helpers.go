package voicechat

import "net/http"

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
