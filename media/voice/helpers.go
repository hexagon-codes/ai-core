package voice

import (
	"fmt"
	"io"
	"net/http"

	"github.com/hexagon-codes/ai-core/llm"
)

func cloneVoiceHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	out := make(map[string]string, len(headers))
	for k, v := range headers {
		out[k] = v
	}
	return out
}

func cloneVoiceNetworkPolicy(policy *llm.NetworkPolicy) *llm.NetworkPolicy {
	if policy == nil {
		return nil
	}
	cloned := *policy
	cloned.AllowedHosts = append([]string(nil), policy.AllowedHosts...)
	return &cloned
}

// readVoiceBody 读取带大小上限的响应体；超过上限时返回明确错误，
// 避免把截断的音频当成完整合成结果。
func readVoiceBody(body io.Reader, limit int64, provider string) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%s: response exceeds %d bytes", provider, limit)
	}
	return data, nil
}

func requestIDFromHeader(header http.Header) string {
	for _, key := range []string{
		"x-request-id",
		"request-id",
		"openai-request-id",
		"anthropic-request-id",
		"x-volc-request-id",
		"x-tt-logid",
		"apim-request-id",
		"ms-cv",
		"trace-id",
	} {
		if value := header.Get(key); value != "" {
			return value
		}
	}
	return ""
}
