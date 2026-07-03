package voice

import (
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
