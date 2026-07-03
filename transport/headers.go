package transport

import (
	"net/http"
	"sort"
	"strings"
)

var unsafeHeaderNames = map[string]struct{}{
	"authorization":       {},
	"proxy-authorization": {},
	"api-key":             {},
	"x-api-key":           {},
	"anthropic-version":   {},
	"content-length":      {},
	"content-type":        {},
	"host":                {},
	"connection":          {},
	"transfer-encoding":   {},
	"trailer":             {},
	"te":                  {},
	"upgrade":             {},
}

// ApplySafeHeaders applies caller-supplied headers without allowing auth or
// transport-level overrides. It returns the skipped header names for diagnostics.
func ApplySafeHeaders(req *http.Request, headers map[string]string) []string {
	if req == nil || len(headers) == 0 {
		return nil
	}
	blocked := make([]string, 0)
	for name, value := range headers {
		trimmedName := strings.TrimSpace(name)
		if trimmedName == "" {
			continue
		}
		canonical := http.CanonicalHeaderKey(trimmedName)
		if _, unsafe := unsafeHeaderNames[strings.ToLower(canonical)]; unsafe {
			blocked = append(blocked, canonical)
			continue
		}
		req.Header.Set(canonical, value)
	}
	if len(blocked) == 0 {
		return nil
	}
	sort.Strings(blocked)
	return blocked
}
