package transport

import (
	"errors"
	"time"
)

// ErrNetworkPolicy is the sentinel wrapped by every network-policy violation.
// Callers classify with errors.Is(err, ErrNetworkPolicy). It lives in the
// transport layer because egress control is a transport concern; the llm
// package re-exports it as a variable alias for backward compatibility.
var ErrNetworkPolicy = errors.New("network policy violation")

// NetworkPolicy constrains upstream egress before a request is sent.
//
// Adapters keep a nil policy by default for backward compatibility. Passing a
// zero-value NetworkPolicy enables the public-HTTPS policy.
type NetworkPolicy struct {
	AllowHTTP        bool
	AllowPrivate     bool
	AllowedHosts     []string
	DNSLookupTimeout time.Duration
}

// PublicNetworkPolicy returns a policy suitable for public SaaS endpoints.
func PublicNetworkPolicy(allowedHosts ...string) NetworkPolicy {
	return NetworkPolicy{
		AllowedHosts:     append([]string(nil), allowedHosts...),
		DNSLookupTimeout: 5 * time.Second,
	}
}
