package llm

import "github.com/hexagon-codes/ai-core/transport"

// NetworkPolicy constrains upstream egress before a provider sends a request.
//
// The canonical definition lives in the transport layer (egress control is a
// transport concern). This is a type alias so all existing llm.NetworkPolicy
// references keep working unchanged and interoperate with transport.NetworkPolicy.
type NetworkPolicy = transport.NetworkPolicy

// ErrNetworkPolicy is the sentinel wrapped by network policy violations.
// It is the same value as transport.ErrNetworkPolicy, so
// errors.Is(err, llm.ErrNetworkPolicy) and errors.Is(err, transport.ErrNetworkPolicy)
// are equivalent.
var ErrNetworkPolicy = transport.ErrNetworkPolicy

// PublicNetworkPolicy returns a policy suitable for public SaaS LLM endpoints.
// It re-exports transport.PublicNetworkPolicy.
var PublicNetworkPolicy = transport.PublicNetworkPolicy
