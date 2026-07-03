package llm

import "github.com/hexagon-codes/ai-core/transport"

// ProviderError carries structured upstream failure details.
//
// The canonical definition (and its Error()/Unwrap() methods) lives in the
// transport layer, which produces it from the shared HTTP layer. This is a
// type alias so all existing llm.ProviderError references keep working
// unchanged and interoperate with transport.ProviderError via errors.As.
type ProviderError = transport.ProviderError
