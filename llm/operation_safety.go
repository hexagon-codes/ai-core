package llm

import (
	"context"

	"github.com/hexagon-codes/ai-core/transport"
)

// OperationSafety is the operation-level automatic replay contract understood
// by ai-core providers and their shared HTTP transport.
type OperationSafety = transport.OperationSafety

const (
	// OperationSafetyDefault preserves the provider's existing retry policy.
	OperationSafetyDefault = transport.OperationSafetyDefault
	// OperationSafetyNonIdempotent disables internal automatic replay after an
	// ambiguous transport or response-decode failure.
	OperationSafetyNonIdempotent = transport.OperationSafetyNonIdempotent
)

// WithOperationSafety marks one provider operation with an explicit automatic
// replay contract while keeping CompletionRequest wire payloads unchanged.
func WithOperationSafety(ctx context.Context, safety OperationSafety) context.Context {
	return transport.WithOperationSafety(ctx, safety)
}

// OperationSafetyFromContext returns the operation-level replay contract.
func OperationSafetyFromContext(ctx context.Context) OperationSafety {
	return transport.OperationSafetyFromContext(ctx)
}
