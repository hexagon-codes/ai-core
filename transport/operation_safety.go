package transport

import "context"

// OperationSafety describes whether the same logical operation may be
// automatically replayed by the HTTP transport after an ambiguous failure.
type OperationSafety uint8

const (
	// OperationSafetyDefault preserves the adapter's existing retry policy.
	OperationSafetyDefault OperationSafety = iota
	// OperationSafetyNonIdempotent disables automatic transport replay. The
	// caller may still start a new logical operation explicitly.
	OperationSafetyNonIdempotent
)

type operationSafetyContextKey struct{}

// WithOperationSafety attaches an explicit operation-level replay contract to
// ctx. It does not alter requests that do not carry the marker.
func WithOperationSafety(ctx context.Context, safety OperationSafety) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, operationSafetyContextKey{}, safety)
}

// OperationSafetyFromContext returns the explicit operation safety marker.
// An unmarked context preserves the transport's existing retry behavior.
func OperationSafetyFromContext(ctx context.Context) OperationSafety {
	if ctx == nil {
		return OperationSafetyDefault
	}
	safety, _ := ctx.Value(operationSafetyContextKey{}).(OperationSafety)
	return safety
}
