package llm

// IsUnsafeToReplay 遍历完整错误树，任一分支声明结果未知或不可重放即返回 true。
// false 标记只表达当前节点，不能掩盖包装或并列分支中的 true 标记。
func IsUnsafeToReplay(err error) bool {
	pending := []error{err}
	for len(pending) > 0 {
		current := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if current == nil {
			continue
		}
		if marker, ok := current.(interface{ UnsafeToReplay() bool }); ok && marker.UnsafeToReplay() {
			return true
		}
		switch wrapped := current.(type) {
		case interface{ Unwrap() []error }:
			pending = append(pending, wrapped.Unwrap()...)
		case interface{ Unwrap() error }:
			pending = append(pending, wrapped.Unwrap())
		}
	}
	return false
}
