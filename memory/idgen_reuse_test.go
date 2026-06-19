package memory

import (
	"strings"
	"testing"
)

// TestDefaultIDGen_UniquePrefixed 验证 memory ID 复用 idgen 后仍唯一、带 mem- 前缀（差分：行为契约不变）。
func TestDefaultIDGen_UniquePrefixed(t *testing.T) {
	const n = 10000
	seen := make(map[string]bool, n)
	for range n {
		id := defaultIDGen()
		if !strings.HasPrefix(id, "mem-") {
			t.Fatalf("ID 应带 mem- 前缀, got %q", id)
		}
		if seen[id] {
			t.Fatalf("ID 碰撞: %q", id)
		}
		seen[id] = true
	}
}

// TestGenerateVectorID_UniquePrefixed 验证 vector ID 复用 idgen 后仍唯一、带 vec- 前缀。
func TestGenerateVectorID_UniquePrefixed(t *testing.T) {
	const n = 10000
	seen := make(map[string]bool, n)
	for range n {
		id := generateVectorID()
		if !strings.HasPrefix(id, "vec-") {
			t.Fatalf("ID 应带 vec- 前缀, got %q", id)
		}
		if seen[id] {
			t.Fatalf("ID 碰撞: %q", id)
		}
		seen[id] = true
	}
}
