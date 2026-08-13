package cicheck

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

var semverPattern = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+$`)

// TestVersionFileIsSemverCompliant 校验 VERSION 文件是合法 SemVer。
// 刻意不与 git tag 精确绑定：tag 一旦推送即不可移动，若测试绑定具体 tag
// 值，VERSION 未同步时该 tag 的 CI 将永远失败且无法通过后续提交修复。
// 版本号与 tag 的一致性由发布流程保证，这里只守住 SemVer 格式不变量。
func TestVersionFileIsSemverCompliant(t *testing.T) {
	t.Parallel()

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate test file")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(currentFile), ".."))
	versionBytes, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		t.Fatalf("read VERSION: %v", err)
	}
	fileVersion := strings.TrimSpace(string(versionBytes))
	if !semverPattern.MatchString(fileVersion) {
		t.Fatalf("VERSION = %q, want SemVer format vMAJOR.MINOR.PATCH", fileVersion)
	}
}
