package cicheck

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestVersionFileMatchesExactReleaseTag(t *testing.T) {
	t.Parallel()

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate test file")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(currentFile), ".."))
	command := exec.Command("git", "describe", "--tags", "--exact-match", "HEAD")
	command.Dir = root
	tagBytes, err := command.Output()
	if err != nil {
		t.Skip("checkout is not an exact release tag")
	}

	versionBytes, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		t.Fatalf("read VERSION: %v", err)
	}
	tag := strings.TrimSpace(string(tagBytes))
	fileVersion := strings.TrimSpace(string(versionBytes))
	if fileVersion != tag {
		t.Fatalf("VERSION = %q, exact release tag = %q", fileVersion, tag)
	}
}
