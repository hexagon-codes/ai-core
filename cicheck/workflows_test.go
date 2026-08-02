package cicheck

import (
	"go/version"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

func TestCIWorkflowsPinPatchedGoToolchain(t *testing.T) {
	t.Parallel()

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate test file")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(currentFile), ".."))
	workflows, err := filepath.Glob(filepath.Join(root, ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatalf("glob workflows: %v", err)
	}
	if len(workflows) == 0 {
		t.Fatal("no GitHub Actions workflows found")
	}

	versionPattern := regexp.MustCompile(`go-version:\s*['"]?([^'"}\s,]+)`)
	exactVersionPattern := regexp.MustCompile(`^\d+\.\d+\.\d+$`)
	const minimumVersion = "1.25.12"
	setupCount := 0
	versionCount := 0
	for _, workflow := range workflows {
		body, readErr := os.ReadFile(workflow)
		if readErr != nil {
			t.Fatalf("read %s: %v", workflow, readErr)
		}
		setupCount += strings.Count(string(body), "uses: actions/setup-go@")
		for _, match := range versionPattern.FindAllStringSubmatch(string(body), -1) {
			versionCount++
			pinnedVersion := match[1]
			if !exactVersionPattern.MatchString(pinnedVersion) {
				t.Errorf("%s uses non-exact Go version %q; want a reproducible patch version", filepath.Base(workflow), pinnedVersion)
				continue
			}
			if version.Compare("go"+pinnedVersion, "go"+minimumVersion) < 0 {
				t.Errorf("%s pins vulnerable Go %q; want at least %s", filepath.Base(workflow), pinnedVersion, minimumVersion)
			}
		}
	}
	if versionCount != setupCount {
		t.Fatalf("found %d setup-go steps but %d pinned go-version values", setupCount, versionCount)
	}
}

func TestCIWorkflowsPinThirdPartyActionsAndGoToolsImmutably(t *testing.T) {
	t.Parallel()

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate test file")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(currentFile), ".."))
	workflows, err := filepath.Glob(filepath.Join(root, ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatalf("glob workflows: %v", err)
	}
	if len(workflows) == 0 {
		t.Fatal("no GitHub Actions workflows found")
	}

	actionPattern := regexp.MustCompile(`uses:\s*([^@\s]+)@([^\s#]+)`)
	immutableRevision := regexp.MustCompile(`^[0-9a-f]{40}$`)
	for _, workflow := range workflows {
		body, readErr := os.ReadFile(workflow)
		if readErr != nil {
			t.Fatalf("read %s: %v", workflow, readErr)
		}
		for _, match := range actionPattern.FindAllStringSubmatch(string(body), -1) {
			if strings.HasPrefix(match[1], "./") {
				continue
			}
			if !immutableRevision.MatchString(match[2]) {
				t.Errorf("%s uses mutable action %s@%s; want a full commit SHA", filepath.Base(workflow), match[1], match[2])
			}
		}
		if strings.Contains(string(body), "go install") && strings.Contains(string(body), "@latest") {
			t.Errorf("%s installs a Go tool at mutable @latest", filepath.Base(workflow))
		}
	}
}
