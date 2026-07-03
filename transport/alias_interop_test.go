package transport_test

// 该文件用外部测试包（transport_test）同时导入 llm 与 transport，验证类型下沉后：
//   1. transport 层是 NetworkPolicy/ProviderError/ErrNetworkPolicy 的本体，llm 只做别名；
//   2. llm.ErrNetworkPolicy 与 transport.ErrNetworkPolicy 是同一 sentinel；
//   3. errors.As 在 *llm.ProviderError 与 *transport.ProviderError 之间互通；
//   4. transport 包（非测试源文件）不再 import ai-core/llm，反向依赖已打破。

import (
	"context"
	"errors"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
	"github.com/hexagon-codes/ai-core/transport"
)

func TestErrNetworkPolicySentinelIsShared(t *testing.T) {
	if llm.ErrNetworkPolicy != transport.ErrNetworkPolicy {
		t.Fatal("llm.ErrNetworkPolicy and transport.ErrNetworkPolicy must be the same sentinel value")
	}

	// 一个真实网络策略违规错误应对两个别名的 errors.Is 都成立。
	err := transport.ValidateURL(context.Background(), "http://127.0.0.1/v1", &transport.NetworkPolicy{AllowHTTP: true})
	if err == nil {
		t.Fatal("expected a network policy violation")
	}
	if !errors.Is(err, llm.ErrNetworkPolicy) {
		t.Fatal("errors.Is(err, llm.ErrNetworkPolicy) must hold for a transport-produced violation")
	}
	if !errors.Is(err, transport.ErrNetworkPolicy) {
		t.Fatal("errors.Is(err, transport.ErrNetworkPolicy) must hold")
	}
}

func TestProviderErrorAliasIsInteroperableWithErrorsAs(t *testing.T) {
	// transport 产生的 *ProviderError 应能被 *llm.ProviderError 的 errors.As 捕获。
	srvErr := error(&transport.ProviderError{Provider: "test", Status: "429 Too Many Requests", Body: "slow down"})

	var viaLLM *llm.ProviderError
	if !errors.As(srvErr, &viaLLM) {
		t.Fatal("errors.As with *llm.ProviderError must catch a *transport.ProviderError")
	}
	var viaTransport *transport.ProviderError
	if !errors.As(srvErr, &viaTransport) {
		t.Fatal("errors.As with *transport.ProviderError must catch it too")
	}
	if viaLLM != viaTransport {
		t.Fatal("both aliases must resolve to the same underlying pointer")
	}
	// 旧字符串契约保持不变。
	if got := viaLLM.Error(); got != "test api error: 429 Too Many Requests, body: slow down" {
		t.Fatalf("ProviderError.Error() contract changed: %q", got)
	}
}

// 守护：transport 包的生产源文件（非 _test.go）绝不能 import ai-core/llm，
// 否则反向依赖回归、transport 无法独立下沉。
func TestTransportHasNoLLMImport(t *testing.T) {
	fset := token.NewFileSet()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, filepath.Join(".", name), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, imp := range file.Imports {
			if strings.Trim(imp.Path.Value, `"`) == "github.com/hexagon-codes/ai-core/llm" {
				t.Fatalf("%s imports ai-core/llm: reverse dependency reintroduced", name)
			}
		}
	}
}
