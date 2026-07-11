package ollama

// AP-207 回归锁(hex-test 20260710)：provider 单元测试禁止连本机真 Ollama。
//
// 根因：ollama.New() 默认 baseURL=localhost:11434,Models()/numCtxForRequest 会真连
// /api/tags 取模型 cap。测试包无 TestMain 隔离 → 开发机跑着真 Ollama 时,Models()
// 返回随真机状态漂移的模型列表 → num_ctx cap 计算非确定 → 用 New() 默认的用例
// 随真机在线与否 flaky(审计期间 -shuffle=on 实测 TestVisionRequestUsesConservativeImageBudget
// 等 FAIL;死端口隔离后连跑 3× 稳定绿)。
//
// TestMain 把 OLLAMA_HOST 钉到死端口,让全包默认 New() 走 fallback 不连真机;需要真机的
// 端到端门用 build tag ollama_e2e 显式 opt-in(numctx_bug20260710_e2e_test.go)。

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	// 显式 E2E 保留调用方指定的真实地址；默认单测钉到死端口保持封闭。
	if os.Getenv("OLLAMA_E2E") == "" {
		os.Setenv("OLLAMA_HOST", "http://127.0.0.1:1")
	}
	os.Exit(m.Run())
}

// TestAP207_DefaultProviderIsHermetic 封闭性契约:默认 Provider 绝不返回真机独有模型。
// RED(无 TestMain + 真机在线):New().Models() 连真机返回 qwen3.5:9b → FAIL。
// GREEN(TestMain 死端口):连不上 → fallback 列表(llama3.2…) 不含真机模型 → PASS。
func TestAP207_DefaultProviderIsHermetic(t *testing.T) {
	models := New().Models()
	for _, mi := range models {
		// qwen3.5:9b 是本机真装模型,不在 Provider 的 fallback 默认列表里;
		// 若出现说明测试连到了真机 /api/tags,即非封闭。
		if mi.ID == "qwen3.5:9b" || mi.Name == "qwen3.5:9b" {
			t.Fatalf("AP-207: 默认 Provider 连到了本机真 Ollama(Models() 含 %q),单元测试非封闭", mi.ID)
		}
	}
}
