//go:build ollama_e2e

package ollama

// BUG-20260710 真机 Ollama E2E(需本机 Ollama 服务 + qwen3.5:9b 或 OLLAMA_E2E_MODEL 指定模型):
//
//	OLLAMA_E2E=1 GOTOOLCHAIN=auto go test -tags ollama_e2e -v -run TestE2E_NumCtx -timeout 30m ./llm/ollama/
//
// 覆盖(2026-07-10 Intel i7-8850H/16GB 纯 CPU 真机取证,数据见测试断言):
//	E1 最小档 4096 实发(/api/ps context_length=4096)+ 默认 keep_alive=30m
//	E5 32768 档内存差(真机实测 +1.103GB: 6.223GB→7.326GB,复现 BUG 取证 +1.03GB)
//	E6 粘性水位防重载(小请求保持 32768,延迟 1.3s vs 重载 36-54s)
//	E2 metadata keep_alive=2m 精确覆盖默认值
//	E3 显式 num_ctx 低于实际 prompt → Ollama 静默截断(WARN 日志 limit=1026 prompt=2524
//	   keep=4;客户端 prompt_eval=1026 无任何错误)——F2 修复(估算计入 schema/图片)的风险实证
//	E4 ⚪ 本机模型 n_ctx_train=262144 ≥ 最高档 32768,"超训练上限"场景不可构造;
//	   自动档经 clampAutomaticNumCtx 恒 ≤32768,现代模型 train ctx 均 ≥32k,风险面仅剩
//	   显式 metadata(显式即契约)。
//
// 附带发现:(a) Ollama 会把过小的显式 num_ctx 钳到服务端最小值 2048;(b) 溢出截断上限
// ≈ n_ctx/2(shift 规则),仅在 prompt 超过 n_ctx 后触发,选档公式保证 prompt<档位故不触发;
// (c) num_ctx 变更触发的重载+长预填充可能超过非流式 120s response-header 超时
// (ollamaDefaultResponseHeaderWait),故 E3 含一次重试。

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hexagon-codes/ai-core/llm"
)

type psModel struct {
	Name          string    `json:"name"`
	Size          int64     `json:"size"`
	ExpiresAt     time.Time `json:"expires_at"`
	ContextLength int       `json:"context_length"`
}

func e2eBaseURL() string {
	if v := os.Getenv("OLLAMA_HOST"); v != "" {
		return v
	}
	return "http://localhost:11434"
}

func e2ePS(t *testing.T) psModel {
	t.Helper()
	resp, err := http.Get(e2eBaseURL() + "/api/ps")
	if err != nil {
		t.Fatalf("/api/ps 失败: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Models []psModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("/api/ps 解析失败: %v", err)
	}
	if len(out.Models) == 0 {
		t.Fatal("/api/ps 无已加载模型")
	}
	return out.Models[0]
}

func TestE2E_NumCtx(t *testing.T) {
	if os.Getenv("OLLAMA_E2E") == "" {
		t.Skip("需真实 Ollama：设 OLLAMA_E2E=1 运行")
	}
	if _, err := http.Get(e2eBaseURL() + "/api/version"); err != nil {
		t.Skipf("本机 Ollama 不可达(%v),跳过真机 E2E", err)
	}
	model := os.Getenv("OLLAMA_E2E_MODEL")
	if model == "" {
		model = "qwen3.5:9b"
	}
	p := New(WithModel(model))

	small := func(maxTok int, meta map[string]any) llm.CompletionRequest {
		if meta == nil {
			meta = map[string]any{}
		}
		meta["think"] = false
		return llm.CompletionRequest{
			Model: model, MaxTokens: maxTok, Metadata: meta,
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "只用一个字回答:1+1=?"}},
		}
	}
	complete := func(tag string, req llm.CompletionRequest, retries int) (*llm.CompletionResponse, time.Duration) {
		t.Helper()
		var lastErr error
		for attempt := 0; attempt <= retries; attempt++ {
			ctx, cancel := context.WithTimeout(context.Background(), 25*time.Minute)
			t0 := time.Now()
			resp, err := p.Complete(ctx, req)
			el := time.Since(t0)
			cancel()
			if err == nil {
				t.Logf("[%s] latency=%.1fs prompt_eval=%d eval=%d", tag, el.Seconds(),
					resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
				return resp, el
			}
			lastErr = err
			t.Logf("[%s] 第 %d 次失败(%.1fs): %v — 重载+长预填充可能超 120s header 超时,重试",
				tag, attempt+1, el.Seconds(), err)
		}
		t.Fatalf("[%s] 重试后仍失败: %v", tag, lastErr)
		return nil, 0
	}

	// E1: 小请求 → 最小档 4096 实发 + 默认 keep_alive=30m
	before := time.Now()
	complete("E1", small(8, nil), 1)
	ps1 := e2ePS(t)
	if ps1.ContextLength != 4096 {
		t.Errorf("E1: 期望 num_ctx=4096 实发, /api/ps context_length=%d", ps1.ContextLength)
	}
	if d := ps1.ExpiresAt.Sub(before); d < 28*time.Minute || d > 32*time.Minute {
		t.Errorf("E1: 默认 keep_alive 应 ≈30m, 实际 %.1fm", d.Minutes())
	}

	// E5: MaxTokens=30000 → 32768 档,取内存差(BUG-20260710 P0 的 +1GB 主张)
	complete("E5", small(30000, nil), 1)
	ps5 := e2ePS(t)
	if ps5.ContextLength != 32768 {
		t.Errorf("E5: 期望 num_ctx=32768, 实际 %d", ps5.ContextLength)
	}
	deltaGB := float64(ps5.Size-ps1.Size) / (1 << 30)
	t.Logf("E5: 32768 档 vs 4096 档驻留差 = %.3f GB(2026-07-10 真机 1.103GB)", deltaGB)
	if deltaGB < 0.5 {
		t.Errorf("E5: 32768 档应显著多驻留内存(≥0.5GB), 实际 %.3fGB", deltaGB)
	}

	// E6: 粘性水位 — 小请求保持 32768,无重载(低延迟)
	_, el6 := complete("E6", small(8, nil), 0)
	ps6 := e2ePS(t)
	if ps6.ContextLength != 32768 {
		t.Errorf("E6: 粘性应保持 32768, 实际 %d", ps6.ContextLength)
	}
	if el6 > 15*time.Second {
		t.Errorf("E6: 无重载请求应秒回(2026-07-10 真机 1.3s), 实际 %.1fs — 疑似发生重载", el6.Seconds())
	}

	// E2: metadata keep_alive=2m 覆盖默认 30m
	before2 := time.Now()
	complete("E2", small(8, map[string]any{"keep_alive": "2m"}), 0)
	ps2 := e2ePS(t)
	if d := ps2.ExpiresAt.Sub(before2); d < time.Minute || d > 3*time.Minute {
		t.Errorf("E2: metadata keep_alive=2m 应生效, 实际 %.1fm", d.Minutes())
	}

	// E3: 显式 num_ctx=1024 + ~2500 token prompt → Ollama 静默截断
	// (服务端钳到最小 2048,溢出后按 shift 规则截到 ≈n_ctx/2;客户端无错误)
	poem := strings.Repeat("床前明月光,疑是地上霜。", 250)
	resp3, _ := complete("E3", llm.CompletionRequest{
		Model: model, MaxTokens: 8,
		Metadata: map[string]any{"num_ctx": 1024, "think": false},
		Messages: []llm.Message{{Role: llm.RoleUser,
			Content: poem + " 只用一个字回答:上面重复的诗句第一个字是什么?"}},
	}, 2)
	if resp3.Usage.PromptTokens >= 2000 {
		t.Errorf("E3: 期望静默截断(prompt_eval<2000, 2026-07-10 真机 1026/2524), 实际 %d — Ollama 截断行为变了?", resp3.Usage.PromptTokens)
	}
	t.Logf("E3: 实发≈2524 token, 服务端仅计 prompt_eval=%d — 静默截断实锤(客户端无任何错误)", resp3.Usage.PromptTokens)
	fmt.Println("E2E 全部完成")
}
