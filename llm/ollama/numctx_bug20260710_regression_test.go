package ollama

// BUG-20260710 review 发现的回归锁定测试(F1/F2/F3/F5 + 高级测试方法矩阵 M1/M9/M10)。
//
// 闭环纪律(devtestops Bug 修复专属闭环):每条 F* 测试先在未修复代码上跑出 RED
// (失败信息反映 bug 具体症状),修复后转 GREEN,永久保留防回归。
//
//   F1 粘性水位绕过模型硬顶     — 模型列表晚于大请求加载时,历史水位可超过模型 cap
//   F2 估算漏算 tool schema/图片 — 档位边界处 num_ctx 偏小 → Ollama 静默截断 prompt
//   F3 输出预算契约锚(2048)     — 注释一度误写 4096,防止后人"按注释纠正代码"
//   F5 p.models 无锁读写竞争    — 并发 Models() + Complete 触发 data race(-race 抓)
//   M1 不变量 fuzz(2000 样本)    — 差分护栏:自动档永不高于老代码(恒 clamp(cap));粘性单调
//   M9 metadata 兼容性矩阵      — 4 键 × 8 类型 × 反例,显式 num_ctx 契约
//   M10 性能基线                — BenchmarkNumCtxForRequest_*(估算引入的每请求开销)

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

// ── F1 ──────────────────────────────────────────────────────────────────────

// TestBUG20260710_F1_StickyMustNotBypassModelCap
// 症状:先来大请求(模型列表未加载)→ 水位记 32768;之后模型列表加载出 cap=8192,
// 小请求仍发 num_ctx=32768,违反"8k 模型不发 16k"契约。
func TestBUG20260710_F1_StickyMustNotBypassModelCap(t *testing.T) {
	p := New()
	// 1) Models() 尚未被调用(p.models 为空)时来了超大请求 → 水位记 32768
	if got := numCtxForSuccessfulRequest(p, mkReq("small8k", 400, 30000)); got != 32768 {
		t.Fatalf("前置: 大请求应选 32768, 实际 %d", got)
	}
	// 2) 模型列表加载完成,该模型上限 8192
	p.models = []llm.ModelInfo{{ID: "small8k", MaxTokens: 8192}}
	// 3) 小请求必须被硬顶钳制(粘性水位不得绕过 cap)
	if got := numCtxForSuccessfulRequest(p, mkReq("small8k", 400, 0)); got != 8192 {
		t.Fatalf("BUG-20260710-F1: 粘性水位绕过模型硬顶, num_ctx=%d, 期望钳到 8192", got)
	}
	// 4) 水位应被钳制值直接回写(不能只在每次返回前临时 clamp)
	p.stickyNumCtxMu.Lock()
	stored := p.stickyNumCtx[stickyNumCtxKey("small8k")]
	p.stickyNumCtxMu.Unlock()
	if stored != 8192 {
		t.Fatalf("BUG-20260710-F1: 脏水位未自愈, 缓存值=%d", stored)
	}
}

// ── F2 ──────────────────────────────────────────────────────────────────────

// TestBUG20260710_F2_ToolSchemaCountedInEstimate
// 症状:估算只累加 tool.Function.Name+Description,不计 Parameters(完整 JSON schema
// 会随 payload["tools"] 发给 Ollama)。大 schema 场景估算偏小 → 选档偏低 → 实际
// prompt 超 num_ctx 时被 Ollama 静默截断(丢最老消息/system prompt)。
// 数值设计(tokenizer 实测中文 0.667 token/字):schema 内嵌 15000 字 ≈ 10000 token,
// needed ≈ 10000+2048+512 ≈ 12600 → 必须选 16384 档;漏算则落 4096 档。
func TestBUG20260710_F2_ToolSchemaCountedInEstimate(t *testing.T) {
	p := New()
	bigSchema := &llm.Schema{Type: "object", Description: strings.Repeat("图", 15000)}
	req := mkReq("toolheavy", 100, 0)
	req.Tools = []llm.ToolDefinition{llm.NewToolDefinition("gen_image", "生成图片", bigSchema)}
	if got := p.numCtxForRequest(req); got != 16384 {
		t.Fatalf("BUG-20260710-F2: tool Parameters schema 未计入估算, num_ctx=%d, 期望 16384", got)
	}
}

// TestBUG20260710_F2_ImagesCountedInEstimate
// 症状:MultiContent 中的图片不占估算。vision 模型每图消耗数百 token,
// 多图请求选档偏低同样触发静默截断。
// 数值设计:8 图的保守预算超过自动上限，因此应封顶到 32768 档。
func TestBUG20260710_F2_ImagesCountedInEstimate(t *testing.T) {
	p := New()
	parts := []llm.ContentPart{{Type: "text", Text: "看图"}}
	for i := 0; i < 8; i++ {
		parts = append(parts, llm.ContentPart{
			Type:     "image_url",
			ImageURL: &llm.ImageURL{URL: "data:image/png;base64,QUJDRA=="},
		})
	}
	req := llm.CompletionRequest{
		Model:    "vision-model",
		Messages: []llm.Message{{Role: llm.RoleUser, MultiContent: parts}},
	}
	if got := p.numCtxForRequest(req); got != 32768 {
		t.Fatalf("BUG-20260710-F2: 图片 token 未计入估算, num_ctx=%d, 期望 32768", got)
	}
}

// ── F3 ──────────────────────────────────────────────────────────────────────

// TestBUG20260710_F3_OutputBudgetContractAnchor
// 契约锚:默认输出预算 = 2048(注释一度误写 4096)。若有人"按注释纠正代码"为 4096,
// 所有小对话将抬到 8192 档,本测试与 TestNumCtx_TierSmallChat 同时 FAIL。
func TestBUG20260710_F3_OutputBudgetContractAnchor(t *testing.T) {
	if got := outputBudget(llm.CompletionRequest{}); got != 2048 {
		t.Fatalf("BUG-20260710-F3: 默认输出预算契约为 2048, 实际 %d", got)
	}
	if got := outputBudget(llm.CompletionRequest{MaxTokens: 777}); got != 777 {
		t.Fatalf("显式 MaxTokens 应原样作为输出预算, 实际 %d", got)
	}
}

// ── F5 ──────────────────────────────────────────────────────────────────────

// TestBUG20260710_F5_ModelsVsNumCtxDataRace
// 症状:Models() 无锁写 p.models,numCtxForRequest 无锁读 → 并发 Complete+Models
// 构成 data race(go test -race 抓)。pre-existing,本次为 numCtxForRequest 引入
// 锁时一并纳入保护。
func TestBUG20260710_F5_ModelsVsNumCtxDataRace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"models":[{"model":"m8k","name":"m8k","capabilities":["tools"],"details":{"context_length":8192}}]}`)
	}))
	defer srv.Close()

	for i := 0; i < 20; i++ {
		p := New(WithBaseURL(srv.URL))
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			p.Models() // 写 p.models
		}()
		go func() {
			defer wg.Done()
			p.numCtxForRequest(mkReq("m8k", 400, 0)) // 读 p.models
		}()
		wg.Wait()
	}
}

// ── M1 不变量 fuzz(方法 1 property-based + 方法 5 差分护栏)─────────────────

// TestBUG20260710_M1_PropertyInvariants 固定种子 2000 样本,断言三条不变量:
//
//	I1 差分:自动档结果 ≤ 老代码结果(老代码恒 clamp(model.MaxTokens),未知模型 32768)
//	   —— 保证新逻辑在任何输入下内存占用都不劣于修复前。
//	I2 容量:结果必须容纳 min(估算需求,模型 cap)。
//	I3 值域:结果 ∈ 分档表 ∪ {模型 cap}。
func TestBUG20260710_M1_PropertyInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(20260710))
	models := []llm.ModelInfo{
		{ID: "cap8k", MaxTokens: 8192},
		{ID: "cap2k", MaxTokens: 2048},
		{ID: "big", MaxTokens: 262144},
	}
	oldCode := map[string]int{"cap8k": 8192, "cap2k": 2048, "big": 32768, "unknown": 32768}
	names := []string{"cap8k", "cap2k", "big", "unknown"}

	for i := 0; i < 2000; i++ {
		p := New()
		p.models = models
		name := names[rng.Intn(len(names))]
		req := mkReq(name, rng.Intn(30000), rng.Intn(3)*rng.Intn(8192))
		got := p.numCtxForRequest(req)

		if got > oldCode[name] { // I1
			t.Fatalf("样本 %d: 差分违反, model=%s num_ctx=%d > 老代码 %d", i, name, got, oldCode[name])
		}
		needed, _ := automaticNumCtxNeeded(req)
		required := min(needed, oldCode[name])
		if got < required { // I2
			t.Fatalf("样本 %d: 容量不足, model=%s num_ctx=%d < required=%d", i, name, got, required)
		}
		switch got { // I3
		case 4096, 8192, 16384, 32768, oldCode[name]:
		default:
			t.Fatalf("样本 %d: 非法档位 %d", i, got)
		}
	}
}

// ── M9 metadata 兼容性矩阵(方法 9)──────────────────────────────────────────

func TestBUG20260710_M9_MetadataCompatMatrix(t *testing.T) {
	keys := []string{"num_ctx", "ollama_num_ctx", "context_length", "max_context_tokens"}
	positives := []any{int(3000), int64(3000), int32(3000), float64(3000), float32(3000),
		json.Number("3000"), "3000", " 3000 "}
	for _, key := range keys {
		for _, val := range positives {
			p := New()
			req := mkReq("m", 400, 0)
			req.Metadata = map[string]any{key: val}
			if got := p.numCtxForRequest(req); got != 3000 {
				t.Errorf("key=%s val=%T(%v): 期望显式 3000, 实际 %d", key, val, val, got)
			}
		}
	}
	// 反例:非法/非正值回落自动档(小请求 4096),不 panic
	for _, val := range []any{0, -1, "abc", nil, true, 0.5, 3.7} {
		p := New()
		req := mkReq("m", 400, 0)
		req.Metadata = map[string]any{"num_ctx": val}
		got := p.numCtxForRequest(req)
		if got != 4096 {
			t.Errorf("val=%v: 非法值应回落自动档 4096, 实际 %d", val, got)
		}
	}
}

// ── M10 性能基线(方法 10)───────────────────────────────────────────────────

func BenchmarkNumCtxForRequest_Small2KB(b *testing.B) {
	p := New()
	req := mkReq("bench", 2000, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.numCtxForRequest(req)
	}
}

func BenchmarkNumCtxForRequest_Large200KB(b *testing.B) {
	p := New()
	req := mkReq("bench", 200000, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.numCtxForRequest(req)
	}
}
