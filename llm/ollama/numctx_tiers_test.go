package ollama

// BUG-20260710 P0 · num_ctx 恒 32768 的代价(真机实测):同模型同问题驻留 +1.03GB 纯脏页
// (KV/buffer 预分配),且 22-token 小问题首推理 230s vs 74s(3×慢)。原生 Ollama 默认 4096。
//
// 修复契约(分档 + Provider 生命周期内只升不降):
//   1. metadata 显式 num_ctx 依旧最高优先(原契约不变,不入粘性水位);
//   2. 自动档从 {4096,8192,16384,32768} 里选:∑(消息+工具)估算 tokens + 输出预算(MaxTokens,
//      未设按 2048)+ 余量,取最小可容纳档;
//   3. **同模型粘性只升不降**(水位是 Provider 级、跨会话共享):num_ctx 变化会触发
//      Ollama runner 重载模型,乱跳比大 ctx 更糟;
//   4. 模型自身 MaxTokens 仍是上限(8k 模型不发 16k),且硬顶优先于粘性水位。
import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func mkReq(model string, contentLen int, maxTokens int) llm.CompletionRequest {
	return llm.CompletionRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Messages: []llm.Message{
			{Role: "system", Content: strings.Repeat("规", contentLen/2)},
			{Role: "user", Content: strings.Repeat("问", contentLen/2)},
		},
	}
}

func numCtxForSuccessfulRequest(p *Provider, req llm.CompletionRequest) int {
	numCtx := p.numCtxForRequest(req)
	if _, explicit := ollamaNumCtxFromMetadata(req.Metadata); !explicit {
		p.commitNumCtx(req.Model, numCtx)
	}
	return numCtx
}

func TestNumCtx_TierSmallChat(t *testing.T) {
	p := New()
	// 小对话(约 400 字符 ≈ 数百 token)+ 默认输出预算 → 最小档 4096
	got := p.numCtxForRequest(mkReq("qwen3.5:9b", 400, 0))
	if got != 4096 {
		t.Fatalf("小对话应选 4096 档,实际 %d", got)
	}
}

func TestNumCtxTierExactBoundaries(t *testing.T) {
	tests := []struct {
		needed int
		want   int
	}{
		{needed: 0, want: 4096},
		{needed: 4096, want: 4096},
		{needed: 4097, want: 8192},
		{needed: 8192, want: 8192},
		{needed: 8193, want: 16384},
		{needed: 16384, want: 16384},
		{needed: 16385, want: 32768},
		{needed: 32768, want: 32768},
		{needed: 32769, want: 32768},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("needed_%d", tt.needed), func(t *testing.T) {
			if got := automaticNumCtxTier(tt.needed); got != tt.want {
				t.Fatalf("automaticNumCtxTier(%d) = %d, want %d", tt.needed, got, tt.want)
			}
		})
	}
}

func TestNumCtx_NonTierModelCaps(t *testing.T) {
	tests := []int{6000, 10000}
	for _, capLimit := range tests {
		t.Run(fmt.Sprintf("cap_%d", capLimit), func(t *testing.T) {
			p := New()
			p.models = []llm.ModelInfo{{ID: "custom-cap", MaxTokens: capLimit}}
			if got := p.numCtxForRequest(mkReq("custom-cap", 20000, 0)); got != capLimit {
				t.Fatalf("num_ctx with model cap %d = %d, want %d", capLimit, got, capLimit)
			}
		})
	}
}

func TestNumCtx_TierGrowsWithPrompt(t *testing.T) {
	p := New()
	// 中文按 tokenizer 口径 ≈0.67 token/字:9000 字 ≈6000 + 2048 输出预算 + 512 余量 → >8192 → 16384 档
	got := p.numCtxForRequest(mkReq("qwen3.5:9b", 9000, 0))
	if got != 16384 {
		t.Fatalf("中大 prompt 应选 16384 档,实际 %d", got)
	}
	// 巨大 prompt → 顶到 32768
	got = p.numCtxForRequest(mkReq("qwen3.5:9b", 120000, 0))
	if got != 32768 {
		t.Fatalf("超大 prompt 应封顶 32768,实际 %d", got)
	}
}

func TestNumCtx_ExplicitMetadataWins(t *testing.T) {
	p := New()
	req := mkReq("qwen3.5:9b", 400, 0)
	req.Metadata = map[string]any{"num_ctx": 2048}
	if got := p.numCtxForRequest(req); got != 2048 {
		t.Fatalf("显式 metadata 应原样生效(可低于自动档),实际 %d", got)
	}
	// 显式不改变自动粘性水位
	if got := p.numCtxForRequest(mkReq("qwen3.5:9b", 400, 0)); got != 4096 {
		t.Fatalf("显式请求后自动档应不受影响,实际 %d", got)
	}
}

func TestNumCtx_StickyOnlyUp(t *testing.T) {
	p := New()
	if got := numCtxForSuccessfulRequest(p, mkReq("m1", 400, 0)); got != 4096 {
		t.Fatalf("first small: %d", got)
	}
	if got := numCtxForSuccessfulRequest(p, mkReq("m1", 9000, 0)); got != 16384 {
		t.Fatalf("grow: %d", got)
	}
	// 回到小 prompt:粘性只升不降(防 runner 反复重载)
	if got := numCtxForSuccessfulRequest(p, mkReq("m1", 400, 0)); got != 16384 {
		t.Fatalf("sticky 应保持 16384,实际 %d", got)
	}
	// 不同模型互不影响
	if got := numCtxForSuccessfulRequest(p, mkReq("m2", 400, 0)); got != 4096 {
		t.Fatalf("m2 应独立从 4096 起,实际 %d", got)
	}
}

func TestNumCtx_ModelCapRespected(t *testing.T) {
	p := New()
	p.models = []llm.ModelInfo{{ID: "small8k", MaxTokens: 8192}} // 同包注入,模拟 fetchLocalModels 结果
	if got := p.numCtxForRequest(mkReq("small8k", 120000, 0)); got != 8192 {
		t.Fatalf("模型上限 8192 应封顶,实际 %d", got)
	}
}

func TestNumCtx_MaxTokensBudgetCounted(t *testing.T) {
	p := New()
	// 小 prompt 但要求 30000 输出 → 档位必须容纳输出预算
	if got := p.numCtxForRequest(mkReq("m3", 400, 30000)); got != 32768 {
		t.Fatalf("大输出预算应抬档到 32768,实际 %d", got)
	}
}

func TestNumCtx_ConcurrentStickyRace(t *testing.T) {
	p := New()
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			numCtxForSuccessfulRequest(p, mkReq(fmt.Sprintf("m%d", i%4), 400+i*500, 0))
		}(i)
	}
	wg.Wait() // -race 下无数据竞争即过
}
