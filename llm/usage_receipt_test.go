package llm_test

import (
	"testing"

	"github.com/hexagon-codes/ai-core/llm"
)

func TestUsageReceiptFromUsage(t *testing.T) {
	if receipt := llm.UsageReceiptFromUsage(llm.Usage{}); receipt != nil {
		t.Fatalf("zero usage receipt = %+v, want nil", receipt)
	}
	usage := llm.Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10}
	receipt := llm.UsageReceiptFromUsage(usage)
	if receipt == nil || *receipt != usage {
		t.Fatalf("receipt = %+v, want %+v", receipt, usage)
	}
	receipt.PromptTokens = 99
	if usage.PromptTokens != 7 {
		t.Fatalf("receipt mutated original usage: %+v", usage)
	}
}
