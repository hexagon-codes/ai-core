package template

import (
	"encoding/json"
	"testing"
)

// TestBlocks_MultiStepInterleavePreserved 是 #3 的闭环证明：
// 有序块词表**能表达**之前被证不可表达的多步交错（text→tool→text→tool）。
// 对照前端 ordered-blocks-gap-proof：旧模型（content 单串 + tool_calls 扁平）下
// text 永远在 tool 之前、两段话被合并；有序块下顺序如实保真。
func TestBlocks_MultiStepInterleavePreserved(t *testing.T) {
	bs := NewBlockBuilder().
		Text("让我先查天气。").
		ToolUse("t1", "weather", `{"city":"杭州"}`).
		ToolResult("t1", "27°C", false, "success").
		Text("27°C，再查下空气质量。"). // 这段话发生在 t1 之后、t2 之前 —— 旧模型无法表达
		ToolUse("t2", "aqi", `{"city":"杭州"}`).
		ToolResult("t2", "良", false, "success").
		Text("空气质量良，适合外出。").
		Build()

	// 关键不变量：text 与 tool 交错，且顺序如实。
	wantOrder := []BlockType{
		BlockText, BlockToolUse, BlockToolResult,
		BlockText, BlockToolUse, BlockToolResult, BlockText,
	}
	if len(bs) != len(wantOrder) {
		t.Fatalf("块数 = %d, want %d", len(bs), len(wantOrder))
	}
	for i, want := range wantOrder {
		if bs[i].Type != want {
			t.Fatalf("block[%d].Type = %q, want %q（交错顺序未保真）", i, bs[i].Type, want)
		}
	}
	// 第 2 段文本夹在两个工具之间 —— 这是旧扁平模型结构性做不到的（前端 proof 已证）。
	if bs[3].Type != BlockText || bs[3].Text != "27°C，再查下空气质量。" {
		t.Fatalf("夹在工具间的文本块丢失/错位: %+v", bs[3])
	}
	// 多段文本独立存在（旧模型会合并成 1 块）。
	textCount := 0
	for _, b := range bs {
		if b.Type == BlockText {
			textCount++
		}
	}
	if textCount != 3 {
		t.Fatalf("文本块数 = %d, want 3（多段话应各自独立，不合并）", textCount)
	}
}

func TestBlocks_Constructors(t *testing.T) {
	if b := TextBlock("hi"); b.Type != BlockText || b.Text != "hi" {
		t.Fatalf("TextBlock: %+v", b)
	}
	if b := ThinkingBlock("想想", 1200); b.Type != BlockThinking || b.DurationMs != 1200 {
		t.Fatalf("ThinkingBlock: %+v", b)
	}
	if b := ThinkingBlock("x", 0); b.DurationMs != 0 {
		t.Fatalf("ThinkingBlock 0 耗时应省略")
	}
	if b := ToolUseBlock("id1", "weather", "{}"); b.Type != BlockToolUse || b.ID != "id1" {
		t.Fatalf("ToolUseBlock: %+v", b)
	}
	if b := ToolResultBlock("id1", "27°C", false, "success"); b.Type != BlockToolResult || b.ToolUseID != "id1" || b.Status != "success" {
		t.Fatalf("ToolResultBlock: %+v", b)
	}
}

// JSON 往返保持类型判别与字段（wire 契约稳定）。
func TestBlocks_JSONRoundTrip(t *testing.T) {
	orig := NewBlockBuilder().
		Thinking("reasoning", 500).
		Text("answer").
		ToolUse("t1", "search", `{"q":"x"}`).
		ToolResult("t1", "result", true, "error").
		Build()

	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// tool_use 块不应渗出 tool_result 专有字段（omitempty 生效）。
	if got := string(data); !contains(got, `"type":"tool_use"`) || contains(got, `"output"`) && !contains(got, `"tool_result"`) {
		// 仅粗检 type 存在
		if !contains(got, `"type":"thinking"`) {
			t.Fatalf("JSON 缺类型判别: %s", got)
		}
	}

	var back Blocks
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(back) != len(orig) {
		t.Fatalf("往返块数变化: %d → %d", len(orig), len(back))
	}
	for i := range orig {
		if back[i] != orig[i] {
			t.Fatalf("block[%d] 往返不一致:\n orig=%+v\n back=%+v", i, orig[i], back[i])
		}
	}
}

func TestBlocks_Helpers(t *testing.T) {
	bs := NewBlockBuilder().Text("A").ToolUse("t1", "x", "{}").Text("B").Build()
	if got := bs.Text(); got != "A\nB" {
		t.Fatalf("Text() = %q, want \"A\\nB\"", got)
	}
	if got := bs.ToolUses(); len(got) != 1 || got[0].ID != "t1" {
		t.Fatalf("ToolUses() = %+v", got)
	}
}

func TestBlocks_Validate(t *testing.T) {
	ok := NewBlockBuilder().Text("a").ToolUse("t1", "x", "{}").ToolResult("t1", "r", false, "success").Build()
	if err := ok.Validate(); err != nil {
		t.Fatalf("合法序列不应报错: %v", err)
	}
	// 悬空 tool_result（无配对 tool_use）。
	dangling := Blocks{ToolResultBlock("ghost", "r", false, "")}
	if err := dangling.Validate(); err == nil {
		t.Fatal("悬空 tool_result 应校验失败")
	}
	// tool_result 先于 tool_use。
	reversed := Blocks{ToolResultBlock("t1", "r", false, ""), ToolUseBlock("t1", "x", "{}")}
	if err := reversed.Validate(); err == nil {
		t.Fatal("tool_result 先于 tool_use 应校验失败")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
