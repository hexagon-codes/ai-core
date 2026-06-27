package streamx

import (
	"encoding/json"
	"strings"
	"testing"
)

// 真机抓取（DeepSeek-V4-Pro / SiliconFlow，2026-06-27 会话→建任务路径）的原样 content：
// 工具调用以 ｜DSML｜ 命名空间的 invoke/parameter 标记泄漏进正文，tool_calls 字段为空。
// 私有标记 ｜ 为全宽竖线 U+FF5C。
const deepseekLeakedContent = "钳住了，马上给你安排 🦀\n\n" +
	"<｜DSML｜tool_calls>\n" +
	"<｜DSML｜invoke name=\"cron_task\">\n" +
	"<｜DSML｜parameter name=\"action\" string=\"true\">create</｜DSML｜parameter>\n" +
	"<｜DSML｜parameter name=\"name\" string=\"true\">百度热搜每日采集</｜DSML｜parameter>\n" +
	"<｜DSML｜parameter name=\"schedule\" string=\"true\">0 9 * * *</｜DSML｜parameter>\n" +
	"<｜DSML｜parameter name=\"description\" string=\"true\">每天早上9点自动采集百度热搜榜数据并写入知识库</｜DSML｜parameter>\n" +
	"<｜DSML｜parameter name=\"instructions\" string=\"true\">1. 访问百度热搜榜页面\n2. 提取热搜标题、排名、热度值\n3. 使用工具写入知识库</｜DSML｜parameter>\n" +
	"<｜DSML｜parameter name=\"enabled\" string=\"true\">true</｜DSML｜parameter>\n" +
	"</｜DSML｜invoke>\n" +
	"</｜DSML｜tool_calls>"

func TestRecoverInlineToolCalls_DeepSeekDSML(t *testing.T) {
	calls, cleaned, ok := RecoverInlineToolCalls(deepseekLeakedContent)
	if !ok {
		t.Fatalf("expected recovery ok=true; leaked tool call was dropped (the bug)")
	}
	if len(calls) != 1 {
		t.Fatalf("expected exactly 1 recovered call, got %d: %+v", len(calls), calls)
	}
	tc := calls[0]
	if tc.Name != "cron_task" {
		t.Errorf("tool name = %q, want cron_task", tc.Name)
	}
	if tc.Type != "function" {
		t.Errorf("tool type = %q, want function", tc.Type)
	}
	if tc.ID == "" {
		t.Errorf("recovered call must carry a non-empty ID (runtime correlates tool_result by ID)")
	}

	// Arguments 必须是合法 JSON，且字段值忠实还原。
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
		t.Fatalf("recovered arguments are not valid JSON: %v\n%s", err, tc.Arguments)
	}
	wantStr := map[string]string{
		"action":   "create",
		"name":     "百度热搜每日采集",
		"schedule": "0 9 * * *",
	}
	for k, want := range wantStr {
		got, _ := args[k].(string)
		if got != want {
			t.Errorf("arg %q = %q, want %q", k, got, want)
		}
	}
	// 多行参数值忠实保留换行。
	instr, _ := args["instructions"].(string)
	if !strings.Contains(instr, "1. 访问百度热搜榜页面") || !strings.Contains(instr, "3. 使用工具写入知识库") {
		t.Errorf("multiline 'instructions' value mangled: %q", instr)
	}
	// string="true" 提示被尊重：enabled 是字符串 "true"，不会被强转成布尔。
	if v, isStr := args["enabled"].(string); !isStr || v != "true" {
		t.Errorf("enabled should be string \"true\" (string=\"true\" hint), got %#v", args["enabled"])
	}

	// content 被剥干净，只剩模型的口语化正文，不再含任何泄漏标记。
	if cleaned != "钳住了，马上给你安排 🦀" {
		t.Errorf("cleaned content = %q, want %q", cleaned, "钳住了，马上给你安排 🦀")
	}
	if strings.Contains(cleaned, "DSML") || strings.Contains(cleaned, "tool_calls") {
		t.Errorf("cleaned content still leaks markup: %q", cleaned)
	}
}

// 真机抓取（DeepSeek-V4-Pro / SiliconFlow，2026-06-27 本会话）：同一模型另一轮以 Hermes/Nous
// 风格 <tool_call>{json}</tool_call> 泄漏工具调用，工具名键为 "tool"、arguments 为对象。
func TestRecoverInlineToolCalls_HermesDialect(t *testing.T) {
	content := "好的，马上为你创建！🦀\n\n" +
		"<tool_call>\n" +
		`{"tool": "cron_task", "arguments": {"action": "create", "name": "每日百度热搜采集", "schedule": "0 9 * * *", "prompt": "采集百度热搜榜并写入知识库"}}` + "\n" +
		"</tool_call>"
	calls, cleaned, ok := RecoverInlineToolCalls(content)
	if !ok || len(calls) != 1 {
		t.Fatalf("expected 1 recovered Hermes call, got ok=%v n=%d", ok, len(calls))
	}
	if calls[0].Name != "cron_task" {
		t.Errorf("tool name = %q, want cron_task", calls[0].Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Arguments), &args); err != nil {
		t.Fatalf("recovered args not valid JSON: %v\n%s", err, calls[0].Arguments)
	}
	if args["action"] != "create" || args["schedule"] != "0 9 * * *" || args["prompt"] == "" {
		t.Errorf("recovered Hermes args wrong/incomplete: %v", args)
	}
	if cleaned != "好的，马上为你创建！🦀" {
		t.Errorf("cleaned content = %q, want %q", cleaned, "好的，马上为你创建！🦀")
	}
	if strings.Contains(cleaned, "tool_call") {
		t.Errorf("cleaned content still leaks Hermes markup: %q", cleaned)
	}
}

// Hermes 变体：工具名键用 "name"、arguments 用字符串包裹的 JSON，也须正确还原。
func TestRecoverInlineToolCalls_HermesNameKeyStringArgs(t *testing.T) {
	content := "<tool_call>{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"北京\\\"}\"}</tool_call>"
	calls, _, ok := RecoverInlineToolCalls(content)
	if !ok || len(calls) != 1 || calls[0].Name != "get_weather" {
		t.Fatalf("want 1 get_weather call, got ok=%v calls=%+v", ok, calls)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Arguments), &args); err != nil {
		t.Fatalf("args not valid JSON: %v\n%s", err, calls[0].Arguments)
	}
	if args["city"] != "北京" {
		t.Errorf("city = %v, want 北京", args["city"])
	}
}

// hex-test 自审挑刺（2026-06-27）：模型讲解工具用法时把完整调用块贴进 ```代码围栏``` 作示例，
// 而本轮并没真调用（tool_calls==[]）——绝不能把"示例"当真调用还原，否则会凭空建出幽灵任务。
// 真机泄漏的调用都是裸标记（不在围栏内），故屏蔽围栏不漏真例。
func TestRecoverInlineToolCalls_IgnoresFencedExample(t *testing.T) {
	// Hermes 示例在围栏内
	hermes := "你可以这样调用：\n```\n<tool_call>\n{\"tool\":\"cron_task\",\"arguments\":{\"action\":\"create\"}}\n</tool_call>\n```\n以上仅为示例。"
	if calls, _, ok := RecoverInlineToolCalls(hermes); ok || len(calls) != 0 {
		t.Errorf("fenced Hermes example must NOT be recovered (phantom call); got ok=%v calls=%+v", ok, calls)
	}
	// 带语言标注的围栏同样屏蔽
	fenced2 := "示例：\n```json\n<tool_call>{\"name\":\"demo\",\"arguments\":{}}</tool_call>\n```"
	if _, _, ok := RecoverInlineToolCalls(fenced2); ok {
		t.Errorf("```json fenced example must NOT be recovered")
	}
	// 关键回归：围栏里是"示例"、围栏外是"真调用"——只还原围栏外的真调用，示例原样保留在正文。
	mixed := "先看示例：\n```\n<tool_call>{\"tool\":\"demo\",\"arguments\":{}}</tool_call>\n```\n现在真建：\n<tool_call>{\"tool\":\"cron_task\",\"arguments\":{\"action\":\"create\",\"schedule\":\"0 9 * * *\"}}</tool_call>"
	calls, cleaned, ok := RecoverInlineToolCalls(mixed)
	if !ok || len(calls) != 1 || calls[0].Name != "cron_task" {
		t.Fatalf("must recover ONLY the real (unfenced) call; got ok=%v calls=%+v", ok, calls)
	}
	if !strings.Contains(cleaned, "demo") || !strings.Contains(cleaned, "```") {
		t.Errorf("fenced example must remain in cleaned content (it's legit prose): %q", cleaned)
	}
}

func TestRecoverInlineToolCalls_NoMarkup(t *testing.T) {
	plain := "好的，我帮你处理。"
	calls, cleaned, ok := RecoverInlineToolCalls(plain)
	if ok || calls != nil || cleaned != plain {
		t.Fatalf("plain content must be untouched; got ok=%v calls=%+v cleaned=%q", ok, calls, cleaned)
	}
}

// 散文里恰好提到 invoke name= 但没有完整 invoke 块时，不得误判为工具调用。
func TestRecoverInlineToolCalls_ProseMentionOnly(t *testing.T) {
	prose := "你可以用形如 invoke name=\"foo\" 的标记写工具调用，但这只是说明文字，没有闭合块。"
	calls, _, ok := RecoverInlineToolCalls(prose)
	if ok || len(calls) != 0 {
		t.Fatalf("prose mention must not be parsed as a tool call; got ok=%v calls=%+v", ok, calls)
	}
}

// 多个 invoke 块（同一条 content 里两次调用）须按序还原，Index 递增、参数类型推断生效。
func TestRecoverInlineToolCalls_MultipleCalls(t *testing.T) {
	content := "<｜DSML｜invoke name=\"a\">" +
		"<｜DSML｜parameter name=\"x\" string=\"true\">1</｜DSML｜parameter>" +
		"</｜DSML｜invoke>" +
		"<｜DSML｜invoke name=\"b\">" +
		"<｜DSML｜parameter name=\"n\">42</｜DSML｜parameter>" +
		"</｜DSML｜invoke>"
	calls, _, ok := RecoverInlineToolCalls(content)
	if !ok || len(calls) != 2 {
		t.Fatalf("want 2 calls, got ok=%v n=%d", ok, len(calls))
	}
	if calls[0].Name != "a" || calls[1].Name != "b" {
		t.Errorf("order/name wrong: %q,%q", calls[0].Name, calls[1].Name)
	}
	if calls[0].Index != 0 || calls[1].Index != 1 {
		t.Errorf("Index not assigned per call: %d,%d", calls[0].Index, calls[1].Index)
	}
	if calls[0].ID == calls[1].ID {
		t.Errorf("recovered calls must have distinct IDs: %q", calls[0].ID)
	}
	// string="true" → 字符串 "1"；无提示的数字 → JSON 数字 42。
	var a, b map[string]any
	_ = json.Unmarshal([]byte(calls[0].Arguments), &a)
	_ = json.Unmarshal([]byte(calls[1].Arguments), &b)
	if v, ok := a["x"].(string); !ok || v != "1" {
		t.Errorf("a.x should be string \"1\", got %#v", a["x"])
	}
	if v, ok := b["n"].(float64); !ok || v != 42 {
		t.Errorf("b.n should be number 42, got %#v", b["n"])
	}
}
