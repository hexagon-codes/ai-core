package streamx

import (
	"encoding/json"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// 部分模型（典型为 DeepSeek 经 SiliconFlow 等 OpenAI 兼容网关）会把工具调用以**纯文本
// 标记**塞进消息 content，而不是放进结构化 tool_calls 字段。网关没有把这套方言翻译成
// OpenAI tool_calls，于是响应到手时 tool_calls=[]、标记泄漏进 content，运行时判定"没有
// 工具调用"，用户的真实意图（如建定时任务）被当成普通文本丢弃。
//
// 真机抓取（DeepSeek-V4-Pro / SiliconFlow）发现**两种**泄漏方言，且同一模型按轮随机切换：
//
//  1. antml 风格（｜DSML｜ 命名空间，｜ 为全宽竖线 U+FF5C）：
//
//     <｜DSML｜tool_calls>
//     <｜DSML｜invoke name="cron_task">
//     <｜DSML｜parameter name="action" string="true">create</｜DSML｜parameter>
//     </｜DSML｜invoke>
//     </｜DSML｜tool_calls>
//
//  2. Hermes/Nous 风格（<tool_call> 包裹 JSON）：
//
//     <tool_call>
//     {"tool": "cron_task", "arguments": {"action": "create", "schedule": "0 9 * * *"}}
//     </tool_call>
//
// RecoverInlineToolCalls 同时识别这两种方言，把内嵌标记还原为结构化工具调用，并返回剥掉
// 标记后的 content。调用方应仅在结构化 tool_calls 为空时调用，避免误改模型既有合法 content。
//
// ★误报防护：```代码围栏``` 里的标记被视为"展示/讲解"而非真调用（模型在解释工具用法时会
// 贴完整示例块），扫描前先把围栏区域屏蔽——真机泄漏的调用都是裸标记，不在围栏内，故不漏真例。
//
// 反模式锁：provider 把工具调用以私有标记内嵌在 content 而非结构化字段时被静默丢弃（AP-111）。
var (
	// antml 方言：一个 invoke 块，捕获工具名(g1)与块体(g2)。(?s) 让 . 跨行匹配多行参数。
	inlineInvokeRe = regexp.MustCompile(`(?s)<[^>]*?\binvoke\s+name="([^"]+)"[^>]*>(.*?)</[^>]*?\binvoke\s*>`)
	// antml 方言：一个 parameter，捕获参数名(g1)、其余属性(g2，含 string="true" 类型提示)、值(g3)。
	inlineParamRe = regexp.MustCompile(`(?s)<[^>]*?\bparameter\s+name="([^"]+)"([^>]*?)>(.*?)</[^>]*?\bparameter\s*>`)
	// antml 方言：<…tool_calls> / </…tool_calls> 包裹标签（还原后一并剥除）。
	inlineWrapRe = regexp.MustCompile(`</?[^>]*?\btool_calls\b[^>]*>`)
	// Hermes 方言：<tool_call> … </tool_call> 包裹一段 JSON。非贪婪到首个闭合标签即可正确
	// 截取嵌套 JSON（闭合标签是可靠分隔符，JSON 本身不含该字面串）。
	hermesBlockRe = regexp.MustCompile(`(?s)<tool_call>\s*(.+?)\s*</tool_call>`)
	// ```代码围栏```（含语言标注）。屏蔽其内部，避免把"示例代码"当真调用。
	fenceRe = regexp.MustCompile("(?s)```.*?```")
)

// RecoverInlineToolCalls 从 content 内嵌的工具调用标记还原结构化工具调用。
//
// 返回 (calls, cleaned, ok)：ok=true 时 calls 为还原出的工具调用、cleaned 为剥掉标记后的
// 正文；ok=false 时 content 原样返回、calls 为 nil。两种方言互斥，按 antml→Hermes 顺序尝试。
// 扫描在"屏蔽代码围栏后的副本"上做（索引与原文对齐），围栏里的示例标记不会被误还原。
func RecoverInlineToolCalls(content string) (calls []ToolCall, cleaned string, ok bool) {
	scan := maskFences(content)
	if calls, spans, ok := recoverAntml(content, scan); ok {
		return calls, removeSpans(content, spans), true
	}
	if calls, spans, ok := recoverHermes(content, scan); ok {
		return calls, removeSpans(content, spans), true
	}
	return nil, content, false
}

// recoverAntml 解析 antml/｜DSML｜ 风格的 invoke/parameter 块；spans 为待从原文剥除的字节区间。
func recoverAntml(content, scan string) (calls []ToolCall, spans [][2]int, ok bool) {
	if !strings.Contains(scan, `invoke name="`) {
		return nil, nil, false
	}
	locs := inlineInvokeRe.FindAllStringSubmatchIndex(scan, -1)
	if len(locs) == 0 {
		return nil, nil, false
	}
	for i, loc := range locs {
		// loc: [fullStart,fullEnd, name0,name1, body0,body1]。索引与原文对齐，从原文取真值。
		calls = append(calls, ToolCall{
			Index:     i,
			ID:        "call_inline_" + strconv.Itoa(i),
			Type:      "function",
			Name:      content[loc[2]:loc[3]],
			Arguments: parseInlineParams(content[loc[4]:loc[5]]),
		})
		spans = append(spans, [2]int{loc[0], loc[1]})
	}
	for _, w := range inlineWrapRe.FindAllStringIndex(scan, -1) {
		spans = append(spans, [2]int{w[0], w[1]})
	}
	return calls, spans, true
}

// recoverHermes 解析 Hermes/Nous 风格的 <tool_call>{json}</tool_call> 块。
func recoverHermes(content, scan string) (calls []ToolCall, spans [][2]int, ok bool) {
	if !strings.Contains(scan, "<tool_call>") {
		return nil, nil, false
	}
	locs := hermesBlockRe.FindAllStringSubmatchIndex(scan, -1)
	if len(locs) == 0 {
		return nil, nil, false
	}
	for _, loc := range locs {
		if tc, parsed := parseHermesCall(content[loc[2]:loc[3]], len(calls)); parsed {
			calls = append(calls, tc)
			spans = append(spans, [2]int{loc[0], loc[1]})
		}
	}
	if len(calls) == 0 {
		return nil, nil, false
	}
	return calls, spans, true
}

// maskFences 返回与 s 等长的副本，把 ```…``` 围栏区域整体替换为空格——索引逐字节对齐，
// 既让围栏内的标记扫不出来，又不影响在原文上按索引取真值/剥除区间。
func maskFences(s string) string {
	return fenceRe.ReplaceAllStringFunc(s, func(m string) string {
		return strings.Repeat(" ", len(m))
	})
}

// removeSpans 从 s 剥除给定字节区间（合并重叠、按起点排序），再 TrimSpace。
func removeSpans(s string, spans [][2]int) string {
	if len(spans) == 0 {
		return strings.TrimSpace(s)
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i][0] < spans[j][0] })
	var b strings.Builder
	prev := 0
	for _, sp := range spans {
		if sp[0] < prev { // 与已剥除区间重叠，跳过
			if sp[1] > prev {
				prev = sp[1]
			}
			continue
		}
		b.WriteString(s[prev:sp[0]])
		prev = sp[1]
	}
	b.WriteString(s[prev:])
	return strings.TrimSpace(b.String())
}

// parseHermesCall 把一段 Hermes 工具调用 JSON 解析成结构化 ToolCall。
// 兼容工具名键 name/tool/function、参数键 arguments/parameters/args/input；参数既可是对象
// 也可是再被字符串包裹的 JSON。
func parseHermesCall(jsonText string, index int) (ToolCall, bool) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(strings.TrimSpace(jsonText)), &m); err != nil {
		return ToolCall{}, false
	}
	name := hermesName(m)
	if name == "" {
		return ToolCall{}, false
	}
	return ToolCall{
		Index:     index,
		ID:        "call_inline_" + strconv.Itoa(index),
		Type:      "function",
		Name:      name,
		Arguments: hermesArguments(m),
	}, true
}

func hermesName(m map[string]json.RawMessage) string {
	for _, k := range []string{"name", "tool", "function_name"} {
		if raw, ok := m[k]; ok {
			var s string
			if json.Unmarshal(raw, &s) == nil && s != "" {
				return s
			}
		}
	}
	// OpenAI 嵌套形态：{"function":{"name":"...","arguments":{...}}}
	if raw, ok := m["function"]; ok {
		var fn map[string]json.RawMessage
		if json.Unmarshal(raw, &fn) == nil {
			var s string
			if r2, ok := fn["name"]; ok && json.Unmarshal(r2, &s) == nil && s != "" {
				return s
			}
		}
	}
	return ""
}

func hermesArguments(m map[string]json.RawMessage) string {
	// 嵌套 function 形态优先取其内部 arguments。
	if raw, ok := m["function"]; ok {
		var fn map[string]json.RawMessage
		if json.Unmarshal(raw, &fn) == nil {
			if a := rawArguments(fn); a != "" {
				return a
			}
		}
	}
	if a := rawArguments(m); a != "" {
		return a
	}
	return "{}"
}

func rawArguments(m map[string]json.RawMessage) string {
	for _, k := range []string{"arguments", "parameters", "args", "input"} {
		raw, ok := m[k]
		if !ok {
			continue
		}
		// arguments 可能是对象，也可能是再被字符串包裹的 JSON 串。
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
		return string(raw)
	}
	return ""
}

// parseInlineParams 把一个 invoke 块体内的 parameter 列表拼成 JSON arguments 字符串，
// 保留模型给出的参数顺序。
func parseInlineParams(body string) string {
	params := inlineParamRe.FindAllStringSubmatch(body, -1)
	var b strings.Builder
	b.WriteByte('{')
	for i, p := range params {
		if i > 0 {
			b.WriteByte(',')
		}
		key, _ := json.Marshal(p[1])
		b.Write(key)
		b.WriteByte(':')
		b.WriteString(coerceInlineParamValue(p[2], p[3]))
	}
	b.WriteByte('}')
	return b.String()
}

// coerceInlineParamValue 把单个参数原始值转成 JSON 片段。
//
// 类型判定：带显式 string="true" 提示 → 一律按 JSON 字符串编码（尊重模型的类型标注）；
// 无提示时若 trim 后是合法的独立 JSON（数字/布尔/null/对象/数组）→ 原样内嵌，否则按字符串。
func coerceInlineParamValue(attrs, raw string) string {
	v := strings.TrimSpace(raw)
	if strings.Contains(attrs, `string="true"`) {
		enc, _ := json.Marshal(v)
		return string(enc)
	}
	if v != "" && json.Valid([]byte(v)) && looksLikeJSONValue(v) {
		return v
	}
	enc, _ := json.Marshal(v)
	return string(enc)
}

// looksLikeJSONValue 防止把恰好通过 json.Valid 的裸字符串（如纯数字 ID）误判为非字符串。
// 仅当首字符落在 JSON 非字符串值的起始集合时才允许原样内嵌。
func looksLikeJSONValue(v string) bool {
	switch v[0] {
	case '{', '[', '-', 't', 'f', 'n', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return true
	default:
		return false
	}
}
