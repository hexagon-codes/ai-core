package template

import (
	"fmt"
	"strings"
)

// ───────────────────────────────────────────────────────────────────────────
// 有序内容块（content block）—— agent 回合的正典表示
//
// 对齐 Anthropic Messages API 的 content block 模型：一个 assistant 回合是**有序的
// 块序列** [text, tool_use, text, tool_use, …]，tool_result 按执行序排列。块的顺序
// 本身携带「先说什么 → 再调什么 → 又说什么」的因果——这正是多步 agent（ReAct /
// orchestrate）保真展示、持久化、replay 所必需的。
//
// 与 ContentPart 的职责分离（关键设计）：
//   - ContentPart 是**输入**多模态（text / image_url，喂给模型，对齐 OpenAI vision）。
//   - Block 是**输出 / 回合**表示（text / thinking / tool_use / tool_result，记录 agent
//     做了什么）。二者互不污染——绝不把输出块塞进输入的 ContentPart，避免 provider
//     序列化误判。
//
// 旧模型的缺陷（已取证）：Message.Content（单串）+ ToolCalls（扁平数组）无任何字段
// 承载 tool 与 text 各片段的相对位置，多步交错结构性不可表达。Block 序列修复之。
// ───────────────────────────────────────────────────────────────────────────

// BlockType 是内容块类型（discriminated union 的判别字段）。
type BlockType string

const (
	// BlockText 文本块（模型产出的可见文字）。
	BlockText BlockType = "text"
	// BlockThinking 思考块（扩展推理过程，对齐 Anthropic thinking / OpenAI reasoning）。
	BlockThinking BlockType = "thinking"
	// BlockToolUse 工具调用块（模型请求调用某工具）。
	BlockToolUse BlockType = "tool_use"
	// BlockToolResult 工具结果块（工具执行回灌，ToolUseID 关联同回合的 tool_use）。
	BlockToolResult BlockType = "tool_result"
)

// Block 是一个有序内容块（按 Type 区分变体，对齐 Anthropic content block）。
// 所有变体字段共用一个结构 + omitempty，使 JSON wire 紧凑且自描述。
type Block struct {
	// Type 块类型（必填，判别字段）。
	Type BlockType `json:"type"`

	// ── text / thinking ──
	// Text 文本内容（text 块的正文，或 thinking 块的推理过程）。
	Text string `json:"text,omitempty"`
	// DurationMs 思考耗时（毫秒，仅 thinking 块）。
	DurationMs int64 `json:"duration_ms,omitempty"`

	// ── tool_use ──
	// ID 工具调用唯一 id（tool_result 经 ToolUseID 回指）。
	ID string `json:"id,omitempty"`
	// Name 工具名。
	Name string `json:"name,omitempty"`
	// Input 调用参数（JSON 串）。
	Input string `json:"input,omitempty"`

	// ── tool_result ──
	// ToolUseID 关联的 tool_use.ID。
	ToolUseID string `json:"tool_use_id,omitempty"`
	// Output 工具结果文本。
	Output string `json:"output,omitempty"`
	// IsError 工具是否失败。
	IsError bool `json:"is_error,omitempty"`
	// Status 执行状态（"success" / "error"，框架在执行点产出；与 runtime.ToolStatus 同构）。
	Status string `json:"status,omitempty"`
}

// ── 构造器（best-practice：构造合法块，避免调用方手填 Type 出错）──

// TextBlock 文本块。
func TextBlock(text string) Block { return Block{Type: BlockText, Text: text} }

// ThinkingBlock 思考块（durationMs<=0 时省略耗时）。
func ThinkingBlock(text string, durationMs int64) Block {
	b := Block{Type: BlockThinking, Text: text}
	if durationMs > 0 {
		b.DurationMs = durationMs
	}
	return b
}

// ToolUseBlock 工具调用块。
func ToolUseBlock(id, name, input string) Block {
	return Block{Type: BlockToolUse, ID: id, Name: name, Input: input}
}

// ToolResultBlock 工具结果块。status 取 "success"/"error"（可空）。
func ToolResultBlock(toolUseID, output string, isError bool, status string) Block {
	return Block{Type: BlockToolResult, ToolUseID: toolUseID, Output: output, IsError: isError, Status: status}
}

// ── Blocks：有序块序列 + 操作 ──

// Blocks 是有序内容块序列（一个 assistant 回合的正典表示）。
type Blocks []Block

// Text 拼接所有 text 块的正文（向后兼容：从有序块退化到扁平 content 字段）。
func (bs Blocks) Text() string {
	var sb strings.Builder
	for _, b := range bs {
		if b.Type == BlockText && b.Text != "" {
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// ToolUses 按序提取所有 tool_use 块。
func (bs Blocks) ToolUses() []Block {
	var out []Block
	for _, b := range bs {
		if b.Type == BlockToolUse {
			out = append(out, b)
		}
	}
	return out
}

// Validate 校验块序列的结构完整性：
//   - 每个块的 Type 合法；
//   - 每个 tool_result 的 ToolUseID 必须有**在其之前**出现的配对 tool_use
//     （tool_result 不能先于其 tool_use，也不能悬空）。
func (bs Blocks) Validate() error {
	seen := map[string]bool{}
	for i, b := range bs {
		switch b.Type {
		case BlockText, BlockThinking:
			// ok
		case BlockToolUse:
			if b.ID == "" {
				return fmt.Errorf("block[%d] tool_use 缺少 id", i)
			}
			seen[b.ID] = true
		case BlockToolResult:
			if b.ToolUseID == "" {
				return fmt.Errorf("block[%d] tool_result 缺少 tool_use_id", i)
			}
			if !seen[b.ToolUseID] {
				return fmt.Errorf("block[%d] tool_result 的 tool_use_id %q 无配对（或先于）tool_use", i, b.ToolUseID)
			}
		default:
			return fmt.Errorf("block[%d] 未知块类型 %q", i, b.Type)
		}
	}
	return nil
}

// ── BlockBuilder：增量构建有序块流（供 hexagon 在 agent 执行循环里按真实顺序拼装）──

// BlockBuilder 按 agent 真实执行顺序增量拼装块序列。连续的同类文本不自动合并——
// 顺序即语义（text→tool→text 的三块必须各自独立，才能表达交错）。
type BlockBuilder struct{ blocks Blocks }

// NewBlockBuilder 新建构建器。
func NewBlockBuilder() *BlockBuilder { return &BlockBuilder{} }

// Text 追加文本块（空串忽略）。
func (b *BlockBuilder) Text(text string) *BlockBuilder {
	if text != "" {
		b.blocks = append(b.blocks, TextBlock(text))
	}
	return b
}

// Thinking 追加思考块（空串忽略）。
func (b *BlockBuilder) Thinking(text string, durationMs int64) *BlockBuilder {
	if text != "" {
		b.blocks = append(b.blocks, ThinkingBlock(text, durationMs))
	}
	return b
}

// ToolUse 追加工具调用块。
func (b *BlockBuilder) ToolUse(id, name, input string) *BlockBuilder {
	b.blocks = append(b.blocks, ToolUseBlock(id, name, input))
	return b
}

// ToolResult 追加工具结果块。
func (b *BlockBuilder) ToolResult(toolUseID, output string, isError bool, status string) *BlockBuilder {
	b.blocks = append(b.blocks, ToolResultBlock(toolUseID, output, isError, status))
	return b
}

// Build 返回拼装好的有序块序列。
func (b *BlockBuilder) Build() Blocks { return b.blocks }
