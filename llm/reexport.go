package llm

// 顶层 re-export — 使上层仓库（hexagon/hexclaw）无需直接引用 ai-core 内部子包。
//
// 注：openai/deepseek 等 provider 构造函数无法在此 re-export，
// 因为它们反向依赖 llm 包（会造成循环 import）。
// provider 构造函数由 hexagon 顶层包 re-export。

import (
	"io"
	"reflect"

	"github.com/hexagon-codes/ai-core/schema"
	"github.com/hexagon-codes/ai-core/streamx"
)

// ─── schema re-exports ─────────────────────────────────

// SchemaOf 从 Go 类型自动生成 JSON Schema
func SchemaOf[T any]() *Schema {
	return schema.Of[T]()
}

// SchemaFromType 从 reflect.Type 生成 Schema
func SchemaFromType(t reflect.Type) *Schema {
	return schema.FromType(t)
}

// SchemaFromValue 从任意值生成 Schema
func SchemaFromValue(v any) *Schema {
	return schema.FromValue(v)
}

// ─── streamx re-exports ────────────────────────────────

// StreamFormat 流式响应数据格式
type StreamFormat = streamx.Format

// 流格式常量
const (
	StreamOpenAIFormat StreamFormat = streamx.OpenAIFormat
	StreamClaudeFormat StreamFormat = streamx.ClaudeFormat
	StreamGeminiFormat StreamFormat = streamx.GeminiFormat
	StreamCustomFormat StreamFormat = streamx.CustomFormat
)

// NewStream 创建��处理器
func NewStream(r io.Reader, format StreamFormat) *Stream {
	return streamx.NewStream(r, format)
}

// NewStreamWithParser 创建自定义解析器的流处理器
func NewStreamWithParser(r io.Reader, parser streamx.ChunkParser) *Stream {
	return streamx.NewStreamWithParser(r, parser)
}

// ChunkParser 流块解析器��口
type ChunkParser = streamx.ChunkParser

// ─── embedding utility ─────────────────────────────────

// EmbeddingDimension 返回给定 Embedding 模型的默认向量维度
//
// 常用维度:
//   - text-embedding-3-small: 1536
//   - text-embedding-3-large: 3072
//   - text-embedding-ada-002: 1536
//
// 未知模型默认返回 1536
func EmbeddingDimension(model string) int {
	switch model {
	case "text-embedding-3-large":
		return 3072
	default:
		return 1536
	}
}
