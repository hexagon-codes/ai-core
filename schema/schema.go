package schema

import (
	"encoding/json"
	"log"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// Draft2020_12 是 JSON Schema 2020-12 方言（dialect）的标识 URI。
//
// 用于在独立 Schema 文档或结构化输出场景声明所用方言。
// 注意：多数 LLM 的工具参数（Anthropic input_schema / OpenAI parameters）
// 不接受内嵌 $schema 字段，因此本框架默认不输出 $schema —— 仅在显式调用
// WithDialect 时才发出，避免污染工具调用请求。
const Draft2020_12 = "https://json-schema.org/draft/2020-12/schema"

// Schema 表示 JSON Schema 定义（兼容 JSON Schema 2020-12 方言）
// 用于描述 LLM 工具参数的结构和约束
type Schema struct {
	// Dialect 为 $schema 关键字，声明本 Schema 所用的 JSON Schema 方言。
	// 默认空（不输出），可设为 Draft2020_12。多用于独立文档/结构化输出，
	// 工具参数一般不应设置（见 Draft2020_12 说明）。
	Dialect     string             `json:"$schema,omitempty"`
	Type        string             `json:"type,omitempty"`
	Title       string             `json:"title,omitempty"`
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	AnyOf       []*Schema          `json:"anyOf,omitempty"`
	OneOf       []*Schema          `json:"oneOf,omitempty"`
	AllOf       []*Schema          `json:"allOf,omitempty"`
	Not         *Schema            `json:"not,omitempty"`
	// AdditionalProperties 控制对象是否允许声明之外的属性。
	// 可为 *Schema（约束额外属性的结构）或 bool（false=封闭对象）。
	// JSON Schema 2020-12 与 OpenAI strict function calling 用 false 表示
	// "不允许未声明属性"。nil 时省略（默认行为不变）。
	AdditionalProperties any      `json:"additionalProperties,omitempty"`
	Enum                 []any    `json:"enum,omitempty"`
	Default              any      `json:"default,omitempty"`
	Minimum              *float64 `json:"minimum,omitempty"`
	Maximum              *float64 `json:"maximum,omitempty"`
	MinLength            *int     `json:"minLength,omitempty"`
	MaxLength            *int     `json:"maxLength,omitempty"`
	Pattern              string   `json:"pattern,omitempty"`
	Format               string   `json:"format,omitempty"`
}

// String 返回 Schema 的 JSON 字符串表示
func (s *Schema) String() string {
	b, _ := json.Marshal(s)
	return string(b)
}

// MarshalJSON 实现 json.Marshaler 接口
func (s *Schema) MarshalJSON() ([]byte, error) {
	type Alias Schema
	return json.Marshal((*Alias)(s))
}

// WithDialect 设置 $schema 方言并返回自身（链式）。
// 典型用法 s.WithDialect(schema.Draft2020_12)。
func (s *Schema) WithDialect(dialect string) *Schema {
	s.Dialect = dialect
	return s
}

// Strict 将对象 Schema 标记为"封闭对象"并返回自身（链式）：
//   - additionalProperties = false（不允许未声明属性）
//   - 所有已声明属性进入 required
//
// 符合 JSON Schema 2020-12 封闭对象约定与 OpenAI strict function calling 要求。
// 仅作用于当前对象层（非递归）；非 object 类型时不做任何改动。
// required 列表按属性名排序以保证输出确定性。
func (s *Schema) Strict() *Schema {
	if s.Type != "object" {
		return s
	}
	s.AdditionalProperties = false
	s.Required = make([]string, 0, len(s.Properties))
	for name := range s.Properties {
		s.Required = append(s.Required, name)
	}
	sort.Strings(s.Required)
	return s
}

// Of 从 Go 类型生成 Schema
// 支持的 struct tag：
//   - json: 字段名（同 encoding/json）
//   - desc: 字段描述
//   - required: 是否必填 ("true" 或 "false")
//   - enum: 枚举值，逗号分隔
//   - default: 默认值
//   - min: 最小值（数字）或最小长度（字符串）
//   - max: 最大值（数字）或最大长度（字符串）
//   - pattern: 正则表达式（字符串）
//   - format: 格式（如 "email"、"uri"、"date-time"）
//
// 示例：
//
//	type Input struct {
//	    Name  string  `json:"name" desc:"用户名" required:"true" min:"1" max:"100"`
//	    Age   int     `json:"age" desc:"年龄" min:"0" max:"150"`
//	    Email string  `json:"email" format:"email"`
//	}
//	schema := schema.Of[Input]()
func Of[T any]() *Schema {
	var zero T
	return fromType(reflect.TypeOf(zero))
}

// FromType 从 reflect.Type 生成 Schema
func FromType(t reflect.Type) *Schema {
	return fromType(t)
}

// FromValue 从任意值生成 Schema
func FromValue(v any) *Schema {
	if v == nil {
		return &Schema{Type: "null"}
	}
	return fromType(reflect.TypeOf(v))
}

func fromType(t reflect.Type) *Schema {
	if t == nil {
		return &Schema{Type: "null"}
	}

	// 处理指针
	if t.Kind() == reflect.Pointer {
		return fromType(t.Elem())
	}

	switch t.Kind() {
	case reflect.String:
		return &Schema{Type: "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &Schema{Type: "integer"}
	case reflect.Float32, reflect.Float64:
		return &Schema{Type: "number"}
	case reflect.Bool:
		return &Schema{Type: "boolean"}
	case reflect.Slice, reflect.Array:
		return &Schema{
			Type:  "array",
			Items: fromType(t.Elem()),
		}
	case reflect.Map:
		return &Schema{Type: "object"}
	case reflect.Struct:
		return fromStruct(t)
	case reflect.Interface:
		return &Schema{Type: "object"}
	default:
		return &Schema{Type: "object"}
	}
}

func fromStruct(t reflect.Type) *Schema {
	schema := &Schema{
		Type:       "object",
		Properties: make(map[string]*Schema),
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if !field.IsExported() {
			continue
		}

		// 获取 JSON tag 或使用字段名
		name := field.Name
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}
		if jsonTag != "" {
			// 解析 json tag（处理 omitempty 等）
			if idx := strings.Index(jsonTag, ","); idx != -1 {
				jsonTag = jsonTag[:idx]
			}
			if jsonTag != "" {
				name = jsonTag
			}
		}

		propSchema := fromType(field.Type)

		// 解析 desc tag
		if desc := field.Tag.Get("desc"); desc != "" {
			propSchema.Description = desc
		}

		// 解析 required tag
		if field.Tag.Get("required") == "true" {
			schema.Required = append(schema.Required, name)
		}

		// 解析 enum tag
		if enumStr := field.Tag.Get("enum"); enumStr != "" {
			parts := strings.Split(enumStr, ",")
			enums := make([]any, len(parts))
			for j, p := range parts {
				enums[j] = strings.TrimSpace(p)
			}
			propSchema.Enum = enums
		}

		// 解析 default tag（根据字段类型转换为正确的值类型）
		if defStr := field.Tag.Get("default"); defStr != "" {
			propSchema.Default = parseDefault(defStr, propSchema.Type)
		}

		// 解析 format tag
		if format := field.Tag.Get("format"); format != "" {
			propSchema.Format = format
		}

		// 解析 pattern tag
		if pattern := field.Tag.Get("pattern"); pattern != "" {
			propSchema.Pattern = pattern
		}

		// 解析 min/max tag
		if minStr := field.Tag.Get("min"); minStr != "" {
			if propSchema.Type == "string" {
				// 字符串最小长度
				minLen := parseInt(minStr)
				propSchema.MinLength = &minLen
			} else {
				// 数字最小值
				minVal := parseFloat(minStr)
				propSchema.Minimum = &minVal
			}
		}
		if maxStr := field.Tag.Get("max"); maxStr != "" {
			if propSchema.Type == "string" {
				// 字符串最大长度
				maxLen := parseInt(maxStr)
				propSchema.MaxLength = &maxLen
			} else {
				// 数字最大值
				maxVal := parseFloat(maxStr)
				propSchema.Maximum = &maxVal
			}
		}

		schema.Properties[name] = propSchema
	}

	return schema
}

func parseInt(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		// 尝试 JSON 解析作为回退（支持带引号的数字）
		var jv int
		if json.Unmarshal([]byte(s), &jv) == nil {
			return jv
		}
		log.Printf("schema: warning: failed to parse integer from tag value %q, using 0", s)
		return 0
	}
	return v
}

func parseFloat(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		// 尝试 JSON 解析作为回退（支持带引号的数字）
		var jv float64
		if json.Unmarshal([]byte(s), &jv) == nil {
			return jv
		}
		log.Printf("schema: warning: failed to parse float from tag value %q, using 0", s)
		return 0
	}
	return v
}

// parseDefault 根据 schema 类型将默认值字符串转换为正确的 Go 类型
func parseDefault(s string, schemaType string) any {
	switch schemaType {
	case "integer":
		if v, err := strconv.Atoi(s); err == nil {
			return v
		}
	case "number":
		if v, err := strconv.ParseFloat(s, 64); err == nil {
			return v
		}
	case "boolean":
		if v, err := strconv.ParseBool(s); err == nil {
			return v
		}
	}
	return s
}

// ============== 构建器模式 ==============

// Builder 提供链式构建 Schema 的能力
type Builder struct {
	schema *Schema
}

// NewBuilder 创建新的 Schema 构建器
func NewBuilder() *Builder {
	return &Builder{
		schema: &Schema{
			Properties: make(map[string]*Schema),
		},
	}
}

// Type 设置 Schema 类型
func (b *Builder) Type(t string) *Builder {
	b.schema.Type = t
	return b
}

// Title 设置标题
func (b *Builder) Title(title string) *Builder {
	b.schema.Title = title
	return b
}

// Description 设置描述
func (b *Builder) Description(desc string) *Builder {
	b.schema.Description = desc
	return b
}

// Property 添加一个属性
func (b *Builder) Property(name string, prop *Schema, required bool) *Builder {
	b.schema.Properties[name] = prop
	if required {
		b.schema.Required = append(b.schema.Required, name)
	}
	return b
}

// Items 设置数组元素的 Schema
func (b *Builder) Items(items *Schema) *Builder {
	b.schema.Items = items
	return b
}

// Enum 设置枚举值
func (b *Builder) Enum(values ...any) *Builder {
	b.schema.Enum = values
	return b
}

// Default 设置默认值
func (b *Builder) Default(v any) *Builder {
	b.schema.Default = v
	return b
}

// Min 设置最小值
func (b *Builder) Min(v float64) *Builder {
	b.schema.Minimum = &v
	return b
}

// Max 设置最大值
func (b *Builder) Max(v float64) *Builder {
	b.schema.Maximum = &v
	return b
}

// MinLength 设置最小长度
func (b *Builder) MinLength(v int) *Builder {
	b.schema.MinLength = &v
	return b
}

// MaxLength 设置最大长度
func (b *Builder) MaxLength(v int) *Builder {
	b.schema.MaxLength = &v
	return b
}

// Pattern 设置正则模式
func (b *Builder) Pattern(p string) *Builder {
	b.schema.Pattern = p
	return b
}

// Format 设置格式
func (b *Builder) Format(f string) *Builder {
	b.schema.Format = f
	return b
}

// Build 返回构建的 Schema
func (b *Builder) Build() *Schema {
	return b.schema
}

// ============== 便捷构造函数 ==============

// String 创建字符串类型的 Schema
func String(desc string) *Schema {
	return &Schema{Type: "string", Description: desc}
}

// Integer 创建整数类型的 Schema
func Integer(desc string) *Schema {
	return &Schema{Type: "integer", Description: desc}
}

// Number 创建数字类型的 Schema
func Number(desc string) *Schema {
	return &Schema{Type: "number", Description: desc}
}

// Boolean 创建布尔类型的 Schema
func Boolean(desc string) *Schema {
	return &Schema{Type: "boolean", Description: desc}
}

// Array 创建数组类型的 Schema
func Array(desc string, items *Schema) *Schema {
	return &Schema{Type: "array", Description: desc, Items: items}
}

// Object 创建对象类型的 Schema
func Object(desc string) *Schema {
	return &Schema{Type: "object", Description: desc, Properties: make(map[string]*Schema)}
}

// StringEnum 创建字符串枚举类型的 Schema
func StringEnum(desc string, values ...string) *Schema {
	enums := make([]any, len(values))
	for i, v := range values {
		enums[i] = v
	}
	return &Schema{Type: "string", Description: desc, Enum: enums}
}
