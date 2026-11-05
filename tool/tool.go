package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/hexagon-codes/ai-core/schema"
)

// Tool 定义工具的核心接口
// 工具是 Agent 与外部世界交互的能力
type Tool interface {
	// Name 返回工具名称，用于 LLM 调用
	Name() string

	// Description 返回工具描述，帮助 LLM 理解何时使用此工具
	Description() string

	// Schema 返回工具参数的 JSON Schema
	Schema() *schema.Schema

	// Execute 执行工具并返回结果
	Execute(ctx context.Context, args map[string]any) (Result, error)

	// Validate 验证参数是否合法
	Validate(args map[string]any) error
}

// Result 表示工具执行结果
type Result struct {
	// Success 是否执行成功
	Success bool `json:"success"`

	// Output 输出数据（可以是任意类型）
	Output any `json:"output,omitempty"`

	// Error 错误信息（如果失败）
	Error string `json:"error,omitempty"`
}

// String 返回结果的字符串表示
func (r Result) String() string {
	if !r.Success {
		return fmt.Sprintf("Error: %s", r.Error)
	}
	b, _ := json.Marshal(r.Output)
	return string(b)
}

// NewResult 创建成功结果
func NewResult(output any) Result {
	return Result{Success: true, Output: output}
}

// NewErrorResult 创建错误结果
func NewErrorResult(err error) Result {
	return Result{Success: false, Error: err.Error()}
}

// ============== 函数式工具 ==============

// FuncTool 是基于函数创建的工具
type FuncTool[I, O any] struct {
	name        string
	description string
	fn          func(context.Context, I) (O, error)
	schema      *schema.Schema

	validatorOnce sync.Once
	validator     *schema.Validator
	validatorErr  error
}

// NewFunc 从函数创建工具
// I 是输入类型，O 是输出类型
// 输入类型应该是一个结构体，其字段会被转换为 JSON Schema
//
// 示例：
//
//	type Input struct {
//	    Query string `json:"query" desc:"搜索关键词" required:"true"`
//	}
//
//	searchTool := tool.NewFunc("search", "搜索网页",
//	    func(ctx context.Context, input Input) ([]string, error) {
//	        // 执行搜索...
//	        return results, nil
//	    },
//	)
func NewFunc[I, O any](name, description string, fn func(context.Context, I) (O, error)) *FuncTool[I, O] {
	return &FuncTool[I, O]{
		name:        name,
		description: description,
		fn:          fn,
		schema:      schema.Of[I](),
	}
}

// Name 返回工具名称
func (t *FuncTool[I, O]) Name() string {
	return t.name
}

// Description 返回工具描述
func (t *FuncTool[I, O]) Description() string {
	return t.description
}

// Schema 返回参数 Schema
func (t *FuncTool[I, O]) Schema() *schema.Schema {
	return t.schema
}

// Execute 执行工具
func (t *FuncTool[I, O]) Execute(ctx context.Context, args map[string]any) (Result, error) {
	// 先验证参数
	if err := t.Validate(args); err != nil {
		return NewErrorResult(err), nil
	}

	// 将 map 转换为输入类型
	var input I
	if err := mapToStruct(args, &input); err != nil {
		return NewErrorResult(err), nil
	}

	// 执行函数
	output, err := t.fn(ctx, input)
	if err != nil {
		return NewErrorResult(err), nil
	}

	return NewResult(output), nil
}

// Validate 验证参数
func (t *FuncTool[I, O]) Validate(args map[string]any) error {
	if t.schema == nil {
		return nil
	}
	t.validatorOnce.Do(func() {
		t.validator, t.validatorErr = schema.NewValidator(t.schema)
	})
	if t.validatorErr != nil {
		return fmt.Errorf("invalid tool schema: %w", t.validatorErr)
	}
	if err := t.validator.Validate(args); err != nil {
		return err
	}
	return nil
}

// mapToStruct 将 map 转换为结构体
func mapToStruct(m map[string]any, v any) error {
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// ============== 简单工具 ==============

// SimpleTool 是一个简单的工具实现
type SimpleTool struct {
	name        string
	description string
	schema      *schema.Schema
	executeFn   func(context.Context, map[string]any) (any, error)
	validateFn  func(map[string]any) error

	validatorOnce sync.Once
	validator     *schema.Validator
	validatorErr  error
}

// SimpleToolOption 是 SimpleTool 的配置选项
type SimpleToolOption func(*SimpleTool)

// WithSchema 设置工具的参数 Schema
func WithSchema(s *schema.Schema) SimpleToolOption {
	return func(t *SimpleTool) {
		t.schema = s
	}
}

// WithValidator 设置参数验证函数
func WithValidator(fn func(map[string]any) error) SimpleToolOption {
	return func(t *SimpleTool) {
		t.validateFn = fn
	}
}

// New 创建一个简单工具
func New(name, description string, execute func(context.Context, map[string]any) (any, error), opts ...SimpleToolOption) *SimpleTool {
	t := &SimpleTool{
		name:        name,
		description: description,
		executeFn:   execute,
		schema:      &schema.Schema{Type: "object", Properties: make(map[string]*schema.Schema)},
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Name 返回工具名称
func (t *SimpleTool) Name() string {
	return t.name
}

// Description 返回工具描述
func (t *SimpleTool) Description() string {
	return t.description
}

// Schema 返回参数 Schema
func (t *SimpleTool) Schema() *schema.Schema {
	return t.schema
}

// Execute 执行工具
func (t *SimpleTool) Execute(ctx context.Context, args map[string]any) (Result, error) {
	// 先验证参数
	if err := t.Validate(args); err != nil {
		return NewErrorResult(err), nil
	}

	output, err := t.executeFn(ctx, args)
	if err != nil {
		return NewErrorResult(err), nil
	}
	return NewResult(output), nil
}

// Validate 验证参数
func (t *SimpleTool) Validate(args map[string]any) error {
	if t.validateFn != nil {
		if err := t.validateFn(args); err != nil {
			return err
		}
	}
	if t.schema == nil {
		return nil
	}
	t.validatorOnce.Do(func() {
		t.validator, t.validatorErr = schema.NewValidator(t.schema)
	})
	if t.validatorErr != nil {
		return fmt.Errorf("invalid tool schema: %w", t.validatorErr)
	}
	return t.validator.Validate(args)
}

// ============== 工具注册中心 ==============

// Registry 是工具注册中心
type Registry struct {
	tools sync.Map
}

// NewRegistry 创建新的工具注册中心
func NewRegistry() *Registry {
	return &Registry{}
}

// Register 注册工具
func (r *Registry) Register(t Tool) error {
	if t.Name() == "" {
		return fmt.Errorf("tool name cannot be empty")
	}
	r.tools.Store(t.Name(), t)
	return nil
}

// RegisterAll 批量注册工具
func (r *Registry) RegisterAll(tools ...Tool) error {
	for _, t := range tools {
		if err := r.Register(t); err != nil {
			return err
		}
	}
	return nil
}

// Get 获取工具
func (r *Registry) Get(name string) (Tool, bool) {
	v, ok := r.tools.Load(name)
	if !ok {
		return nil, false
	}
	tool, ok := v.(Tool)
	if !ok {
		return nil, false
	}
	return tool, true
}

// MustGet 获取工具，如果不存在则 panic
func (r *Registry) MustGet(name string) Tool {
	t, ok := r.Get(name)
	if !ok {
		panic(fmt.Sprintf("tool not found: %s", name))
	}
	return t
}

// List 列出所有工具名称
func (r *Registry) List() []string {
	var names []string
	r.tools.Range(func(key, _ any) bool {
		if name, ok := key.(string); ok {
			names = append(names, name)
		}
		return true
	})
	return names
}

// All 返回所有工具
func (r *Registry) All() []Tool {
	var tools []Tool
	r.tools.Range(func(_, value any) bool {
		if tool, ok := value.(Tool); ok {
			tools = append(tools, tool)
		}
		return true
	})
	return tools
}

// Remove 移除工具
func (r *Registry) Remove(name string) {
	r.tools.Delete(name)
}

// Clear 清空所有工具
func (r *Registry) Clear() {
	r.tools.Range(func(key, value any) bool {
		r.tools.Delete(key)
		return true
	})
}

// ============== 全局注册中心 ==============

var globalRegistry = NewRegistry()

// Register 注册工具到全局注册中心
func Register(t Tool) error {
	return globalRegistry.Register(t)
}

// Get 从全局注册中心获取工具
func Get(name string) (Tool, bool) {
	return globalRegistry.Get(name)
}

// MustGet 从全局注册中心获取工具，如果不存在则 panic
func MustGet(name string) Tool {
	return globalRegistry.MustGet(name)
}

// List 列出全局注册中心的所有工具名称
func List() []string {
	return globalRegistry.List()
}

// All 返回全局注册中心的所有工具
func All() []Tool {
	return globalRegistry.All()
}

// ============== 工具类型转换 ==============

// ToLLMToolDefinition 将 Tool 转换为 LLM 工具定义格式
type LLMToolDefinition struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  *schema.Schema `json:"parameters"`
	} `json:"function"`
}

// ToLLMFormat 将 Tool 转换为 LLM API 所需的格式
func ToLLMFormat(t Tool) LLMToolDefinition {
	def := LLMToolDefinition{Type: "function"}
	def.Function.Name = t.Name()
	def.Function.Description = t.Description()
	def.Function.Parameters = t.Schema()
	return def
}

// ToLLMFormatBatch 批量转换工具
func ToLLMFormatBatch(tools []Tool) []LLMToolDefinition {
	defs := make([]LLMToolDefinition, len(tools))
	for i, t := range tools {
		defs[i] = ToLLMFormat(t)
	}
	return defs
}

// ============== 辅助函数 ==============

// ParseArgs 从 JSON 字符串解析参数
func ParseArgs(jsonStr string) (map[string]any, error) {
	var args map[string]any
	if jsonStr == "" {
		return make(map[string]any), nil
	}
	err := json.Unmarshal([]byte(jsonStr), &args)
	return args, err
}

// StructToArgs 将结构体转换为参数 map
func StructToArgs(v any) (map[string]any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var args map[string]any
	err = json.Unmarshal(b, &args)
	return args, err
}

// ArgsToStruct 将参数 map 转换为结构体
func ArgsToStruct[T any](args map[string]any) (T, error) {
	var result T
	err := mapToStruct(args, &result)
	return result, err
}
