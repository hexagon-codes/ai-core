package llm

import (
	"context"

	"github.com/hexagon-codes/ai-core/schema"
	"github.com/hexagon-codes/ai-core/streamx"
	"github.com/hexagon-codes/ai-core/template"
)

// 重新导出 template 包的核心类型，便于使用
type (
	// Role 消息角色类型
	Role = template.Role

	// Message 聊天消息
	Message = template.Message

	// ToolCallRef 轻量工具调用引用
	ToolCallRef = template.ToolCallRef

	// ContentPart 多模态内容部分 (文本/图片)
	ContentPart = template.ContentPart

	// ImageURL 图片 URL
	ImageURL = template.ImageURL
)

// 重新导出角色常量
const (
	RoleSystem    = template.RoleSystem
	RoleUser      = template.RoleUser
	RoleAssistant = template.RoleAssistant
	RoleTool      = template.RoleTool
)

// 重新导出 streamx 的核心类型
type (
	// StreamChunk 流式响应块
	StreamChunk = streamx.Chunk

	// StreamResult 流式响应的完整结果
	StreamResult = streamx.Result

	// Usage Token 使用统计
	Usage = streamx.Usage

	// ToolCall 工具调用
	ToolCall = streamx.ToolCall

	// Stream LLM 流式响应
	Stream = streamx.Stream
)

// 重新导出 schema 类型
type Schema = schema.Schema

// Provider 定义 LLM 提供者的核心接口
// 所有 LLM 服务（OpenAI、Anthropic、DeepSeek 等）都应实现此接口
type Provider interface {
	// Name 返回提供者名称（如 "openai"、"anthropic"）
	Name() string

	// Complete 执行非流式补全请求
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)

	// Stream 执行流式补全请求
	Stream(ctx context.Context, req CompletionRequest) (*Stream, error)

	// Models 返回可用模型列表
	Models() []ModelInfo

	// CountTokens 计算消息的 Token 数量。
	// 该签名为兼容既有实现而保留；可取消计数由可选的 ContextTokenCounter 提供。
	CountTokens(messages []Message) (int, error)
}

// EmbeddingProvider 定义支持向量嵌入的 Provider
type EmbeddingProvider interface {
	Provider

	// Embed 生成文本的向量嵌入
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// EmbedWithModel 使用指定模型生成嵌入
	EmbedWithModel(ctx context.Context, model string, texts []string) ([][]float32, error)
}

// ToolDefinition 定义一个工具给 LLM 使用
type ToolDefinition struct {
	Type     string          `json:"type"`
	Function ToolFunctionDef `json:"function"`
}

// ToolFunctionDef 定义函数工具
type ToolFunctionDef struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Parameters  *Schema `json:"parameters"`
}

// NewToolDefinition 创建工具定义
func NewToolDefinition(name, description string, parameters *Schema) ToolDefinition {
	return ToolDefinition{
		Type: "function",
		Function: ToolFunctionDef{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// ReasoningPolicyScope is an explicit, typed opt-in for adapter inference from
// coarse semantic reasoning metadata. The zero value disables inference.
type ReasoningPolicyScope string

const (
	// ReasoningPolicyScopeStructuredVisionRecognition allows a Provider adapter
	// to translate the semantic thinking mode for a structured vision
	// recognition request. It is control-plane state and is never serialized
	// directly into an upstream payload.
	ReasoningPolicyScopeStructuredVisionRecognition ReasoningPolicyScope = "structured_vision_recognition"
)

// CompletionRequest 表示补全请求
type CompletionRequest struct {
	// Model 模型名称（如 "gpt-4o"、"deepseek-chat"）
	Model string `json:"model"`

	// Messages 对话消息列表
	Messages []Message `json:"messages"`

	// Tools 可用工具列表
	Tools []ToolDefinition `json:"tools,omitempty"`

	// ToolChoice 工具选择策略
	// 可以是 "auto"、"none"、"required" 或指定工具名
	ToolChoice any `json:"tool_choice,omitempty"`

	// MaxTokens 最大生成 Token 数
	MaxTokens int `json:"max_tokens,omitempty"`

	// Temperature 采样温度 (0-2)
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP 核采样参数 (0-1)
	TopP *float64 `json:"top_p,omitempty"`

	// Stop 停止词列表
	Stop []string `json:"stop,omitempty"`

	// User 用户标识（用于追踪和滥用检测）
	User string `json:"user,omitempty"`

	// Metadata 额外元数据
	Metadata map[string]any `json:"metadata,omitempty"`

	// ReasoningPolicyScope explicitly opts this request into adapter inference
	// from coarse reasoning metadata such as thinking=off. Direct standard
	// parameters such as reasoning_effort remain independent of this scope.
	ReasoningPolicyScope ReasoningPolicyScope `json:"-"`

	// ResponseFormat 响应格式约束
	//
	// 用于请求 LLM 以特定格式输出，如 JSON 模式。
	// 不同 Provider 的支持程度不同：
	//   - OpenAI: 支持 "json_object" 和 "json_schema"
	//   - DeepSeek: 支持 "json_object"（兼容 OpenAI）
	//   - Gemini: 通过 responseMimeType 支持 "application/json"
	//   - Anthropic: 不原生支持，可通过 tool_use 模拟
	//   - Qwen/Ark: 支持 "json_object"（兼容 OpenAI）
	//   - Ollama: 通过 format 参数支持 "json"
	//
	// 不支持的 Provider 会忽略此字段，上层应降级为 Prompt 工程
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat 响应格式定义
//
// 用于约束 LLM 的输出格式，主要支持 JSON 模式和结构化输出。
type ResponseFormat struct {
	// Type 格式类型
	//   - "text": 普通文本输出（默认）
	//   - "json_object": JSON 对象输出
	//   - "json_schema": 带 Schema 约束的 JSON 输出（仅 OpenAI GPT-4o+ 支持）
	Type string `json:"type"`

	// JSONSchema 结构化输出的 JSON Schema 定义
	// 仅当 Type 为 "json_schema" 时有效
	JSONSchema *ResponseFormatJSONSchema `json:"json_schema,omitempty"`
}

// ResponseFormatJSONSchema 结构化 JSON Schema 定义
type ResponseFormatJSONSchema struct {
	// Name Schema 名称
	Name string `json:"name"`

	// Description Schema 描述
	Description string `json:"description,omitempty"`

	// Schema JSON Schema 定义
	Schema *Schema `json:"schema"`

	// Strict 是否启用严格模式
	// 启用后 LLM 输出必须严格匹配 Schema
	Strict bool `json:"strict,omitempty"`
}

// CompletionResponse 表示补全响应
type CompletionResponse struct {
	// ID 响应的唯一标识符
	ID string `json:"id"`

	// Model 使用的模型名称
	Model string `json:"model"`

	// Content 生成的文本内容
	Content string `json:"content"`

	// ToolCalls 工具调用列表（如果有）
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// Usage Token 使用统计
	Usage Usage `json:"usage"`

	// FinishReason 结束原因
	FinishReason string `json:"finish_reason,omitempty"`

	// Created 创建时间戳
	Created int64 `json:"created"`
}

// HasToolCalls 检查响应是否包含工具调用
func (r *CompletionResponse) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// ModelInfo 包含模型信息
type ModelInfo struct {
	// ID 模型标识符
	ID string `json:"id"`

	// Name 模型显示名称
	Name string `json:"name"`

	// Description 模型描述
	Description string `json:"description,omitempty"`

	// MaxTokens 最大上下文长度
	MaxTokens int `json:"max_tokens"`

	// InputCost 输入成本（每百万 Token）
	InputCost float64 `json:"input_cost"`

	// OutputCost 输出成本（每百万 Token）
	OutputCost float64 `json:"output_cost"`

	// Features 支持的特性列表
	Features []string `json:"features,omitempty"`
}

// HasFeature 检查模型是否支持某个特性
func (m ModelInfo) HasFeature(feature string) bool {
	for _, f := range m.Features {
		if f == feature {
			return true
		}
	}
	return false
}

// 常用特性常量
const (
	FeatureVision          = "vision"           // 图像理解
	FeatureFunctions       = "functions"        // 函数调用
	FeatureJSON            = "json_mode"        // JSON 模式
	FeatureStreaming       = "streaming"        // 流式响应
	FeatureEmbedding       = "embedding"        // 向量嵌入
	FeatureImageGeneration = "image_generation" // 图片生成
	FeatureVideoGeneration = "video_generation" // 视频生成
)

// ============== 便捷函数 ==============

// NewMessages 创建消息列表的便捷函数
func NewMessages(system string, userMessages ...string) []Message {
	return template.BuildMessages(system, userMessages...)
}

// NewMessage 创建单条消息
func NewMessage(role Role, content string) Message {
	return Message{Role: role, Content: content}
}

// SystemMessage 创建系统消息
func SystemMessage(content string) Message {
	return Message{Role: RoleSystem, Content: content}
}

// UserMessage 创建用户消息
func UserMessage(content string) Message {
	return Message{Role: RoleUser, Content: content}
}

// AssistantMessage 创建助手消息
func AssistantMessage(content string) Message {
	return Message{Role: RoleAssistant, Content: content}
}

// ToolMessage 创建工具结果消息
func ToolMessage(content string) Message {
	return Message{Role: RoleTool, Content: content}
}

// ToolResultMessage 创建带 tool_call_id 的工具结果消息
// callID 关联到 assistant 消息中 ToolCalls[].ID
func ToolResultMessage(callID, content string) Message {
	return Message{Role: RoleTool, Content: content, ToolCallID: callID}
}

// AssistantToolCallMessage 创建包含工具调用请求的 assistant 消息
// NewTextPart 创建文本内容部分 (多模态消息用)
var NewTextPart = template.NewTextPart

// NewImageURLPart 创建图片 URL 内容部分 (多模态消息用)
var NewImageURLPart = template.NewImageURLPart

func AssistantToolCallMessage(content string, calls []ToolCallRef) Message {
	return Message{Role: RoleAssistant, Content: content, ToolCalls: calls}
}
