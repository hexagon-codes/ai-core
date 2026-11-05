package template

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"text/template"
)

var (
	// ErrEmptyName 表示模板名称为空，无法注册
	ErrEmptyName = errors.New("template: name is empty")
	// ErrNotFound 表示在注册表中未找到指定名称的模板
	ErrNotFound = errors.New("template: not found")
	// ErrRenderFailed 表示模板渲染过程中发生错误
	ErrRenderFailed = errors.New("template: render failed")
)

// Role 定义消息的角色类型
// 与主流 AI API 的消息角色保持一致
type Role string

const (
	// RoleSystem 系统角色，用于设置 AI 的行为和上下文
	RoleSystem Role = "system"
	// RoleUser 用户角色，表示用户的输入消息
	RoleUser Role = "user"
	// RoleAssistant 助手角色，表示 AI 的回复消息
	RoleAssistant Role = "assistant"
	// RoleTool 工具角色，用于返回工具调用的结果
	RoleTool Role = "tool"
)

// Message 表示聊天消息
// 与 OpenAI/Claude 等 API 的消息格式兼容
//
// 支持两种内容模式：
//   - 纯文本模式：使用 Content 字段
//   - 多模态模式：使用 MultiContent 字段（图片+文本混合）
//
// 当 MultiContent 不为空时，优先使用 MultiContent（OpenAI vision 等多模态 API 格式）
type Message struct {
	// Role 消息角色
	Role Role `json:"role"`
	// Content 消息内容（纯文本模式）
	Content string `json:"content"`
	// Name 可选的发送者名称
	Name string `json:"name,omitempty"`
	// MultiContent 多模态内容（图片+文本混合），用于 Vision 等多模态 API
	// 当此字段不为空时，Content 字段将被忽略
	MultiContent []ContentPart `json:"multi_content,omitempty"`
	// ToolCallID 工具调用 ID（当 Role=RoleTool 时必填）
	// 关联到 assistant 消息中 ToolCalls[].ID，用于 checkpoint/resume/replay
	ToolCallID string `json:"tool_call_id,omitempty"`
	// ToolResultError 是工具执行点提供的失败事实，不通过正文关键字猜测。
	ToolResultError bool `json:"tool_result_error,omitempty"`
	// ToolCalls 工具调用列表（当 Role=RoleAssistant 且 LLM 请求工具调用时填充）
	ToolCalls []ToolCallRef `json:"tool_calls,omitempty"`
	// ProviderContinuation 只由可信适配器产生，公开消息序列化不会透出协议私有状态。
	ProviderContinuation *ProviderContinuation `json:"-"`
}

// ToolCallRef 轻量工具调用引用，存储在 Message 中
// 完整的 ToolCall 定义在 streamx 包中
type ToolCallRef struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ContentPart 多模态消息的内容部分
// 对应 OpenAI 的 content array 中的一个元素
type ContentPart struct {
	// Type 内容类型："text" 或 "image_url"
	Type string `json:"type"`
	// Text 文本内容（当 Type="text" 时使用）
	Text string `json:"text,omitempty"`
	// ImageURL 图片 URL（当 Type="image_url" 时使用）
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL 图片 URL 定义
type ImageURL struct {
	// URL 图片地址，支持 http(s) URL 或 base64 data URI
	// base64 格式: "data:image/png;base64,iVBORw0KGgo..."
	URL string `json:"url"`
	// Detail 图片分析详细程度: "auto"、"low" 或 "high"
	// 默认 "auto"
	Detail string `json:"detail,omitempty"`
}

// NewTextPart 创建文本内容部分
func NewTextPart(text string) ContentPart {
	return ContentPart{Type: "text", Text: text}
}

// NewImageURLPart 创建图片 URL 内容部分
func NewImageURLPart(url string, detail string) ContentPart {
	if detail == "" {
		detail = "auto"
	}
	return ContentPart{
		Type:     "image_url",
		ImageURL: &ImageURL{URL: url, Detail: detail},
	}
}

// HasMultiContent 检查消息是否包含多模态内容
func (m Message) HasMultiContent() bool {
	return len(m.MultiContent) > 0
}

// messageTemplate 内部结构，存储单条消息的模板定义
type messageTemplate struct {
	Role     Role   // 消息角色
	Template string // 模板字符串，支持 Go template 语法
	Name     string // 可选的发送者名称
}

// Template 是 Prompt 模板的核心类型
// 支持定义多条消息模板，使用 Go template 语法进行变量替换
//
// 主要功能：
//   - 支持 system/user/assistant/tool 等多种角色
//   - 支持模板变量替换（{{.Variable}}）
//   - 内置常用函数（json、upper、lower、trim 等）
//   - 支持自定义模板函数
//   - 支持自定义分隔符（避免与前端框架冲突）
type Template struct {
	name     string            // 模板名称，用于注册表查找
	messages []messageTemplate // 消息模板列表
	funcs    template.FuncMap  // 自定义模板函数
	delims   [2]string         // 模板分隔符，默认 {{ 和 }}
}

// New 创建新的 Prompt 模板
// name 参数用于在注册表中标识模板
//
// 示例：
//
//	tmpl := template.New("greeting").
//	    System("你是一个友好的助手").
//	    User("你好，我是 {{.Name}}")
func New(name string) *Template {
	return &Template{
		name:     name,
		messages: make([]messageTemplate, 0),
		funcs:    defaultFuncs(),
		delims:   [2]string{"{{", "}}"},
	}
}

// defaultFuncs 返回内置的模板函数集合
// 这些函数可以在模板中直接使用
func defaultFuncs() template.FuncMap {
	return template.FuncMap{
		// json 将值序列化为 JSON 字符串
		// 用法：{{json .Data}}
		"json": func(v any) string {
			b, _ := json.Marshal(v)
			return string(b)
		},
		// join 连接字符串切片
		// 用法：{{join .Items ", "}}
		"join": strings.Join,
		// upper 转换为大写
		// 用法：{{upper .Name}}
		"upper": strings.ToUpper,
		// lower 转换为小写
		// 用法：{{lower .Name}}
		"lower": strings.ToLower,
		// trim 去除首尾空白
		// 用法：{{trim .Text}}
		"trim": strings.TrimSpace,
		// default 提供默认值
		// 用法：{{default "默认值" .OptionalField}}
		"default": func(def, val string) string {
			if val == "" {
				return def
			}
			return val
		},
		// truncate 截断字符串到指定长度
		// 用法：{{truncate 100 .LongText}}
		"truncate": func(length int, s string) string {
			runes := []rune(s)
			if len(runes) <= length {
				return s
			}
			return string(runes[:length]) + "..."
		},
	}
}

// Name 返回模板名称
func (t *Template) Name() string {
	return t.name
}

// Funcs 添加自定义模板函数
// 新函数会覆盖同名的内置函数
// 支持链式调用
//
// 示例：
//
//	tmpl.Funcs(map[string]any{
//	    "double": func(s string) string { return s + s },
//	})
func (t *Template) Funcs(funcs template.FuncMap) *Template {
	for k, v := range funcs {
		t.funcs[k] = v
	}
	return t
}

// Delims 设置模板分隔符
// 当默认的 {{ }} 与其他模板系统冲突时使用
// 支持链式调用
//
// 示例：
//
//	tmpl.Delims("[[", "]]")  // 使用 [[.Name]] 语法
func (t *Template) Delims(left, right string) *Template {
	t.delims = [2]string{left, right}
	return t
}

// System 添加系统角色消息模板
// 系统消息用于设置 AI 的行为、角色和上下文
// 支持链式调用
func (t *Template) System(content string) *Template {
	t.messages = append(t.messages, messageTemplate{
		Role:     RoleSystem,
		Template: content,
	})
	return t
}

// User 添加用户角色消息模板
// 支持链式调用
func (t *Template) User(content string) *Template {
	t.messages = append(t.messages, messageTemplate{
		Role:     RoleUser,
		Template: content,
	})
	return t
}

// UserWithName 添加带发送者名称的用户消息模板
// 用于多用户对话场景，区分不同用户
// 支持链式调用
func (t *Template) UserWithName(name, content string) *Template {
	t.messages = append(t.messages, messageTemplate{
		Role:     RoleUser,
		Template: content,
		Name:     name,
	})
	return t
}

// Assistant 添加助手角色消息模板
// 用于在对话历史中添加 AI 的回复
// 支持链式调用
func (t *Template) Assistant(content string) *Template {
	t.messages = append(t.messages, messageTemplate{
		Role:     RoleAssistant,
		Template: content,
	})
	return t
}

// Tool 添加工具角色消息模板
// 用于返回工具/函数调用的结果
// 支持链式调用
func (t *Template) Tool(content string) *Template {
	t.messages = append(t.messages, messageTemplate{
		Role:     RoleTool,
		Template: content,
	})
	return t
}

// Message 添加自定义角色消息模板
// 用于添加非标准角色或动态角色的消息
// 支持链式调用
func (t *Template) Message(role Role, content string) *Template {
	t.messages = append(t.messages, messageTemplate{
		Role:     role,
		Template: content,
	})
	return t
}

// Render 使用给定数据渲染模板
// 返回渲染后的消息列表，可直接用于 AI API 调用
//
// 参数 data 可以是 map[string]any 或结构体
// 模板中通过 {{.FieldName}} 访问字段
//
// 示例：
//
//	messages, err := tmpl.Render(map[string]any{
//	    "Name": "Alice",
//	    "Topic": "Go 编程",
//	})
func (t *Template) Render(data any) ([]Message, error) {
	messages := make([]Message, 0, len(t.messages))

	for _, mt := range t.messages {
		content, err := t.renderString(mt.Template, data)
		if err != nil {
			return nil, err
		}

		msg := Message{
			Role:    mt.Role,
			Content: content,
		}
		if mt.Name != "" {
			msg.Name = mt.Name
		}

		messages = append(messages, msg)
	}

	return messages, nil
}

// RenderToString 将模板渲染为单个格式化字符串
// 格式为 "role: content\n\nrole: content"
// 主要用于调试和日志输出
func (t *Template) RenderToString(data any) (string, error) {
	messages, err := t.Render(data)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	for i, msg := range messages {
		if i > 0 {
			buf.WriteString("\n\n")
		}
		buf.WriteString(string(msg.Role))
		buf.WriteString(": ")
		buf.WriteString(msg.Content)
	}

	return buf.String(), nil
}

// renderString 渲染单个模板字符串
// 使用配置的分隔符和函数集合
func (t *Template) renderString(tmplStr string, data any) (string, error) {
	tmpl, err := template.New("").
		Delims(t.delims[0], t.delims[1]).
		Funcs(t.funcs).
		Parse(tmplStr)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", err
	}

	return buf.String(), nil
}

// Clone 创建模板的深拷贝
// 克隆后的模板可以独立修改，不影响原模板
func (t *Template) Clone() *Template {
	clone := &Template{
		name:     t.name,
		messages: make([]messageTemplate, len(t.messages)),
		funcs:    make(template.FuncMap),
		delims:   t.delims,
	}

	copy(clone.messages, t.messages)

	for k, v := range t.funcs {
		clone.funcs[k] = v
	}

	return clone
}

// ============== 模板注册表 ==============

// Registry 是模板的注册表/仓库
// 支持按名称存储和检索模板
// 使用 sync.Map 实现，线程安全
type Registry struct {
	templates sync.Map
}

// globalRegistry 是全局模板注册表
// 提供应用级别的模板共享
var globalRegistry = &Registry{}

// Register 将模板注册到全局注册表
// 模板必须有非空名称
func Register(t *Template) error {
	if t.name == "" {
		return ErrEmptyName
	}
	globalRegistry.templates.Store(t.name, t)
	return nil
}

// Get 从全局注册表获取指定名称的模板
// 如果模板不存在，返回 ErrNotFound
func Get(name string) (*Template, error) {
	v, ok := globalRegistry.templates.Load(name)
	if !ok {
		return nil, ErrNotFound
	}
	return v.(*Template), nil
}

// MustGet 从全局注册表获取模板
// 如果模板不存在，触发 panic
// 适用于程序启动时的初始化阶段
func MustGet(name string) *Template {
	t, err := Get(name)
	if err != nil {
		panic(err)
	}
	return t
}

// NewRegistry 创建新的模板注册表实例
// 用于需要隔离模板命名空间的场景
func NewRegistry() *Registry {
	return &Registry{}
}

// Register 将模板注册到此注册表
func (r *Registry) Register(t *Template) error {
	if t.name == "" {
		return ErrEmptyName
	}
	r.templates.Store(t.name, t)
	return nil
}

// Get 从此注册表获取指定名称的模板
func (r *Registry) Get(name string) (*Template, error) {
	v, ok := r.templates.Load(name)
	if !ok {
		return nil, ErrNotFound
	}
	return v.(*Template), nil
}

// List 返回注册表中所有模板的名称列表
func (r *Registry) List() []string {
	var names []string
	r.templates.Range(func(key, _ any) bool {
		names = append(names, key.(string))
		return true
	})
	return names
}

// ============== 文件加载 ==============

// LoadFile 从文件加载模板
// 文件格式见 Parse 函数说明
func LoadFile(path string) (*Template, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	return Parse(string(data))
}

// Parse 解析模板字符串为 Template 对象
// 支持特殊格式定义多条消息模板
//
// 格式说明：
//   - @name: 模板名称
//   - @system: 系统消息
//   - @user: 用户消息
//   - @assistant: 助手消息
//   - @tool: 工具消息
//
// 消息内容支持多行，直到遇到下一个 @ 指令
//
// 示例：
//
//	@name: greeting
//	@system: 你是一个友好的助手。
//	请用中文回答。
//	@user: 你好，{{.Name}}！
func Parse(content string) (*Template, error) {
	lines := strings.Split(content, "\n")

	var name string
	t := New("")
	var currentRole Role
	var currentContent strings.Builder

	flushMessage := func() {
		if currentRole != "" && currentContent.Len() > 0 {
			t.Message(currentRole, strings.TrimSpace(currentContent.String()))
			currentContent.Reset()
		}
	}

	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "@name:") {
			name = strings.TrimSpace(strings.TrimPrefix(line, "@name:"))
			t.name = name
		} else if strings.HasPrefix(line, "@system:") {
			flushMessage()
			currentRole = RoleSystem
			currentContent.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "@system:")))
		} else if strings.HasPrefix(line, "@user:") {
			flushMessage()
			currentRole = RoleUser
			currentContent.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "@user:")))
		} else if strings.HasPrefix(line, "@assistant:") {
			flushMessage()
			currentRole = RoleAssistant
			currentContent.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "@assistant:")))
		} else if strings.HasPrefix(line, "@tool:") {
			flushMessage()
			currentRole = RoleTool
			currentContent.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "@tool:")))
		} else if line != "" && currentRole != "" {
			if currentContent.Len() > 0 {
				currentContent.WriteString("\n")
			}
			currentContent.WriteString(line)
		}
	}

	flushMessage()

	return t, nil
}

// ============== 便捷函数 ==============

// QuickRender 快速渲染单个模板字符串
// 适用于简单的一次性渲染场景
//
// 示例：
//
//	result, err := template.QuickRender("你好，{{.Name}}！", map[string]any{"Name": "Alice"})
func QuickRender(tmpl string, data any) (string, error) {
	t := New("quick").System(tmpl)
	messages, err := t.Render(data)
	if err != nil {
		return "", err
	}
	if len(messages) > 0 {
		return messages[0].Content, nil
	}
	return "", nil
}

// BuildMessages 快速构建消息列表
// 第一个参数是系统消息（可为空），后续参数是用户消息
//
// 示例：
//
//	messages := template.BuildMessages("你是助手", "问题1", "问题2")
func BuildMessages(system string, userMessages ...string) []Message {
	messages := make([]Message, 0, 1+len(userMessages))

	if system != "" {
		messages = append(messages, Message{
			Role:    RoleSystem,
			Content: system,
		})
	}

	for _, um := range userMessages {
		messages = append(messages, Message{
			Role:    RoleUser,
			Content: um,
		})
	}

	return messages
}

// ChatHistory 是对话历史的构建器
// 提供便捷的方法来构建多轮对话的消息列表
// 支持链式调用
type ChatHistory struct {
	messages []Message
}

// NewChatHistory 创建新的对话历史构建器
//
// 示例：
//
//	history := template.NewChatHistory().
//	    System("你是助手").
//	    User("你好").
//	    Assistant("你好！有什么可以帮助你的？").
//	    User("天气如何？")
func NewChatHistory() *ChatHistory {
	return &ChatHistory{
		messages: make([]Message, 0),
	}
}

// System 添加系统消息，支持链式调用
func (h *ChatHistory) System(content string) *ChatHistory {
	h.messages = append(h.messages, Message{Role: RoleSystem, Content: content})
	return h
}

// User 添加用户消息，支持链式调用
func (h *ChatHistory) User(content string) *ChatHistory {
	h.messages = append(h.messages, Message{Role: RoleUser, Content: content})
	return h
}

// Assistant 添加助手消息，支持链式调用
func (h *ChatHistory) Assistant(content string) *ChatHistory {
	h.messages = append(h.messages, Message{Role: RoleAssistant, Content: content})
	return h
}

// Add 添加自定义角色的消息，支持链式调用
func (h *ChatHistory) Add(role Role, content string) *ChatHistory {
	h.messages = append(h.messages, Message{Role: role, Content: content})
	return h
}

// Messages 返回消息列表的副本
// 返回副本以防止外部修改内部状态
func (h *ChatHistory) Messages() []Message {
	result := make([]Message, len(h.messages))
	copy(result, h.messages)
	return result
}

// Clone 创建对话历史的深拷贝
// 用于创建对话分支
func (h *ChatHistory) Clone() *ChatHistory {
	clone := &ChatHistory{
		messages: make([]Message, len(h.messages)),
	}
	copy(clone.messages, h.messages)
	return clone
}

// Len 返回消息数量
func (h *ChatHistory) Len() int {
	return len(h.messages)
}

// Clear 清空所有消息，支持链式调用
func (h *ChatHistory) Clear() *ChatHistory {
	h.messages = h.messages[:0]
	return h
}

// Last 返回最后 n 条消息
// 用于截取最近的对话上下文
// 如果 n 大于消息总数，返回所有消息
func (h *ChatHistory) Last(n int) []Message {
	if n <= 0 {
		return nil
	}
	if n >= len(h.messages) {
		result := make([]Message, len(h.messages))
		copy(result, h.messages)
		return result
	}
	result := make([]Message, n)
	copy(result, h.messages[len(h.messages)-n:])
	return result
}
