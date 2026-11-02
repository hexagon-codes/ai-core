package template

import "encoding/json"

// ProviderContinuation 保留一次完整模型响应的有序协议条目。
// Source 是适配器生成的来源指纹；调用方只传递该不透明值，不据此推断模型能力。
// 本类型只用于显式的内部续传或持久化，不应写入公开消息、日志或用户正文。
type ProviderContinuation struct {
	Protocol string            `json:"protocol"`
	Source   string            `json:"source"`
	Model    string            `json:"model"`
	Output   []json.RawMessage `json:"output"`
}

// String 避免普通格式化日志展开协议私有内容。
func (c ProviderContinuation) String() string {
	return "[private provider continuation]"
}

// GoString 同样隐藏调试格式中的协议私有内容。
func (c ProviderContinuation) GoString() string { return "[private provider continuation]" }

// Clone 深拷贝协议条目，避免运行快照与后续请求共享可变字节。
func (c *ProviderContinuation) Clone() *ProviderContinuation {
	if c == nil {
		return nil
	}
	cloned := *c
	if c.Output != nil {
		cloned.Output = make([]json.RawMessage, len(c.Output))
		for i, item := range c.Output {
			cloned.Output[i] = append(json.RawMessage(nil), item...)
		}
	}
	return &cloned
}
