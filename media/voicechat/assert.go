package voicechat

// 编译期接口断言：确保具体 Provider 实现满足接口契约。
var _ Provider = (*openAIVoiceChat)(nil)
