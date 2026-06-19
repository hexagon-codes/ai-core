package image

// 编译期接口断言：确保具体 Provider 实现满足接口契约。
// 接口若发生漂移，将在此处直接编译失败，而非在远处的调用点暴露。
var _ Provider = (*openAICompatProvider)(nil)
