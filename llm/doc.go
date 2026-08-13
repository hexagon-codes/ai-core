// Package llm 提供 LLM (Large Language Model) 提供者的抽象接口
//
// 本包定义了与各种 LLM 服务（OpenAI、Anthropic、DeepSeek 等）交互的统一接口。
// 具体的 Provider 实现应该在各自的子包中（如 openai、deepseek 等）。
//
// # 主要接口
//
//   - Provider: LLM 提供者的核心接口，定义了补全、流式响应等方法
//   - ToolProvider: 支持工具调用的 Provider 扩展接口
//   - ContextTokenCounter: 支持取消和截止时间的可选 Token 计数扩展接口
//
// # 基本用法
//
//	provider := openai.New(os.Getenv("OPENAI_API_KEY"))
//
//	req := llm.CompletionRequest{
//	    Model: "gpt-4o",
//	    Messages: []llm.Message{
//	        {Role: llm.RoleUser, Content: "Hello!"},
//	    },
//	}
//
//	resp, err := provider.Complete(ctx, req)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(resp.Content)
//
// # 流式响应
//
//	stream, err := provider.Stream(ctx, req)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer stream.Close()
//
//	for chunk := range stream.Chunks() {
//	    fmt.Print(chunk.Content)
//	}
package llm
