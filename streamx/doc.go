// Package streamx 提供 AI API 流式响应的统一抽象层
//
// 本包将不同 AI 厂商（OpenAI, Claude, Gemini 等）的流式响应
// 统一为一致的接口，简化流式处理的复杂度。
//
// 基本用法：
//
//	stream := streamx.NewStream(resp.Body, streamx.OpenAIFormat)
//	for chunk := range stream.Chunks() {
//	    fmt.Print(chunk.Content)
//	}
//
// 收集完整响应：
//
//	result, err := stream.Collect()
//	fmt.Println(result.Content)
//	fmt.Printf("Tokens: %d\n", result.Usage.TotalTokens)
//
// OpenAI 用量报告：
//
// Chunk.Usage 保存上游报告的本次请求累计快照，nil 表示该帧没有用量信息。
// Result.Usage 保留最后一份已报告快照：重复帧不累加，无用量帧不清零，
// choices 为空的统计帧也会更新结果。不同请求的用量由调用方另行累加。
//
// 使用回调处理：
//
//	stream.OnChunk(func(chunk *Chunk) {
//	    fmt.Print(chunk.Content)
//	}).OnDone(func(result *Result) {
//	    fmt.Printf("\nTotal: %d tokens\n", result.Usage.TotalTokens)
//	}).OnError(func(err error) {
//	    log.Printf("Error: %v", err)
//	}).Start()
package streamx
