# ai-core

[English](README_EN.md) | 中文

Go 语言的 AI 基础能力库，为 [Hexagon](https://github.com/hexagon-codes/hexagon) AI Agent 框架提供核心支持。

## 特性

- **统一的 LLM 接口** - 一套代码，多家 Provider（OpenAI、Anthropic、DeepSeek、Gemini、通义千问、豆包、文心一言、Ollama）
- **中间件机制** - 可组合的 Provider 装饰器：重试（含不可重试错误检测）、限流、超时、回调、缓存（含 singleflight 防击穿，支持语义缓存）
- **流式响应** - 统一的 SSE 流式处理，支持回调和 channel 两种模式；通用 `StreamReader` 四件套（Concat/Merge/Copy + Close 契约）与 gRPC 分帧解析
- **工具系统** - 类型安全的工具定义，从 Go 结构体自动生成 JSON Schema
- **记忆系统** *(Experimental)* - 多种记忆策略（缓冲、摘要、向量检索、多层组合、实体记忆），条目支持溯源血缘与多模态。此接口处于实验阶段，后续版本可能发生变更
- **媒体生成** - 独立媒体子域（图像 / 视频 / 语音 / 语音对话），统一 Submit→Poll 异步任务模型 + 同步包装
- **智能路由** - 多 Provider 路由器，支持轮询、加权、最低延迟、降级等策略；任务感知智能路由
- **用量追踪** - Token 消耗统计和成本估算（原子累加器，裁剪后数值一致），支持请求追踪器
- **结构化输出** - ResponseFormat 支持 JSON 模式和 JSON Schema 约束（含 JSON Schema 2020-12 方言）

## 安装

```bash
go get github.com/hexagon-codes/ai-core@v0.1.4
```

## 快速开始

### 基本对话

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/hexagon-codes/ai-core/llm"
    "github.com/hexagon-codes/ai-core/llm/openai"
)

func main() {
    provider := openai.New(os.Getenv("OPENAI_API_KEY"))

    resp, err := provider.Complete(context.Background(), llm.CompletionRequest{
        Model: "gpt-4o",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: "你好！"},
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Content)
}
```

### 流式响应

```go
stream, err := provider.Stream(ctx, llm.CompletionRequest{
    Model: "gpt-4o",
    Messages: llm.NewMessages("你是一个助手", "讲个笑话"),
})
if err != nil {
    panic(err)
}
defer stream.Close()

for chunk := range stream.Chunks() {
    fmt.Print(chunk.Content)
}
```

### 工具调用

```go
import "github.com/hexagon-codes/ai-core/tool"

type WeatherInput struct {
    City string `json:"city" desc:"城市名称" required:"true"`
}

weatherTool := tool.NewFunc("get_weather", "获取城市天气",
    func(ctx context.Context, input WeatherInput) (string, error) {
        return fmt.Sprintf("%s: 晴，25°C", input.City), nil
    },
)

// 转换为 LLM 工具定义
toolDef := llm.NewToolDefinition(
    weatherTool.Name(),
    weatherTool.Description(),
    weatherTool.Schema(),
)

resp, _ := provider.Complete(ctx, llm.CompletionRequest{
    Model:    "gpt-4o",
    Messages: []llm.Message{{Role: llm.RoleUser, Content: "北京天气怎么样？"}},
    Tools:    []llm.ToolDefinition{toolDef},
})

if resp.HasToolCalls() {
    for _, tc := range resp.ToolCalls {
        args, _ := tool.ParseArgs(tc.Arguments)
        result, _ := weatherTool.Execute(ctx, args)
        fmt.Println(result)
    }
}
```

### 中间件

```go
import (
    "github.com/hexagon-codes/ai-core/llm"
    "github.com/hexagon-codes/ai-core/llm/cache"
)

// 组合多个中间件：重试 → 限流 → 缓存
enhanced := llm.Chain(provider,
    llm.WithRetry(3, time.Second),       // 指数退避重试（自动跳过 401/403 等不可重试错误）
    llm.WithRateLimit(10),               // 10 QPS 令牌桶限流
    llm.WithTimeout(30 * time.Second),   // 请求超时（不影响 Stream）
    llm.WithCache(cache.NewMemoryCache(), nil), // LRU 内存缓存（singleflight 防击穿）
)

resp, _ := enhanced.Complete(ctx, req)
```

### 多 Provider 路由

```go
import (
    "github.com/hexagon-codes/ai-core/llm/router"
    "github.com/hexagon-codes/ai-core/llm/openai"
    "github.com/hexagon-codes/ai-core/llm/deepseek"
)

r := router.NewBuilder().
    Add("openai", openai.New(openaiKey)).
    Add("deepseek", deepseek.New(deepseekKey)).
    Strategy(router.StrategyLeastLatency).
    Fallback("deepseek").
    EnableHealthCheck().
    Build()

// 使用路由器（自动选择最优 Provider）
resp, _ := r.Complete(ctx, req)
```

### 记忆系统 *(Experimental)*

> **注意：** 记忆接口处于实验阶段，后续版本可能发生不兼容变更。

```go
import "github.com/hexagon-codes/ai-core/memory"

// 缓冲记忆 — 保留最近 N 条消息
buf := memory.NewBuffer(100)
buf.Save(ctx, memory.NewUserEntry("你好"))
buf.Save(ctx, memory.NewAssistantEntry("你好！有什么可以帮助你的？"))

// Get() / Delete() 在条目不存在时返回 memory.ErrNotFound
entry, err := buf.Get(ctx, "some-id")
if errors.Is(err, memory.ErrNotFound) {
    // 处理未找到
}

// 摘要记忆 — 超过阈值自动压缩为摘要（doSummarize 并发安全：Clear + 重新写入在锁内完成）
sum := memory.NewSummaryMemory(summarizer, memory.WithMaxEntries(20))

// 向量记忆 — 语义检索
vec := memory.NewVectorMemory(embedder)
results, _ := vec.SemanticSearch(ctx, "之前讨论的架构方案", 5)

// 多层记忆 — 工作记忆 → 短期记忆 → 长期记忆
// Transfer() 使用独立的 transferMu 锁，避免在调用 Embedder 时阻塞读操作
multi := memory.NewMultiLayerMemory(
    memory.WithSummarizer(summarizer),
    memory.WithEmbedder(embedder),
)
```

## 包结构

| 包 | 说明 |
|---|------|
| `llm` | LLM Provider 抽象接口、中间件（重试/限流/超时/回调/缓存） |
| `llm/openai` | OpenAI 实现（GPT-4o、GPT-4-Turbo、o1、o3-mini 等） |
| `llm/anthropic` | Anthropic Claude 实现 |
| `llm/deepseek` | DeepSeek 实现 |
| `llm/gemini` | Google Gemini 实现 |
| `llm/qwen` | 通义千问实现 |
| `llm/ark` | 豆包（字节跳动）实现 |
| `llm/ernie` | 文心一言（百度）实现 |
| `llm/ollama` | Ollama 本地模型实现 |
| `llm/router` | 多 Provider 智能路由、任务感知路由（SmartRouter） |
| `llm/cache` | LRU 内存缓存 + 语义缓存（支持 TTL、singleflight 防击穿、SQLite 持久化） |
| `media` | 媒体生成子域：`media/image` `media/video` `media/voice` `media/voicechat`（Submit→Poll 任务模型） |
| `gateway/llmcall` | LLM 调用网关：带进度上报与瞬时错误重试的统一调用入口 |
| `memory` | Agent 记忆系统（缓冲/摘要/向量/多层/实体），支持溯源血缘与多模态条目 *Experimental* |
| `tool` | 工具定义和注册 |
| `schema` | JSON Schema 生成（从 Go 结构体反射，支持 JSON Schema 2020-12 方言） |
| `streamx` | 流式响应统一抽象（OpenAI/Claude/Gemini 格式）+ 通用 `StreamReader` / gRPC 分帧 |
| `template` | Prompt 模板引擎（支持多模态） |
| `tokenizer` | Token 计数估算 |
| `meter` | 用量统计和成本追踪（原子累加成本计数器） |
| `store/vector` | 向量存储抽象（内存/Qdrant） |

## 支持的 LLM Provider

| Provider | 模型示例 | 特性 |
|----------|---------|------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3-mini | 流式、函数调用、视觉 |
| Anthropic | claude-opus-4, claude-sonnet-4, claude-3.5-sonnet, claude-3.5-haiku | 流式、函数调用、视觉 |
| DeepSeek | deepseek-chat, deepseek-reasoner | 流式、函数调用 |
| Gemini | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash | 流式、函数调用、视觉、Embedding |
| 通义千问 | qwen-turbo, qwen-plus, qwen-max, qwen-vl-max | 流式、函数调用、视觉 |
| 豆包 | doubao-pro-*, doubao-lite-*, doubao-vision-pro-* | 流式、函数调用、视觉 |
| 文心一言 | ernie-4.5-8k, ernie-4.0-8k, ernie-3.5-8k, ernie-x1 | 流式 |
| Ollama | llama3.2, llama3.1, qwen2.5, mistral, codellama, llava | 流式、函数调用、视觉 |

## 路由策略

| 策略 | 说明 |
|------|------|
| `StrategyRoundRobin` | 轮询，均匀分发请求 |
| `StrategyRandom` | 随机选择 |
| `StrategyLeastLatency` | 选择延迟最低的 Provider |
| `StrategyLeastCost` | 选择成本最低的 Provider |
| `StrategyWeighted` | 按权重分发 |
| `StrategyFallback` | 按顺序尝试，失败后降级 |
| `StrategyModelMatch` | 根据请求的模型自动匹配 Provider |

## 设计原则

- **依赖精简** — 仅依赖生态内的 `toolkit`（通用工具底座）与少量必要库（如语义缓存的 SQLite 驱动），复用 toolkit 不重造轮子
- **接口驱动** — Provider、Memory、Tool、VectorStore 等核心类型均为接口，便于测试和扩展
- **并发安全** — 所有公共类型均通过 `sync.RWMutex` 或 `atomic` 保证线程安全
- **函数式选项** — 统一使用 `With*()` 选项模式配置组件
- **外部调用不持锁** — 调用 LLM/Embedder 等外部服务前释放锁，避免阻塞

## 许可证

[Apache License 2.0](LICENSE)
