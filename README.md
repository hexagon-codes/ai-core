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
- **能力矩阵** - `catalog` 统一登记 Provider / Model / Modality / Feature，支持上层做模型选择、运营展示和 conformance 校验
- **生产级上游传输层** - `transport` 统一 HTTP client、超时、错误诊断、Header 安全、错误体截断、RequestID / Retry-After 提取和 SSRF/network policy
- **智能路由** - 多 Provider 路由器，支持轮询、加权、最低延迟、降级等策略；任务感知智能路由
- **可取消 Token 计数** - 可选 `ContextTokenCounter` 支持取消和截止时间，兼容 helper 保留旧 Provider
- **用量追踪** - Token 消耗统计和成本估算（原子累加器，裁剪后数值一致），支持请求追踪器
- **结构化输出** - ResponseFormat 支持 JSON 模式和 JSON Schema 约束（含 JSON Schema 2020-12 方言）

## 安装

```bash
go get github.com/hexagon-codes/ai-core@v0.2.8
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

需要让 Token 计数响应取消或截止时间时，使用兼容入口：

```go
tokens, err := llm.CountTokensContext(ctx, r, req.Messages)
```

该入口优先调用可选的 `llm.ContextTokenCounter`。旧 Provider 无需修改即可继续使用，但旧同步计数一旦开始便无法被 helper 强制取消；路由健康检查会跳过这类 legacy-only Provider，并保留其既有健康状态。

### OpenAI-compatible Provider

```go
import (
    "fmt"

    "github.com/hexagon-codes/ai-core/llm"
    "github.com/hexagon-codes/ai-core/llm/compatible"
)

// apiKey 为空时会读取 profile 对应的环境变量，例如 OPENROUTER_API_KEY。
provider, err := compatible.NewProfile(compatible.ProfileOpenRouter, "")
if err != nil {
    panic(err)
}

resp, err := provider.Complete(ctx, llm.CompletionRequest{
    Model: "openai/gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: "用三句话解释 RAG"},
    },
})
if err != nil {
    panic(err)
}
fmt.Println(resp.Content)
```

已内置 `openrouter`、`groq`、`together`、`perplexity`、`xai`、`mistral`、`fireworks`、`deepinfra`、`siliconflow-cn`、`moonshot-cn`、`baichuan-cn`、`stepfun-cn`、`modelark-cn`、`modelark-global`、`doubao-cn`、`doubao-global` 等 profile；私有兼容网关可使用 `compatible.New()` + `WithBaseURL()` + `WithModel()` 配置。

### 媒体长任务

```go
import (
    "fmt"
    "os"
    "time"

    "github.com/hexagon-codes/ai-core/media/video"
)

seedance := video.NewSeedanceCN(os.Getenv("ARK_API_KEY"))
svc := video.NewService(map[string]video.Provider{
    seedance.Name(): seedance,
}, seedance.Name())

status, err := svc.SubmitAndWait(ctx, "", video.Request{
    Prompt:         "一艘白色帆船穿过清晨海雾",
    Ratio:          "16:9",
    Duration:       5,
    IdempotencyKey: "job-20260702-demo",
}, 2*time.Second)
if err != nil {
    panic(err)
}
fmt.Println(status.VideoURL)
```

计费型 Submit 请求只有在设置 `IdempotencyKey` 时才启用自动重试，避免上游 5xx/网络抖动场景下重复创建任务和重复计费。图像异步 Provider 可使用 `media/image.SubmitAndWait`，视频 Provider 可使用 `Service.SubmitAndWait`。

### 能力矩阵与网络策略

```go
import (
    "fmt"
    "time"

    "github.com/hexagon-codes/ai-core/catalog"
    "github.com/hexagon-codes/ai-core/llm"
    "github.com/hexagon-codes/ai-core/llm/openai"
    "github.com/hexagon-codes/ai-core/media/video"
)

safeOpenAI := openai.New(openaiKey,
    openai.WithNetworkPolicy(llm.PublicNetworkPolicy("api.openai.com")),
    openai.WithRequestTimeout(30*time.Second),
    openai.WithStreamIdleTimeout(90*time.Second),
)

registry := catalog.NewRegistry()
seedance := video.NewSeedanceCN("")
registry.RegisterProvider(seedance)

rows := registry.Find(catalog.Query{
    Modality: catalog.ModalityVideo,
    Feature:  catalog.FeatureAsyncTask,
})
fmt.Println(safeOpenAI.Name(), len(rows))
```

`catalog` 只登记 Provider / Model / Modality / Feature / Cost 等静态能力，便于上层做模型选择、运营展示和 conformance 校验；`transport` 负责共享 HTTP 出口、结构化 `ProviderError`、错误体截断、重试与 SSRF/network policy。

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
| `llm/compatible` | OpenAI-compatible 通用适配器（OpenRouter / Groq / Together / Perplexity / xAI / Mistral / Fireworks / DeepInfra / SiliconFlow / Moonshot / ModelArk / Doubao 国内外等） |
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
| `catalog` | Provider 能力矩阵与模型能力注册表 |
| `transport` | 共享上游 HTTP 传输层：诊断、超时、Header 安全、SSRF/network policy |
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
| Doubao / ModelArk 国内外 | doubao-seed-*, doubao-pro-*, BytePlus ModelArk endpoint | OpenAI-compatible、流式、函数调用、视觉、区域 profile |
| 文心一言 | ernie-4.5-8k, ernie-4.0-8k, ernie-3.5-8k, ernie-x1 | 流式 |
| Ollama | llama3.2, llama3.1, qwen2.5, mistral, codellama, llava | JSON Lines 流式、函数调用、视觉、模型能力发现 |
| OpenAI-compatible 长尾 | OpenRouter、Groq、Together、Perplexity、xAI、Mistral、Fireworks、DeepInfra、SiliconFlow、Moonshot、Baichuan、StepFun | 统一 Complete / Stream / Embedding |

## 支持的媒体 Provider

| Modality | Provider | 模型 / 通道示例 | 说明 |
|---|---|---|---|
| 图像 | OpenAI / Zhipu / OpenAI-compatible | DALL-E、gpt-image-1、CogView、自托管兼容网关 | 同步生成 |
| 图像 | BFL FLUX | flux-pro-1.1、flux-pro-1.1-ultra、flux-kontext-* | 官方异步 Submit→Poll |
| 图像 | Async compatible | Midjourney-compatible、Ideogram、Recraft、Stability、企业代理 | 合规网关 Submit→Poll |
| 视频 | Seedance / Doubao 国内 | doubao-seedance-*，火山方舟中国区 | 官方异步 Submit→Poll |
| 视频 | Seedance 全球 | dreamina-seedance-*，BytePlus ModelArk 海外区 | 官方异步 Submit→Poll |
| 视频 | Kling | kling-v3、kling-v2-1-master、kling-v1-6 | 官方/兼容异步 Submit→Poll，支持 AK/SK JWT 或 Bearer |
| 视频 | Vidu | viduq3-*、viduq2-*、vidu2.0 | 官方异步 Submit→Poll |
| 视频 | Google Veo | veo-3.1-*、veo-3.0-* | Gemini API long-running operation |
| 视频 | Video compatible | Midjourney-compatible、Runway、Pika、Luma、Hailuo、Wan、企业代理 | 合规网关 Submit→Poll |
| 语音 | OpenAI / Azure / Edge / MiniMax / ElevenLabs | Whisper、OpenAI TTS、Azure Speech、Edge Read Aloud、MiniMax、ElevenLabs | STT / TTS |

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
- **底座边界清晰** — ai-core 只负责模型适配、任务抽象、流式、工具、计量、能力矩阵和 Provider conformance；账号、钱包、队列、产物落盘、审核流程和运营配置由 hexclaw / hexeye-server 上层实现
- **外部调用不持锁** — 调用 LLM/Embedder 等外部服务前释放锁，避免阻塞

## 许可证

[Apache License 2.0](LICENSE)
