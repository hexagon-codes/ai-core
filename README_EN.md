# ai-core

[中文](README.md) | English

A Go library providing core AI capabilities for the [Hexagon](https://github.com/hexagon-codes/hexagon) AI Agent framework.

## Features

- **Unified LLM Interface** — One API, multiple providers (OpenAI, Anthropic, DeepSeek, Gemini, Qwen, Doubao, ERNIE, Ollama)
- **Middleware Pipeline** — Composable provider decorators: retry (with non-retryable error detection), rate limiting, timeout, callbacks, caching (with singleflight to prevent thundering herd, including semantic cache)
- **Streaming** — Unified SSE streaming with both callback and channel modes; generic `StreamReader` quartet (Concat/Merge/Copy + Close contract) and gRPC frame parsing
- **Tool System** — Type-safe tool definitions with automatic JSON Schema generation from Go structs
- **Memory System** *(Experimental)* — Multiple memory strategies (buffer, summary, vector retrieval, multi-layer, entity memory), entries support provenance lineage and multimodal content. This interface is experimental and may change in future versions
- **Media Generation** — Dedicated media subdomain (image / video / voice / voice chat) with a unified Submit→Poll async task model plus synchronous wrappers
- **Capability Catalog** — `catalog` registers Provider / Model / Modality / Feature rows for model selection, operational display, and conformance checks
- **Production Upstream Transport** — `transport` centralizes HTTP clients, timeouts, diagnostics, safe headers, bounded error bodies, RequestID / Retry-After extraction, and SSRF/network policy
- **Smart Routing** — Multi-provider router with round-robin, weighted, least-latency, fallback strategies; task-aware intelligent routing
- **Usage Tracking** — Token consumption statistics and cost estimation (atomic cumulative counter, consistent after pruning) with request tracing
- **Structured Output** — ResponseFormat supporting JSON mode and JSON Schema constraints (including the JSON Schema 2020-12 dialect)

## Installation

```bash
go get github.com/hexagon-codes/ai-core@v0.1.4
```

## Quick Start

### Basic Completion

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
            {Role: llm.RoleUser, Content: "Hello!"},
        },
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Content)
}
```

### Streaming

```go
stream, err := provider.Stream(ctx, llm.CompletionRequest{
    Model: "gpt-4o",
    Messages: llm.NewMessages("You are an assistant", "Tell me a joke"),
})
if err != nil {
    panic(err)
}
defer stream.Close()

for chunk := range stream.Chunks() {
    fmt.Print(chunk.Content)
}
```

### Tool Calling

```go
import "github.com/hexagon-codes/ai-core/tool"

type WeatherInput struct {
    City string `json:"city" desc:"City name" required:"true"`
}

weatherTool := tool.NewFunc("get_weather", "Get city weather",
    func(ctx context.Context, input WeatherInput) (string, error) {
        return fmt.Sprintf("%s: Sunny, 25°C", input.City), nil
    },
)

// Convert to LLM tool definition
toolDef := llm.NewToolDefinition(
    weatherTool.Name(),
    weatherTool.Description(),
    weatherTool.Schema(),
)

resp, _ := provider.Complete(ctx, llm.CompletionRequest{
    Model:    "gpt-4o",
    Messages: []llm.Message{{Role: llm.RoleUser, Content: "What's the weather in Beijing?"}},
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

### Middleware

```go
import (
    "github.com/hexagon-codes/ai-core/llm"
    "github.com/hexagon-codes/ai-core/llm/cache"
)

// Chain multiple middleware: retry → rate limit → cache
enhanced := llm.Chain(provider,
    llm.WithRetry(3, time.Second),       // Exponential backoff retry (skips 401/403 etc.)
    llm.WithRateLimit(10),               // 10 QPS token bucket rate limiting
    llm.WithTimeout(30 * time.Second),   // Request timeout (does not affect Stream)
    llm.WithCache(cache.NewMemoryCache(), nil), // LRU in-memory cache (singleflight to prevent thundering herd)
)

resp, _ := enhanced.Complete(ctx, req)
```

### Multi-Provider Routing

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

// Use router (automatically selects optimal provider)
resp, _ := r.Complete(ctx, req)
```

### OpenAI-Compatible Providers

```go
import (
    "fmt"

    "github.com/hexagon-codes/ai-core/llm"
    "github.com/hexagon-codes/ai-core/llm/compatible"
)

// When apiKey is empty, the profile's environment variable is used,
// for example OPENROUTER_API_KEY.
provider, err := compatible.NewProfile(compatible.ProfileOpenRouter, "")
if err != nil {
    panic(err)
}

resp, err := provider.Complete(ctx, llm.CompletionRequest{
    Model: "openai/gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: "Explain RAG in three sentences"},
    },
})
if err != nil {
    panic(err)
}
fmt.Println(resp.Content)
```

Built-in profiles include `openrouter`, `groq`, `together`, `perplexity`, `xai`, `mistral`, `fireworks`, `deepinfra`, `siliconflow-cn`, `moonshot-cn`, `baichuan-cn`, `stepfun-cn`, `modelark-cn`, `modelark-global`, `doubao-cn`, and `doubao-global`. Private compatible gateways can use `compatible.New()` with `WithBaseURL()` and `WithModel()`.

### Media Async Tasks

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
    Prompt:         "A white sailboat crossing morning sea fog",
    Ratio:          "16:9",
    Duration:       5,
    IdempotencyKey: "job-20260702-demo",
}, 2*time.Second)
if err != nil {
    panic(err)
}
fmt.Println(status.VideoURL)
```

Billing-sensitive Submit calls are retried automatically only when `IdempotencyKey` is set, avoiding duplicate task creation and duplicate billing after ambiguous 5xx/network failures. Async image providers can use `media/image.SubmitAndWait`; video providers can use `Service.SubmitAndWait`.

### Capability Catalog And Network Policy

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

`catalog` records static Provider / Model / Modality / Feature / Cost metadata for model selection, operational display, and conformance tests. `transport` handles shared upstream HTTP behavior, structured `ProviderError`, bounded error bodies, retries, and SSRF/network policy.

### Memory System *(Experimental)*

> **Note:** The Memory interface is experimental and may change in future versions.

```go
import "github.com/hexagon-codes/ai-core/memory"

// Buffer memory — keep last N messages
buf := memory.NewBuffer(100)
buf.Save(ctx, memory.NewUserEntry("Hello"))
buf.Save(ctx, memory.NewAssistantEntry("Hi! How can I help you?"))

// Get() / Delete() return memory.ErrNotFound when the entry does not exist
entry, err := buf.Get(ctx, "some-id")
if errors.Is(err, memory.ErrNotFound) {
    // handle not found
}

// Summary memory — auto-compress to summary when threshold exceeded
// doSummarize is concurrency-safe (Clear + re-insert under lock)
sum := memory.NewSummaryMemory(summarizer, memory.WithMaxEntries(20))

// Vector memory — semantic retrieval
vec := memory.NewVectorMemory(embedder)
results, _ := vec.SemanticSearch(ctx, "architecture discussion from earlier", 5)

// Multi-layer memory — working → short-term → long-term
// Transfer() uses a dedicated transferMu to avoid blocking reads during embedder calls
multi := memory.NewMultiLayerMemory(
    memory.WithSummarizer(summarizer),
    memory.WithEmbedder(embedder),
)
```

## Package Structure

| Package | Description |
|---------|-------------|
| `llm` | LLM Provider abstraction, middleware (retry/rate-limit/timeout/callback/cache) |
| `llm/openai` | OpenAI implementation (GPT-4o, GPT-4-Turbo, o1, o3-mini, etc.) |
| `llm/compatible` | Generic OpenAI-compatible adapter (OpenRouter / Groq / Together / Perplexity / xAI / Mistral / Fireworks / DeepInfra / SiliconFlow / Moonshot / ModelArk / Doubao and more) |
| `llm/anthropic` | Anthropic Claude implementation |
| `llm/deepseek` | DeepSeek implementation |
| `llm/gemini` | Google Gemini implementation |
| `llm/qwen` | Qwen (Alibaba) implementation |
| `llm/ark` | Doubao (ByteDance) implementation |
| `llm/ernie` | ERNIE (Baidu) implementation |
| `llm/ollama` | Ollama local model implementation |
| `llm/router` | Multi-provider intelligent routing, task-aware routing (SmartRouter) |
| `llm/cache` | LRU in-memory cache + semantic cache (TTL, singleflight, SQLite persistence) |
| `media` | Media generation subdomain: `media/image` `media/video` `media/voice` `media/voicechat` (Submit→Poll task model) |
| `catalog` | Provider capability matrix and model capability registry |
| `transport` | Shared upstream HTTP transport: diagnostics, timeouts, safe headers, SSRF/network policy |
| `gateway/llmcall` | LLM call gateway: unified entry point with progress reporting and transient-error retry |
| `memory` | Agent memory system (buffer/summary/vector/multi-layer/entity), with provenance lineage and multimodal entries *Experimental* |
| `tool` | Tool definition and registration |
| `schema` | JSON Schema generation (reflection from Go structs, supports the JSON Schema 2020-12 dialect) |
| `streamx` | Unified streaming abstraction (OpenAI/Claude/Gemini formats) + generic `StreamReader` / gRPC framing |
| `template` | Prompt template engine (multimodal support) |
| `tokenizer` | Token count estimation |
| `meter` | Usage statistics and cost tracking (atomic cumulative cost counter) |
| `store/vector` | Vector storage abstraction (in-memory/Qdrant) |

## Supported LLM Providers

| Provider | Model Examples | Features |
|----------|---------------|----------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o3-mini | Streaming, function calling, vision |
| Anthropic | claude-opus-4, claude-sonnet-4, claude-3.5-sonnet, claude-3.5-haiku | Streaming, function calling, vision |
| DeepSeek | deepseek-chat, deepseek-reasoner | Streaming, function calling |
| Gemini | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash | Streaming, function calling, vision, embedding |
| Qwen | qwen-turbo, qwen-plus, qwen-max, qwen-vl-max | Streaming, function calling, vision |
| Doubao | doubao-pro-*, doubao-lite-*, doubao-vision-pro-* | Streaming, function calling, vision |
| Doubao / ModelArk CN and global | doubao-seed-*, doubao-pro-*, BytePlus ModelArk endpoints | OpenAI-compatible, streaming, function calling, vision, regional profiles |
| ERNIE | ernie-4.5-8k, ernie-4.0-8k, ernie-3.5-8k, ernie-x1 | Streaming |
| Ollama | llama3.2, llama3.1, qwen2.5, mistral, codellama, llava | JSON Lines streaming, function calling, vision, model capability discovery |
| Long-tail OpenAI-compatible | OpenRouter, Groq, Together, Perplexity, xAI, Mistral, Fireworks, DeepInfra, SiliconFlow, Moonshot, Baichuan, StepFun | Unified Complete / Stream / Embedding |

## Supported Media Providers

| Modality | Provider | Model / Channel Examples | Notes |
|----------|----------|--------------------------|-------|
| Image | OpenAI / Zhipu / OpenAI-compatible | DALL-E, gpt-image-1, CogView, self-hosted compatible gateways | Synchronous generation |
| Image | BFL FLUX | flux-pro-1.1, flux-pro-1.1-ultra, flux-kontext-* | Official async Submit→Poll |
| Image | Async compatible | Midjourney-compatible, Ideogram, Recraft, Stability, enterprise proxies | Compliant gateway Submit→Poll |
| Video | Seedance / Doubao CN | doubao-seedance-*, Volcengine Ark China region | Official async Submit→Poll |
| Video | Seedance global | dreamina-seedance-*, BytePlus ModelArk global region | Official async Submit→Poll |
| Video | Kling | kling-v3, kling-v2-1-master, kling-v1-6 | Official/compatible async Submit→Poll, AK/SK JWT or Bearer |
| Video | Vidu | viduq3-*, viduq2-*, vidu2.0 | Official async Submit→Poll |
| Video | Google Veo | veo-3.1-*, veo-3.0-* | Gemini API long-running operation |
| Video | Video compatible | Midjourney-compatible, Runway, Pika, Luma, Hailuo, Wan, enterprise proxies | Compliant gateway Submit→Poll |
| Voice | OpenAI / Azure / Edge / MiniMax / ElevenLabs | Whisper, OpenAI TTS, Azure Speech, Edge Read Aloud, MiniMax, ElevenLabs | STT / TTS |

## Routing Strategies

| Strategy | Description |
|----------|-------------|
| `StrategyRoundRobin` | Round-robin, distribute requests evenly |
| `StrategyRandom` | Random selection |
| `StrategyLeastLatency` | Select provider with lowest latency |
| `StrategyLeastCost` | Select provider with lowest cost |
| `StrategyWeighted` | Distribute by weight |
| `StrategyFallback` | Try in order, fall back on failure |
| `StrategyModelMatch` | Auto-match provider based on requested model |

## Design Principles

- **Minimal Dependencies** — Depends only on the in-ecosystem `toolkit` (shared utility foundation) plus a few necessary libraries (such as the SQLite driver for the semantic cache); reuses toolkit instead of reinventing wheels
- **Interface-Driven** — Provider, Memory, Tool, VectorStore and other core types are interfaces for easy testing and extension
- **Concurrency-Safe** — All public types are thread-safe via `sync.RWMutex` or `atomic`
- **Functional Options** — Unified `With*()` option pattern for component configuration
- **Clear Foundation Boundary** — ai-core owns model adapters, task abstractions, streaming, tools, metering, capability catalog, and provider conformance; accounts, wallets, queues, artifact storage, review workflows, and operational configuration stay in upper layers such as hexclaw / hexeye-server
- **No Lock During External Calls** — Locks are released before calling LLM/Embedder services to avoid blocking

## License

[Apache License 2.0](LICENSE)
