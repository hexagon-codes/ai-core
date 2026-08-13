# Changelog

本文件记录 ai-core 的用户可见变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Changed
- CI 按公共 Go 开源库边界收敛为单一工作流，只验证本库可控的不变量；移除对 toolkit、hexagon、hexclaw 浮动默认分支的反向强绑定，下游项目改由消费方在依赖升级时验证。
- 最低 Go 与当前稳定 Go 分层测试，race 只运行一次；新增 `gofmt`、`go mod tidy -diff`，移除重复 build、无人消费的 coverage 产物与 tag 重复运行。
- lint 改用 PR / push 的精确 Git 基线，不再假设 `HEAD~1`；漏洞扫描删除日志字符串豁免并恢复失败关闭。
- API 兼容检查并入统一 CI，动态选择目标分支上最新正式 SemVer tag，并以 `apidiff` 只比较公开 API；不再吞掉工具错误或受版本建议时序影响。
- GitHub Actions 升级为原生 Node 24 的版本并继续固定完整 commit SHA；checkout 不持久化凭据，工作流启用并发取消，所有 job 增加超时。

## [0.2.8]

### Added
- `llm`：新增可选 `ContextTokenCounter` 与 `CountTokensContext` 兼容分派。既有 `Provider` / `TokenCounter` 方法集保持不变；支持新能力的实现可在 Token 计数期间响应取消和截止时间，旧实现继续同步回退。

### Fixed
- `llm/router`：健康检查的 5 秒 deadline 现在会传入 context-aware Token 计数实现；legacy-only Provider 不再进入不可取消探测，而是保留既有健康状态。Router、官方 Provider、OpenAI-compatible Provider 与内置中间件会保留真实的 context-aware 能力边界。

## [0.2.7]
### Added
- `store/vector/qdrant`：新增 `MaxResponseBytes`、`WithMaxResponseBytes`、类型化配置/HTTP/文档/检索错误，以及显式的 `PointIDStrategy` 迁移开关。
- `schema`：`Schema` 新增 `anyOf`、`oneOf`、`allOf`、`not` 组合关键字；仅由组合关键字构成的 Schema 会省略空 `type`，并在 JSON 序列化时保留各分支约束。
- `llm` / `transport`：新增操作级自动重放契约 `OperationSafety` / `WithOperationSafety`；标记为 `OperationSafetyNonIdempotent` 时，LLM 重试中间件、OpenAI 响应解码重试、Router 重试与 fallback、共享 HTTP transport 均不自动重放，未标记请求保持既有重试策略。
- `streamx`：`Chunk` 新增 `ReasoningDisclosure`，并提供 `ReasoningDisclosureEvidence` 可信来源契约；OpenAI-compatible 与 Ollama 的推理数据块会附带已知来源和方言，只有明确公开且 Provider / Model 证据完整的冻结路由可标记为 `visible`，缺失证据或未知方言失败关闭为 `not_exposed`。
- `catalog`：新增 Provider / Model 能力矩阵注册表，统一描述 `Modality`、`Feature`、上下文窗口、异步能力、官方 API 标记、来源和成本信息；Provider 可实现 `catalog.Provider` 暴露 `Capabilities()`，上层可通过 `Registry` + `Query` 做模型选择、运营展示和 conformance 校验。
- `transport`：新增共享上游 HTTP 传输层，提供结构化 `ProviderError`、错误体限长与请求预览、RequestID / Retry-After 提取、安全 Header 合并、请求超时、流式 idle timeout、瞬时错误重试和 SSRF/network policy。`llm.ProviderError`、`llm.NetworkPolicy`、`llm.ErrNetworkPolicy` 作为别名保留兼容。
- `llm/compatible`：新增 OpenAI-compatible 通用 Provider，支持 `Complete` / `Stream` / `Embed` / `Models` / `Capabilities`；内置 OpenRouter、Groq、Together、Perplexity、xAI、Mistral、Fireworks、DeepInfra、SiliconFlow、Moonshot、Baichuan、StepFun、ModelArk、Doubao 国内外 profile。
- `media/image`：新增 BFL FLUX 官方异步 Provider、异步兼容网关 Provider、Midjourney-compatible 便捷构造器，以及 `AsyncImageProvider` / `SubmitAndWait` 同步包装。
- `media/video`：新增 Seedance / Doubao 中国区与全球区、Kling、Vidu、Google Veo、兼容网关 Provider；支持首帧/尾帧、音轨、画幅、时长、回调、幂等键和能力矩阵输出。
- `media/voice`：新增 ElevenLabs TTS Provider。

### Changed
- **BREAKING（Qdrant 持久数据）**：Qdrant 新集合的默认 point ID 从易碰撞的 31 倍 `uint64` 哈希改为 SHA-256 派生 UUIDv8。已有集合升级前应完成重建；迁移窗口可显式使用已弃用的 `PointIDLegacyHash31` 读取旧映射，但不得继续用于新数据。
- `store/vector/qdrant`：构造阶段严格校验 host/port/collection/dimension/distance/timeout；写入与删除固定 `wait=true`，只有集合查询明确返回 404 时才自动创建，`Clear` 不再吞掉删除失败；批配置非法时在启动 goroutine 前失败关闭。
- CI：第三方 GitHub Actions 全部固定到不可变 commit SHA，`gorelease` / `govulncheck` 固定工具版本。
- `llm/openai`：仅当请求以 `ReasoningPolicyScopeStructuredVisionRecognition` 显式声明结构化视觉识别作用域时，OpenAI 推理模型才会把 `thinking` 元数据推断为标准 `reasoning_effort`（`on` → `high`，`off` → 模型支持的最低强度；GPT-5.1+ 为 `none`、GPT-5 为 `minimal`、o1/o3/o4 与 `codex-*` 为 `low`）；零值作用域保持既有 wire 不变，显式 `reasoning_effort` 仍独立透传且优先，非推理模型不做推断，私有或 loopback compatible 网关也不会丢失该参数，并继续与厂商方言 `enable_thinking` 隔离。
- 多数 LLM 与媒体 Provider 迁移到共享 `transport`，补齐可配置 HTTP client、额外安全 Header、请求超时、流式 idle timeout、network policy 和结构化错误诊断。
- `llm` 中间件重试逻辑识别结构化 `ProviderError`：408/409/429/5xx 可重试，400/401/402/403/404/422 与 network policy 错误不重试。
- `llm` 缓存默认 key 改为完整请求及控制平面策略作用域的稳定 JSON hash，覆盖 `MultiContent`、`Metadata`、`ReasoningPolicyScope`、tools、response format 等会影响上游语义的字段；不可序列化请求会跳过缓存读写，避免跨策略复用、伪 key 漂移和缓存堆积。
- `transport`：新增仅由调用方 context 绑定、可按 action 限定的 `BeforeSendHook`，在全部本地前置检查通过后、紧邻 `http.Client.Do` 前执行；hook 失败保持零请求，hook-bound 请求不自动跟随 307/308 保留方法重定向，且 `llm` 缓存会绕过命中以保证真实物理发送边界可观察。
- `llm/ollama`：适配新版 Ollama 元数据与媒体能力，`/api/tags` 读取 `model`、`capabilities`、`context_length`，缺失能力时回退 `/api/show`；OpenAI 多模态 `image_url` 会转换为 Ollama `images` base64 数组；请求默认写入 `num_ctx`，并支持通过 metadata 覆盖。
- `streamx`：`CustomFormat` 支持原始 JSON Lines 流，兼容 Ollama 等非 SSE 上游，同时保留 `data:` SSE 解析路径。
- `media` 统一任务状态归一化，新增 `TaskCancelled`，`TaskState.Terminal()` 覆盖成功、失败和取消。
- 依赖：`github.com/hexagon-codes/toolkit` v0.2.3 → v0.2.6；`go` 指令随 toolkit 要求提升至 1.25.7。
- 依赖：`github.com/hexagon-codes/toolkit` v0.2.6 → v0.3.4；`go` 指令随 toolkit 要求提升至 1.25.12。
- `llm/middleware`、`llm/router`、`gateway/llmcall`：适配 toolkit v0.3.x 的 `util/retry` 重构——重试条件入口统一为 `If`，指数退避显式选择 `retry.DelayType(retry.ExponentialBackoff)` 搭配 `Multiplier`，重试耗尽错误固定同时包装 `ErrMaxAttemptsReached` 与最终业务错误；新增契约测试钉死"倍数参数必须显式声明指数策略"。
- CI 门禁精简：`VERSION` 校验改为只检查 SemVer 格式（不再与 git tag 精确绑定，避免已推送 tag 无法修复的死锁）；删除断言 workflow 文件内容的元层次合同测试；移除 codecov 上传（tokenless 上传已被 GitHub 拒绝）。

### Fixed
- `streamx`：处理 `io.Reader` 同时返回末帧数据与 `io.EOF` 的合法情况，避免无尾换行 SSE/JSONL 丢失最后一个 token、finish reason 或 tool delta。
- `store/vector/qdrant`：阻止不同文档 ID 的确定性哈希碰撞覆盖、metadata 覆盖内部 `_original_id/content/created_at` 字段、无界响应读取、认证错误触发错误建库，以及 `Clear` 删除失败后伪成功。
- `streamx.Stream.Close`：先关闭底层 reader 再等待后台 goroutine，避免 processLoop 阻塞在读输入时 `Close()` 卡住；`Collect()` 在 chunk channel 关闭后补读一次错误通道，避免漏报末尾错误。
- `media/video`：`content_moderated` 映射为失败终态，避免审核拦截后 `WaitFor` 长时间继续轮询。
- `media` Submit 重试策略：计费型任务创建请求未提供 `IdempotencyKey` 时禁用自动重试，避免二义性失败后重复创建任务和重复计费。

## [0.1.11]
### Added
- `template`：新增**有序内容块**（content block）模型 —— `Block` / `Blocks` / `BlockBuilder`，对齐 Anthropic Messages API 的 content block（`text` / `thinking` / `tool_use` / `tool_result`）。块的**顺序即语义**，可保真表达多步 agent（ReAct / orchestrate）的「先说什么 → 再调什么 → 又说什么」交错结构，供持久化与 replay。与输入侧多模态 `ContentPart` 职责分离、互不污染。提供构造器、`Blocks.Text()`/`ToolUses()` 退化辅助、`Validate()`（校验 `tool_result` 必须有在其之前出现的配对 `tool_use`），及 `BlockBuilder` 增量拼装。

## [0.1.10]
### Changed
- 依赖：`github.com/hexagon-codes/toolkit` v0.2.2 → v0.2.3。

## [0.1.9]
### Changed
- 依赖：`github.com/hexagon-codes/toolkit` v0.2.1 → v0.2.2。

## [0.1.8]
### Fixed
- `media/image`：`truncateForError` 改为 **rune-safe 截断**（委托 `toolkit/lang/stringx.SubString`）—— 原以字节切片 `s[:maxLen]`，当 `maxLen` 落在多字节 UTF-8 字符（如 CJK）中间时会切断码点产出乱码（BUG-20260625 F-4）；无需截断时不再追加省略号，与旧实现「`<=maxLen` 原样返回」语义一致。

## [0.1.7]
### Changed
- `gateway/llmcall`：`CallWithProgress` 的重试退避委托给 `toolkit/util/retry.DoWithContext`，替换手写 `for attempt` 循环。**行为保真**：退避序列 `baseBackoff*2^(n-1)`、仅瞬时错误重试（`If(isTransient)`）、ctx 取消/超时透传 `ctx.Err()`、`attempts` 计数与失败错误包装 `(attempts=maxRetries)` 均与旧实现逐项一致。
- 依赖：`github.com/hexagon-codes/toolkit` v0.2.0 → v0.2.1（`net/httpx.RawClient` 默认遵循 `HTTP(S)_PROXY`/`NO_PROXY`）。

## [0.1.6]
### Changed
- `streamx`：`Timeout` 改为**无损语义** —— 超时返回 `ErrStreamTimeout` 不再丢弃在途元素，该元素在后续 `recv` 按序投递（原实现会丢弃超时后首条到达的元素，造成静默数据丢失）。与 stdlib `select` + 超时不消费 channel 值一致；"丢弃慢元素 / 只取最新"请改用 `Throttle`/`Debounce`/`Window`。
- `llm/failover`：`ClassifyError` 凭证关键词收窄 —— 裸"无效"不再判为 `FailInvalidKey`，避免"无效参数 / 无效模型 / 无效分辨率"等非凭证错误被误判为凭证无效；非凭证错误改归 `FailUnknown`。

## [0.1.5]
### Changed
- `llm/failover`：`ClassifyError` 错误分类加固 —— 移除裸 HTTP 状态码子串匹配（避免 request id / 模型名误命中），无效凭证判定置于配额之前以便快速失败。
- `llm/router`：`ExecuteWithRetry` 改为 ctx 感知的指数退避重试（每次尝试前检查 ctx，退避受取消/超时中断），ctx 取消/超时直接透传。
- `meter`：累计成本由微美元 `int64` 改为 `float64` 精度原子累加（CAS），消除单条低额成本取整丢失，使全局成本与按模型成本一致。
- `streamx`：`Timeout` 用持久化后台 goroutine + `done` 通道 + finalizer，未显式 `Close()` 也不泄漏 goroutine；`Distinct` 对可比较类型用 set 做 O(1) 命中判定。

### Fixed
- `streamx/Distinct`：回退到 `equals` 线性扫描时未迁移已见集合，导致之前已见的可比较元素被再次发射；改为先迁移已见集合再回退。

### Dependencies
- 依赖 `toolkit` 升级 v0.1.0 → v0.2.0。ai-core 未使用 v0.2.0 破坏性变更涉及的包（`crypto/sign` 签名 wire 格式、`lang/contextx` `Pool.Wait()` 行为）；所用 `util/retry`/`lang/syncx`/`util/idgen`/`cache/local`/`util/logger`/`net/httpx` 在 v0.1.0→v0.2.0 仅 README 变更，`retry` 公共 API 签名逐字不变，无源码影响。

## [0.1.4]
### Added
- `streamx`：下沉通用 `StreamReader`/`Stream` 四件套（Concat/Merge/Copy + Close 契约）+ gRPC 分帧解析。
- `media`：独立媒体子域（image/video/voice/voicechat + 统一 Submit→Poll 任务模型 + 同步包装）。
- `llm/cache`：`SemanticCache`（语义缓存 + singleflight + SQLite 持久化）。
- `llm/ernie`：新增百度文心一言（ERNIE）Provider。
- `gateway/llmcall`：带进度上报与瞬时错误重试的统一 LLM 调用入口。
- `llm` 能力接口（capabilities）：ISP 窄能力接口（Completer/Streamer/ChatProvider/TokenCounter/ModelLister）。
- `llm` 故障转移（failover）：`FailoverReason` 枚举 + `ClassifyError`。
- `schema`：JSON Schema 2020-12 方言（`Draft2020_12` + `WithDialect`）+ `Strict()`。
- `memory`：条目支持溯源血缘（ParentID/CauseBy）与多模态（Modality/MediaRef）。

### Changed
- `memory`：ID 生成复用 `toolkit/util/idgen.NanoID`（去手搓计数器+随机字节）。

### Dependencies
- 依赖 `toolkit` 升级至 v0.1.0（四仓 lockstep 发布）。

### Added (治理)
- CI（build/vet/race/lint/govulncheck）、`.golangci.yml`、`CONTRIBUTING.md`、`COMPATIBILITY.md`。

## [0.1.3]
- 基线版本（llm / tool / memory / schema / streamx / template / tokenizer / meter）。
