# Changelog

本文件记录 ai-core 的用户可见变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.8]
### Fixed
- `media/image`：`truncateForError` 改为 **rune-safe 截断**（委托 `toolkit/lang/stringx.SubString`）—— 原以字节切片 `s[:maxLen]`，当 `maxLen` 落在多字节 UTF-8 字符（如 CJK）中间时会切断码点产出乱码（BUG-20260625 F-4）；无需截断时不再追加省略号，与旧实现「`<=maxLen` 原样返回」语义一致。

## [0.1.7]
### Changed
- `gateway/llmcall`：`CallWithProgress` 的重试退避委托给 `toolkit/util/retry.DoWithContext`，替换手写 `for attempt` 循环。**行为保真**：退避序列 `baseBackoff*2^(n-1)`、仅瞬时错误重试（`RetryIf(isTransient)`）、ctx 取消/超时透传 `ctx.Err()`、`attempts` 计数与失败错误包装 `(attempts=maxRetries)` 均与旧实现逐项一致。
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
