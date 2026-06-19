# Changelog

本文件记录 ai-core 的用户可见变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]
### Added
- `streamx`：下沉通用 `StreamReader`/`Stream` 四件套（Concat/Merge/Copy + Close 契约）+ gRPC 分帧。
- `media`：独立媒体子域（image/video/voice/voicechat + 统一 Submit→Poll 任务模型）。
- `llm/cache`：`SemanticCache`（语义缓存 + singleflight + SQLite 持久化）。
- `llm/capabilities`：ISP 窄能力接口（Completer/Streamer/ChatProvider/TokenCounter/ModelLister）。
- `llm/failover`：`FailoverReason` 枚举 + `ClassifyError`。
- `schema`：JSON Schema 2020-12 方言 + `Strict()`。

### Changed
- `memory`：ID 生成复用 `toolkit/util/idgen.NanoID`（去手搓计数器+随机字节）。

### Added (治理)
- CI（build/vet/race/lint/govulncheck）、`.golangci.yml`、`CONTRIBUTING.md`。

## [0.1.3]
- 基线版本（llm / tool / memory / schema / streamx / template / tokenizer / meter）。
