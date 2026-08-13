# 兼容性与稳定性策略 — ai-core

ai-core 是面向任意 Go 应用与 Agent 框架的开源 AI 基础能力库。库自身只依赖 `go.mod` 中声明最低所需版本的直接依赖，不把任何具体消费项目的源码、默认分支或测试结果作为发布前提。

## SemVer 承诺

- 遵循 [SemVer](https://semver.org/lang/zh-CN/)，导出标识符（公开 API）是兼容性契约。
- v1 及以上版本的 patch / minor 不得破坏导出 API，破坏式变更只能在 major 版本发布；v0.x 的 patch 不得破坏 API，破坏式变更至少提升 minor，并在 CHANGELOG 显著标注 BREAKING。
- 内部包（`internal/`）与未导出标识符不属于公开兼容合同。
- 已发布的 tag 不删除、不移动；修复通过新版本发布。

## 公开 API

- `catalog`、`transport`、`llm/compatible` 及 `media/*` 中导出的 Provider 构造器、Option、请求与响应结构体均按 SemVer 维护。
- `llm.ProviderError`、`llm.NetworkPolicy`、`llm.ErrNetworkPolicy` 是 `transport` 同名类型和值的别名；调用方可继续使用 `llm` 路径，也可逐步迁移到 `transport` 路径，`errors.As` / `errors.Is` 在两种路径间等价。
- `catalog.Capability` 是调用方做模型选择与 conformance 校验的稳定数据契约；新增字段只做加法，已有字段语义不得在 patch / minor 中收窄。
- Network policy 默认不启用，以保持本地服务与私有兼容网关的向后兼容；显式传入零值 `NetworkPolicy` 时启用 public HTTPS 策略。
- `llm.Provider` / `llm.TokenCounter` 保留既有 `CountTokens` 方法集；可取消计数通过 opt-in 的 `llm.ContextTokenCounter` 提供。`llm.CountTokensContext` 对旧实现仅保证调用前检查 context，旧同步调用开始后无法被强制取消；健康检查会跳过旧实现并保留既有健康状态。

## 自动门禁

统一的 `.github/workflows/ci.yml` 只验证本库可控的不变量：

1. 最低支持 Go 版本运行全量测试，当前稳定 Go 版本运行全量 race 测试。
2. `gofmt`、`go mod tidy -diff`、`go vet` 与 golangci-lint 校验新增代码。
3. `govulncheck` 对依赖与可达代码执行失败关闭的漏洞扫描。
4. Pull Request 与 main push 均以前一状态可达的最新正式 SemVer tag 为基线执行 `apidiff`；工具错误与破坏性公开 API 变更都会阻断。

具体消费项目的兼容验证由消费方在升级 ai-core 版本时执行，不反向决定 ai-core 自身 CI 是否通过。
自动门禁默认阻断所有破坏性变更，包括 v0.x；确需发布 v0.x 破坏性变更时，必须先获得维护者对具体变更与迁移方案的明确批准，再按 minor 版本和 BREAKING 记录处理，不能通过吞掉检查错误放行。

## 弃用与发布

- 弃用先标 `// Deprecated: 用 X 替代。将在 vN 移除。`，至少保留一个 minor 周期并记录到 CHANGELOG，到期后才可删除。
- 移除导出 API 属于破坏式变更，按上述 SemVer 规则发布。
- ai-core 是源码 Go module，不维护应用部署或制品发布流水线；通过不可移动的 SemVer tag 发布。

## 消费方升级建议

- 固定明确版本，阅读 CHANGELOG 后再升级。
- 在消费项目自身的 CI 中运行编译、测试与必要的集成回归；不要依赖任意具名下游项目替代自身验证。
