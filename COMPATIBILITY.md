# 兼容性与稳定性策略 — ai-core

ai-core 是面向任意 Go 应用与 Agent 框架的开源 AI 基础能力库。库自身只依赖 `go.mod` 中声明最低所需版本的直接依赖，不把任何具体消费项目的源码、默认分支或测试结果作为发布前提。

## SemVer 承诺

- 遵循 [SemVer](https://semver.org/lang/zh-CN/)，导出标识符（公开 API）是兼容性契约。
- v1 及以上版本的 patch / minor 不得破坏导出 API，破坏式变更只能在 major 版本发布；v0.x 的 patch 不得破坏 API，破坏式变更至少提升 minor，并在 CHANGELOG 显著标注 BREAKING。
- 内部包（`internal/`）与未导出标识符不属于公开兼容合同。
- 已发布的 tag 不删除、不移动；修复通过新版本发布。

## 公开 API

- 当前本地交付目标为 **v0.3.0**，最低 Go **1.26.10**，Toolkit 目标 **v0.3.7**。目标版本与本地候选验证不代表 tag 已创建、推送或公共代理可解析；发布时仍按下述既有流程核验。
- v0.3.0 的 `StreamReader.Copy(n>=2)` 转移源读取权，消费者只读取返回分支；原 reader 的关闭表示整体关闭。Gemini 流 parser 按响应独立实例化。完整调用迁移见 [CHANGELOG](CHANGELOG.md)。
- Copy传播Merge的SourceEOF时仍推进各分支游标；该标记只表示一个源结束，不关闭整个合并流。真正EOF、真实错误及Close继续按原所有权合同释放。
- `schema.Schema.AdditionalKeywords`保存尚无类型字段的JSON Schema关键字，参与序列化与本地Validator验证；解码保留额外数值字面精度。已声明的类型字段优先；唯一的同名形态补充是 `Items == nil` 时，通过 `AdditionalKeywords["items"]` 的 `[]any` 保留旧方言 tuple 数组，包括其布尔元素和引用。JSON 解码自动保留该形态；显式 `Items` 不被额外值覆盖。非法方言形态仍交给 Validator 报错，不静默删除约束。该能力负责表示保真，不等同于所有模型供应商都支持全部关键词；Provider能力仍以其实际声明为准。

- `catalog`、`transport`、`llm/compatible` 及 `media/*` 中导出的 Provider 构造器、Option、请求与响应结构体均按 SemVer 维护。
- `llm.ProviderError`、`llm.NetworkPolicy`、`llm.ErrNetworkPolicy` 是 `transport` 同名类型和值的别名；调用方可继续使用 `llm` 路径，也可逐步迁移到 `transport` 路径，`errors.As` / `errors.Is` 在两种路径间等价。
- `catalog.Capability` 是调用方做模型选择与 conformance 校验的稳定数据契约；新增字段只做加法，已有字段语义不得在 patch / minor 中收窄。
- Network policy 默认不启用，以保持本地服务与私有兼容网关的向后兼容；显式传入零值 `NetworkPolicy` 时启用 public HTTPS 策略。
- `llm.Provider` / `llm.TokenCounter` 保留既有 `CountTokens` 方法集；可取消计数通过 opt-in 的 `llm.ContextTokenCounter` 提供。`llm.CountTokensContext` 对旧实现仅保证调用前检查 context，旧同步调用开始后无法被强制取消；健康检查会跳过旧实现并保留既有健康状态。
- `streamx.Chunk.Usage` 是可选的单请求累计用量快照，nil 表示该帧未报告用量。OpenAI 格式的 `streamx.Result.Usage` 保留最后一份已报告快照，空 `choices` 的统计帧有效；调用方不得将同一次请求的快照逐帧累加，不同请求的结果仍可累加。
- `llm.GenerationOptions` 只表示生成参数；`Clone`/`ApplyTo` 保留参数的独立副本及错误出口，不负责消息、工具绑定或 Run 生命周期。框架应在实际请求构建处应用，不能仅保存配置却继续发送旧参数。
- `llm.AttemptReceipt` 按实际 HTTP AttemptID/ReceiptID 记录累计回执，`UsageFields` 表示供应商实际报告的维度。`AttemptLedger.Summary` 的 UnknownAttempts/PartialAttempts 必须与 Usage 一起解释；只报告 prompt/completion 不代表报告了 total，不能填造实报 TotalTokens。上层预算可另算已知下界，账本仍保留原始用量来源并按实际 attempt 幂等处理。
- `media/video.Service.Submit` 返回的不透明任务引用必须原样传入 `Poll`，调用方不得依赖其内部编码。`Poll` 继续接受可唯一解析的历史 `{provider}::{taskID}` 引用；存在歧义的旧引用会返回错误。提交成功后首次轮询失败时，`SubmitAndWait` 返回状态仍包含可恢复的 `TaskID`。
- `media/voice.ChainedTTS` 在 context 标记 `transport.OperationSafetyNonIdempotent` 时，仅对明确未被接纳的合成失败执行 Provider 回退；结果未知时停止，避免重复执行可能已接纳的请求。取消和超时也不会触发回退。`SynthesizeResult.Provider` 与 `IsDefinitiveSynthesisFailure` 为新增的兼容性 API。

## CI/CD 边界与门禁

ai-core 是公共源码 Go module，不是需要运行时部署的应用，因此不建设 CD、镜像发布、服务器部署或自动推广流水线。版本发布由维护者在验证完成后创建正式 SemVer tag；tag 是唯一的发布来源，已发布 tag 不删除、不移动，发现问题通过新的 patch 版本修复。

普通 Pull Request 只运行唯一的 `CI / Verify` 检查，验证本库可控且稳定的不变量：

1. 按 `go.mod` 声明的最低支持 Go 版本运行全量测试。
2. 执行 `gofmt`、`go mod tidy -diff`、`go vet` 和固定版本的 correctness-only lint。
3. 只在 Pull Request 上运行，不因合并后的 main push 再重复触发同一套必需检查。

安全审计与发布判断不属于普通代码验证：

- `govulncheck` 放在独立的定期或手动 `Security Audit` 流程中；它反映外部漏洞数据库和工具链变化，不得让无代码变化的普通 PR 因时间漂移反复失败。
- `apidiff` 和 SemVer 兼容性检查放在独立的发布前流程中，以明确指定的上一个正式 tag 为基线；工具或基线问题只阻断发布，不改变普通 Verify 的职责。
- 具体消费项目的编译、测试和集成回归由消费方在升级 ai-core 时执行，不反向决定 ai-core 普通 Verify 是否通过。

普通 Verify 仍应阻断当前代码真实违反的格式、模块、静态正确性或测试不变量。不得通过吞掉检查错误、动态猜测基线或修改 CI 配置来掩盖问题。

v1 及以上版本的破坏性公开 API 只能在 major 版本发布。v0.x 仍遵循以下明确规则：patch 版本不得破坏 API；确需破坏时至少提升 minor 版本，提前获得维护者对具体变更与迁移方案的明确批准，并在 CHANGELOG 显著标注 `BREAKING`。发布前兼容性流程必须拒绝未满足这些规则的 tag。

## 弃用与发布

- v1 及以后，弃用先标 `// Deprecated: 用 X 替代。将在 vN 移除。`，至少保留一个 minor 周期并记录到 CHANGELOG，到期后才可按 major 规则删除。
- v0.x 已明确采用一次性迁移的 breaking minor，可同步删除旧入口而不保留应用 shim；必须在 CHANGELOG 列明实际删除与迁移方式，不能以此跳过发布前的 API 核对。未列明删除的现有 API 不因本条自动弃用或移除。
- 移除导出 API 属于破坏式变更，按上述 SemVer 规则发布。
- ai-core 是源码 Go module，不维护应用部署或制品发布流水线；通过不可移动的 SemVer tag 发布。

## 消费方升级建议

- 固定明确版本，阅读 CHANGELOG 后再升级。
- 在消费项目自身的 CI 中运行编译、测试与必要的集成回归；不要依赖任意具名下游项目替代自身验证。
