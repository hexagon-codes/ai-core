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
- `schema.Schema.MinimumNumber` / `MaximumNumber` 是非空时优先的精确 `json.Number` 表示；`Minimum` / `Maximum` 保留为兼容的近似 `float64`。结构化输出和 MCP 往返应以精确字段为准，不能把大整数边界或枚举重新量化为 `float64`。
- `schema.Schema` 的 `Enum`、`AnyOf`、`OneOf`、`AllOf` 使用非 nil 切片表示“已声明”，空数组也是约束而不是未声明；序列化不得用 `omitempty` 删除。空 `enum` 拒绝所有实例，空 `anyOf` 保持非法 Schema，MCP 适配器不得把二者净化为宽松 Schema。
- `streamx.Buffer`、`streamx.Timeout` 与 `streamx.Copy` 对 Merge 的 `SourceEOF` 语义一致：它只标记单个源结束，后续源仍须继续读取；真正 EOF、真实错误和主动 Close 才结束整个流。

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

- `schema.Of` 对 `json:",string"` 整数字段保留 Go 整数语法：小数和指数形式不作为整数接受；无符号整数字符串不接受负号。该约束同样作用于组合及引用 Schema，显式 `pattern` 继续生效；整数指针以 string/null 组合表示，描述、默认值和字段约束保留在属性层，显式 `enum` 未包含 null 时仍拒绝 null。非空值继续受整数语法和数值标签约束。Go 允许的整数字符串前导零不会被误当成非法裸 JSON 数字。

## 2026-11-04 Schema 与流修复

- `json:",string"` 数值字段带 `min`/`max` 时，`Schema.Of` 生成显式 `x-hexagon-string-number` 标记；Validator 只对带标记的字符串执行数值边界检查。普通 `type:string` 上的 `minimum`/`maximum` 保持 JSON Schema 注解语义，不再被误当字符串内数值约束。
- 字符串数值约束通过编译器原生 vocabulary 扩展执行，与 `anyOf`/`oneOf`/`allOf`、条件及引用共同决议；`oneOf` 按包括数值约束在内的结果计算唯一匹配分支。文档内 `$id` 资源、相对引用、锚点及 `$dynamicRef` 由同一编译器处理，不另行重写引用或跳过无法独立编译的分支。signed 整数和带边界的浮点字符串遵循最低 Go 1.26.10 的 `json:",string"` 解码规则：接受合法前导零，但拒绝前导加号；数值比较仍保留原始字面精度。普通标准 Schema 的字符串不因 `minimum`/`maximum` 被改成数值字符串。
- `Copy` 的所有分支在源 EOF 时携带同一源关闭错误；`Collect` 不再把 `EOF + 清理失败` 当正常结束。`ForEach`/`Reduce` 在 ctx 取消时主动关闭源并返回取消原因，不无限阻塞，也不把取消伪装成 nil 成功。

## 精确Schema与合流适配

- `Schema.Of`识别标准JSON表示：`json.Number`为number，`json.RawMessage`为任意JSON，字节切片为base64字符串或null；`json:",string"`的基本类型字段及enum/default按实际JSON字符串表示生成。contentEncoding是标准Schema注解，不等于自动启用内容断言；任意自定义Marshaler不自动推断。需要更窄的供应商原生约束时由调用者显式提供Schema。
- 越界的integer/number enum/default标签保留为`json.Number`，不截断、不改变为字符串，也不把配置值钳制为Go类型上限。Schema数字表示和解码到具体Go类型的范围检查仍是两个合同；无效类型范围由实际Go解码报错。
- Backpressure、Debounce与Buffer/Timeout采用一致的SourceEOF语义；Debounce在源完成时立即刷新最后值，真实上游错误在已产生的值之后返回。消费者Close释放源和内部等待，不用正常EOF掩盖失败。
- `PipeWithClose`的回调仍只调用一次；panic转换为含栈的`ClosePanicError`并保留原始error因果链。`Collect`返回已读取的值并合并关闭错误，该类清理失败不可触发整次读取的自动重放；Merge/Zip保留所有输入的关闭错误。Close回调仍不得等待当前读取回调自身结束。
- Pipe支持Send与写端Close并发：关闭会唤醒阻塞发送，已缓冲值仍可读完；首次Close/CloseWithError确定终态，后续关闭不覆盖原错误，CloseWithError(nil)等同Close。Zip在右源发出非终态SourceEOF时保留未配对的左值，下一次继续读取右源。
- 库创建的Copy、Buffer、Timeout、Backpressure、Debounce后台读取通过`SourcePanicError`返回用户读取/映射/观察回调的异常，不让panic逃逸为进程退出。该错误保留Value/Stack与原error因果链并标记不可重放；调用者同步执行的普通回调仍遵守对应API自身合同。

- `schema.Of` 的静态数值 min/max 标签写入精确 Number 字段；超出本机 int 范围但可表示为 uint64 的 integer enum/default 保留数字字面量，不退回字符串。`time.Time` 及其指针生成 `string`、`format: date-time`，与 Go JSON 表示一致；自定义 MarshalJSON 类型不据此获得自动推断承诺，format 验证仍依赖 Validator 配置。

- `Schema.MinimumNumber` / `MaximumNumber` 为可选 `json.Number`，非空时优先于既有 `*float64` 字段；JSON解码保留原始数值字面量，同时为可表示的旧字段提供近似读数。需要改写下界/上界时同步修改或清空相应Number字段，不能只改近似字段。JSON线上仍为数字。
- `Enum`、`Default`及额外关键字的动态数字解码为 `json.Number`。消费者按数值范围调用Int64/Float64/String并处理错误，不把大整数强转float64作为身份。非nil空Enum/AnyOf/OneOf/AllOf保留声明存在性；Validator继续按原始Schema判断，不把空约束删除为无限制。
- `Buffer(Merge(...))`和`Timeout(Merge(...))`不会在单个SourceEOF处结束整体流。Buffer消费非终态标记并继续；Timeout按顺序传递标记及后续数据。真正EOF、源错误与主动Close语义不变。

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
