# 兼容性与稳定性策略 — ai-core

ai-core 是 Hexagon 生态的**共享底座**，被多个独立产品依赖（toolkit ← ai-core/hexagon/hexclaw；ai-core ← hexagon/hexclaw）。底座的接口稳定性直接决定上游能否安心 pin 版本、避免 lockstep。

## SemVer 承诺
- 遵循 [SemVer](https://semver.org/lang/zh-CN/)。**导出标识符（公开 API）**是兼容性契约。
- **patch / minor 不得破坏导出 API**（仅加法）；破坏式变更只能在 **major**（v0.x 阶段为 minor，且需在 CHANGELOG 显著标注 BREAKING）。
- 内部包（`internal/`）、未导出标识符、`examples/` 不在契约内。

## 新增底座 API
- `catalog`、`transport`、`llm/compatible` 及 `media/*` 中导出的 Provider 构造器、Option、请求/响应结构体均属于公开 API，后续按 SemVer 维护。
- `llm.ProviderError` / `llm.NetworkPolicy` / `llm.ErrNetworkPolicy` 是 `transport` 同名类型和值的别名；调用方可继续使用 `llm` 路径，也可逐步迁移到 `transport` 路径，`errors.As` / `errors.Is` 在两种路径间等价。
- `catalog.Capability` 是上层做模型选择、运营展示和 conformance 校验的稳定数据契约；新增字段只做加法，已有字段语义不得在 patch / minor 中收窄。
- Network policy 默认不启用，以保持本地网关、测试服务和私有兼容网关的向后兼容；显式传入零值 `NetworkPolicy` 时启用 public HTTPS 策略。

## 自动门禁
1. **API 兼容性检测**：`.github/workflows/api-compat.yml` 用 `gorelease` 对照上一 tag 检测破坏式变更，提示版本号应如何升。
2. **下游接缝契约**：`.github/workflows/downstream.yml` 在 go.work 下用本仓改动跑全部直接消费者的 build+test —— 下游绿才算接口未破。

## 弃用流程
- 弃用先标 `// Deprecated: 用 X 替代。将在 vN 移除。`，保留 ≥1 个 minor 周期，CHANGELOG 记录，到期才删。
- 移除导出 API = major（v0.x 为 minor + BREAKING 标注）。

## 升级建议（给上游 hexagon / hexclaw）
- pin 明确版本；底座 minor/patch 可放心升；见到 BREAKING 标注再评估迁移。
- 本仓 CI 已保证"改动 → 下游全绿"，故底座的非破坏式演进对上游透明。
