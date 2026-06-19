# 兼容性与稳定性策略 — ai-core

ai-core 是 Hexagon 生态的**共享底座**，被多个独立产品依赖（toolkit ← ai-core/hexagon/hexclaw；ai-core ← hexagon/hexclaw）。底座的接口稳定性直接决定上游能否安心 pin 版本、避免 lockstep。

## SemVer 承诺
- 遵循 [SemVer](https://semver.org/lang/zh-CN/)。**导出标识符（公开 API）**是兼容性契约。
- **patch / minor 不得破坏导出 API**（仅加法）；破坏式变更只能在 **major**（v0.x 阶段为 minor，且需在 CHANGELOG 显著标注 BREAKING）。
- 内部包（`internal/`）、未导出标识符、`examples/` 不在契约内。

## 自动门禁
1. **API 兼容性检测**：`.github/workflows/api-compat.yml` 用 `gorelease` 对照上一 tag 检测破坏式变更，提示版本号应如何升。
2. **下游接缝契约**：`.github/workflows/downstream.yml` 在 go.work 下用本仓改动跑全部直接消费者的 build+test —— 下游绿才算接口未破。

## 弃用流程
- 弃用先标 `// Deprecated: 用 X 替代。将在 vN 移除。`，保留 ≥1 个 minor 周期，CHANGELOG 记录，到期才删。
- 移除导出 API = major（v0.x 为 minor + BREAKING 标注）。

## 升级建议（给上游 hexagon / hexclaw）
- pin 明确版本；底座 minor/patch 可放心升；见到 BREAKING 标注再评估迁移。
- 本仓 CI 已保证"改动 → 下游全绿"，故底座的非破坏式演进对上游透明。
