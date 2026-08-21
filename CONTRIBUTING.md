# 贡献指南 — ai-core

ai-core 是面向任意 Go 应用与 Agent 框架的开源 AI 基础能力库，提供 LLM Provider、Tool、Memory、Schema、streamx、template、tokenizer、meter、media、store/vector 与 gateway/llmcall 等通用能力。

## 分层边界

- ai-core 只依赖 `go.mod` 中明确声明的通用底层库，不得依赖任何具体消费项目或上层框架。
- 通用工具（重试、HTTP、缓存、ID、哈希等）统一复用 `toolkit`，不得重复实现。
- 本仓只放 AI 基础抽象与 Provider 实现；业务流程和 Agent 编排由调用方或上层框架负责。

## 本地开发

提交前运行下面的 Verify 命令；这组检查与普通 PR 上的 `CI / Verify` 保持一致：

```bash
test -z "$(gofmt -l .)"
go mod tidy -diff
go vet -mod=readonly ./...
go test -mod=readonly -count=1 ./...
golangci-lint run --timeout=5m
```

本地使用与 CI 固定版本一致的 `golangci-lint`。普通 Verify 不依赖远端分支、提交数量、漏洞数据库状态或历史发布 tag，因此代码提交不需要修改 CI 配置。

`Security Audit` 负责依赖和可达代码的漏洞扫描，按工作流的定时计划或手动触发；它不是普通 PR 的 required check。`Release Preflight` 只在发布前手动触发，负责发布所需的 API 兼容性、漏洞和版本检查；通过后再创建不可移动的 SemVer tag。两者失败时都应修复对应的安全或发布问题，但不把外部漏洞数据库变化转化为普通代码提交失败。

主干只接受 Pull Request 合并，禁止直接 push；PR 必须通过 `CI / Verify` 后才能合并。合并到 `main` 不重复执行另一套普通检查。

## 提交规范

- Conventional Commits：`feat(llm/openai): ...` / `fix(streamx): ...` / `chore: ...`
- 注释使用中文并只描述功能；返回错误、状态提示和提示语使用英文。
- 新增 Provider 必须包含编译期接口断言（`var _ llm.Provider = (*X)(nil)`）与往返测试。

## PR Checklist

- [ ] `gofmt`、`go mod tidy -diff` 与 `go vet` 全绿
- [ ] 普通测试全绿
- [ ] 新增或修改代码没有 lint 问题
- [ ] 公开 API 有 GoDoc 注释；兼容性检查在发布前由 `Release Preflight` 执行
- [ ] 未引入对具体消费项目或上层框架的依赖
- [ ] 新 Provider 有接口断言与测试
- [ ] 用户可见变更已记录到 CHANGELOG.md
