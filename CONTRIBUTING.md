# 贡献指南 — ai-core

ai-core 是面向任意 Go 应用与 Agent 框架的开源 AI 基础能力库，提供 LLM Provider、Tool、Memory、Schema、streamx、template、tokenizer、meter、media、store/vector 与 gateway/llmcall 等通用能力。

## 分层边界

- ai-core 只依赖 `go.mod` 中明确声明的通用底层库，不得依赖任何具体消费项目或上层框架。
- 通用工具（重试、HTTP、缓存、ID、哈希等）统一复用 `toolkit`，不得重复实现。
- 本仓只放 AI 基础抽象与 Provider 实现；业务流程和 Agent 编排由调用方或上层框架负责。

## 本地开发

```bash
test -z "$(gofmt -l .)"
go mod tidy -diff
go vet -mod=readonly ./...
go test -mod=readonly -count=1 ./...
go test -mod=readonly -race -count=1 ./...
BASE_SHA="$(git merge-base HEAD origin/main)"
golangci-lint run --timeout=5m --new-from-rev="$BASE_SHA"
govulncheck ./...
```

上述示例以本地 `origin/main` 的共同祖先为 lint 基线；提交前请先更新远端引用。CI 会直接使用 PR 目标分支或 push 前的精确提交检查新增问题，不会用 `HEAD~1` 假设一次只提交一个 commit，也不会关闭 linter 来隐藏历史问题。

## 提交规范

- Conventional Commits：`feat(llm/openai): ...` / `fix(streamx): ...` / `chore: ...`
- 注释使用中文并只描述功能；返回错误、状态提示和提示语使用英文。
- 新增 Provider 必须包含编译期接口断言（`var _ llm.Provider = (*X)(nil)`）与往返测试。

## PR Checklist

- [ ] `gofmt`、`go mod tidy -diff` 与 `go vet` 全绿
- [ ] 普通测试和 race 测试全绿
- [ ] 新增或修改代码没有 lint 问题
- [ ] 公开 API 有 GoDoc 注释，并通过 API 兼容检查
- [ ] 未引入对具体消费项目或上层框架的依赖
- [ ] 新 Provider 有接口断言与测试
- [ ] 用户可见变更已记录到 CHANGELOG.md
