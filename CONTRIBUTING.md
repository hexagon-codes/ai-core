# 贡献指南 — ai-core

ai-core 是面向任意 Go 应用与 Agent 框架的开源 AI 基础能力库，提供 LLM Provider、Tool、Memory、Schema、streamx、template、tokenizer、meter、media、store/vector 与 gateway/llmcall 等通用能力。

## 分层边界

- ai-core 只依赖 `go.mod` 中明确声明的通用底层库，不得依赖任何具体消费项目或上层框架。
- 通用工具（重试、HTTP、缓存、ID、哈希等）统一复用 `toolkit`，不得重复实现。
- 本仓只放 AI 基础抽象与 Provider 实现；业务流程和 Agent 编排由调用方或上层框架负责。

## 本地开发

提交前在最终工作区运行下面的 Verify 命令；这组检查与普通 PR 上的 `CI / Verify` 保持一致。先从 `go.mod` 选择工具链，再同时固定 `go` 和 `gofmt`，避免本机较新 Go 的行为掩盖最低支持版本的问题：

```bash
set -euo pipefail
export GOWORK=off
export GOTOOLCHAIN="go$(awk '$1 == "go" { print $2; exit }' go.mod)"
ci_goroot="$(go env GOROOT)"
export PATH="$ci_goroot/bin:$PATH"
export GOTOOLCHAIN=local
go version

test -z "$(gofmt -l .)"
go mod tidy -diff
go vet -mod=readonly ./...
golangci-lint run --timeout=5m --enable-only=bodyclose,errcheck,ineffassign,staticcheck,unused --new-from-rev="${CI_BASE_REV:-origin/main}"
go test -mod=readonly -count=1 ./...
```

本地使用 `.github/workflows/ci.yml` 指定版本的 `golangci-lint`。提交前更新 `origin/main` 并固定检查基线；修复已推送的失败提交时，设置 `CI_BASE_REV` 为该批变更前的成功提交，不能以已经包含待验代码的 `HEAD` 作为基线而漏查。普通 Verify 只检查本次变更新增的 lint 问题；代码提交不需要修改 CI 配置。

执行 `go get` 或其他依赖更新后，先用同一工具链运行 `go mod tidy`，再审查 `go.mod` / `go.sum` 差异。`go get` 可能留下旧版本及当前导入链不使用的传递依赖校验和；仅构建成功或模块可下载不能替代 `go mod tidy -diff`。五项 Verify 均须针对同一最终工作区通过，不能用一次包级测试或旧提交的 CI 成功代替。

`Security Audit` 负责依赖和可达代码的漏洞扫描，按工作流的定时计划或手动触发；它不是普通 PR 的 required check。`Release Preflight` 只在发布前手动触发，负责发布所需的 API 兼容性、漏洞和版本检查；通过后再创建不可移动的 SemVer tag。两者失败时都应修复对应的安全或发布问题，但不把外部漏洞数据库变化转化为普通代码提交失败。

Pull Request 和 `main` push 都运行同一套 `CI / Verify`，不维护两套不同的普通检查。

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
