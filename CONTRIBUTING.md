# 贡献指南 — ai-core

ai-core 是 Hexagon 生态的 **L1 AI 基础能力库**（LLM Provider / Tool / Memory / Schema / streamx / template / tokenizer / meter / media / store/vector）。被 hexagon/hexclaw 依赖。

## 分层铁律
- ai-core 仅可依赖 **toolkit (L0)**，**不得依赖** hexagon / hexclaw。
- 通用工具（重试/HTTP/缓存/ID/哈希等）一律复用 toolkit，禁止重造轮子。
- 仅放 AI 基础抽象与 Provider 实现；Agent 编排属 hexagon。

## 本地开发
```bash
go build ./...
go vet ./...
go test -race ./...
golangci-lint run        # 配置见 .golangci.yml
govulncheck ./...
```

## 提交规范
- Conventional Commits：`feat(llm/openai): ...` / `fix(streamx): ...` / `chore: ...`
- 注释中文、只写功能描述，禁暴露内部开发文档/对标框架/调研出处。
- 新增 Provider 必须实现编译期接口断言（`var _ llm.Provider = (*X)(nil)`）+ 往返测试。

## PR Checklist
- [ ] `go test -race ./...` 全绿
- [ ] `golangci-lint run` 0 issue
- [ ] 公开 API 有 GoDoc 注释
- [ ] 仅依赖 toolkit，未引入对 hexagon/hexclaw 的依赖
- [ ] 新 Provider 有接口断言 + 测试
- [ ] CHANGELOG.md 记录用户可见变更
