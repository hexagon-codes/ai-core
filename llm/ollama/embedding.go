package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/hexagon-codes/ai-core/llm"
)

// 默认 Embedding 模型
const defaultEmbeddingModel = "nomic-embed-text"

// Embed 使用默认模型生成文本的向量嵌入
//
// 默认使用 nomic-embed-text 模型。
// 使用前需确保模型已通过 ollama pull 拉取到本地。
//
// 参数:
//   - ctx: 上下文
//   - texts: 要生成嵌入的文本列表
//
// 返回:
//   - [][]float32: 向量列表，与输入文本一一对应
//   - error: 生成失败时返回错误
func (p *Provider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return p.EmbedWithModel(ctx, defaultEmbeddingModel, texts)
}

// EmbedWithModel 使用指定模型生成文本的向量嵌入
//
// Ollama 使用 /api/embed 端点。
//
// 常用 Embedding 模型:
//   - nomic-embed-text (768 维)
//   - mxbai-embed-large (1024 维)
//   - all-minilm (384 维)
//   - snowflake-arctic-embed (1024 维)
//
// 参数:
//   - ctx: 上下文
//   - model: 模型名称
//   - texts: 要生成嵌入的文本列表
//
// 返回:
//   - [][]float32: 向量列表，与输入文本一一对应
//   - error: 生成失败时返回错误
func (p *Provider) EmbedWithModel(ctx context.Context, model string, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Ollama /api/embed 支持批量输入
	payload := map[string]any{
		"model": model,
		"input": texts,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("序列化 embedding 请求失败: %w", err)
	}

	resp, err := p.doRequest(ctx, http.MethodPost, p.baseURL+"/api/embed", body, false)
	if err != nil {
		return nil, fmt.Errorf("ollama embedding 请求失败: %w", err)
	}
	defer resp.Body.Close()

	var result ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析 embedding 响应失败: %w", err)
	}

	// 转换 [][]float64 → [][]float32
	embeddings := make([][]float32, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		vec := make([]float32, len(emb))
		for j, v := range emb {
			vec[j] = float32(v)
		}
		embeddings[i] = vec
	}

	return embeddings, nil
}

// ollamaEmbedResponse Ollama /api/embed 响应结构
//
// Ollama 返回 float64 类型的向量，需要转换为 float32
type ollamaEmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}

// 确保实现了 EmbeddingProvider 接口
var _ llm.EmbeddingProvider = (*Provider)(nil)
