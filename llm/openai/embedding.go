package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/hexagon-codes/ai-core/llm"
)

// 默认 Embedding 模型和维度
const (
	defaultEmbeddingModel = "text-embedding-3-small"
)

// EmbeddingDimension 返回指定 Embedding 模型的默认向量维度
//
// 支持的模型:
//   - text-embedding-3-small: 1536
//   - text-embedding-3-large: 3072
//   - text-embedding-ada-002: 1536
//
// 未知模型默认返回 1536
func EmbeddingDimension(model string) int {
	switch model {
	case "text-embedding-3-large":
		return 3072
	default:
		return 1536
	}
}

// Embed 使用默认模型生成文本的向量嵌入
//
// 默认使用 text-embedding-3-small 模型，维度 1536。
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
// 支持的模型:
//   - text-embedding-3-small (1536 维，推荐)
//   - text-embedding-3-large (3072 维，更高质量)
//   - text-embedding-ada-002 (1536 维，旧版)
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
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// 构建请求体
	payload := embeddingRequest{
		Model: model,
		Input: texts,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("序列化 embedding 请求失败: %w", err)
	}

	result, err := doJSONWithDecodeRetry[embeddingResponse](ctx, p, "embedding", "/embeddings", body)
	if err != nil {
		var decodeErr *responseDecodeError
		if errors.As(err, &decodeErr) {
			return nil, fmt.Errorf("解析 embedding 响应失败: %w", decodeErr.Err)
		}
		return nil, fmt.Errorf("embedding 请求失败: %w", err)
	}

	// API 返回结果可能乱序，按 index 排序
	sort.Slice(result.Data, func(i, j int) bool {
		return result.Data[i].Index < result.Data[j].Index
	})

	embeddings := make([][]float32, len(result.Data))
	for i, item := range result.Data {
		embeddings[i] = item.Embedding
	}

	return embeddings, nil
}

// embeddingRequest OpenAI Embedding API 请求结构
type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// embeddingResponse OpenAI Embedding API 响应结构
type embeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// 确保 OpenAI Provider 实现了 EmbeddingProvider 接口
var _ llm.EmbeddingProvider = (*Provider)(nil)
