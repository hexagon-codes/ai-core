package qwen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"

	"github.com/hexagon-codes/ai-core/llm"
)

// 默认 Embedding 模型
const defaultEmbeddingModel = "text-embedding-v3"

// EmbeddingDimension 返回指定 Embedding 模型的默认向量维度
//
// 支持的模型:
//   - text-embedding-v3: 1024（默认）
//   - text-embedding-v2: 1536
//   - text-embedding-v1: 1536
//
// 未知模型默认返回 1024
func EmbeddingDimension(model string) int {
	switch model {
	case "text-embedding-v2", "text-embedding-v1":
		return 1536
	default:
		return 1024
	}
}

// Embed 使用默认模型生成文本的向量嵌入
//
// 默认使用 text-embedding-v3 模型，维度 1024。
// 通义千问的 Embedding API 兼容 OpenAI 格式。
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
// 通义千问的 Embedding API 使用 OpenAI 兼容格式（/v1/embeddings 端点）。
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

	payload := map[string]any{
		"model": model,
		"input": texts,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("序列化 embedding 请求失败: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	p.setHeaders(httpReq)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("embedding 请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("qwen embedding api error: %s (failed to read body: %v)", resp.Status, readErr)
		}
		return nil, fmt.Errorf("qwen embedding api error: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析 embedding 响应失败: %w", err)
	}

	// 校验返回向量数与输入 texts 数一致
	// 调用方依赖向量与文本一一对应（embeddings[i] 对应 texts[i]），
	// 若数量不等则一一对应假设被破坏，必须显式报错而非静默返回错位结果。
	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("qwen embedding 响应向量数(%d)与输入 texts 数(%d)不一致", len(result.Data), len(texts))
	}

	// 按 index 排序确保顺序正确
	sort.Slice(result.Data, func(i, j int) bool {
		return result.Data[i].Index < result.Data[j].Index
	})

	embeddings := make([][]float32, len(result.Data))
	for i, item := range result.Data {
		embeddings[i] = item.Embedding
	}

	return embeddings, nil
}

// embeddingResponse OpenAI 兼容 Embedding API 响应结构
type embeddingResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// 确保实现了 EmbeddingProvider 接口
var _ llm.EmbeddingProvider = (*Provider)(nil)
