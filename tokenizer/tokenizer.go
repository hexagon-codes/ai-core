package tokenizer

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Model 定义支持的 AI 模型类型
// 不同模型有不同的分词规则和上下文限制
type Model string

const (
	// GPT4 OpenAI GPT-4 及 GPT-4 Turbo 系列
	// 使用 cl100k_base 编码，上下文窗口 8K-128K
	GPT4 Model = "gpt-4"

	// GPT4o OpenAI GPT-4o 系列（优化版本）
	// 使用 o200k_base 编码，支持更长上下文和更快响应
	GPT4o Model = "gpt-4o"

	// GPT35Turbo OpenAI GPT-3.5 Turbo 系列
	// 使用 cl100k_base 编码，上下文窗口 4K-16K
	GPT35Turbo Model = "gpt-3.5-turbo"

	// Claude3 Anthropic Claude 3 系列（Opus/Sonnet/Haiku）
	// 使用自研分词器，上下文窗口 200K
	Claude3 Model = "claude-3"

	// Gemini Google Gemini 系列
	// 使用 SentencePiece 分词，上下文窗口最高 2M
	Gemini Model = "gemini"

	// DeepSeek DeepSeek 系列
	// 国产大模型，上下文窗口 64K
	DeepSeek Model = "deepseek"

	// Qwen 阿里通义千问系列
	// 国产大模型，支持中文优化
	Qwen Model = "qwen"
)

// Message 表示聊天消息结构
// 与 OpenAI Chat API 的消息格式兼容
type Message struct {
	// Role 消息角色：system（系统提示）、user（用户）、assistant（助手）
	Role string `json:"role"`
	// Content 消息内容
	Content string `json:"content"`
	// Name 可选的发送者名称，用于多用户场景
	Name string `json:"name,omitempty"`
}

// Counter 是 Token 计数器
// 根据模型类型使用不同的估算参数
// 提供快速的 Token 数量估算，用于成本预估和限制检查
type Counter struct {
	model            Model   // 目标模型
	tokensPerMessage int     // 每条消息的固定开销 Token 数
	tokensPerName    int     // 消息 Name 字段的额外 Token 数
	charsPerToken    float64 // 平均每个 Token 对应的字符数（用于英文估算）
}

// New 创建指定模型的 Token 计数器
// 根据模型类型初始化不同的估算参数
//
// 示例：
//
//	counter := tokenizer.New(tokenizer.GPT4)
//	count := counter.Count("Hello, world!")
func New(model Model) *Counter {
	c := &Counter{
		model: model,
	}

	// 根据模型设置参数
	switch model {
	case GPT4, GPT4o:
		c.tokensPerMessage = 3
		c.tokensPerName = 1
		c.charsPerToken = 4.0
	case GPT35Turbo:
		c.tokensPerMessage = 4
		c.tokensPerName = -1
		c.charsPerToken = 4.0
	case Claude3:
		c.tokensPerMessage = 3
		c.tokensPerName = 1
		c.charsPerToken = 4.0
	case Gemini:
		c.tokensPerMessage = 3
		c.tokensPerName = 1
		c.charsPerToken = 4.0
	default:
		c.tokensPerMessage = 3
		c.tokensPerName = 1
		c.charsPerToken = 4.0
	}

	return c
}

// Count 计算文本的 Token 数（快速估算）
// 使用混合策略估算：
//   - 英文/数字：约 4 个字符一个 Token
//   - 中文：约 1.5 个字符一个 Token
//   - 特殊字符：约 2 个字符一个 Token
//
// 注意：这是估算值，实际值可能有 ±10% 的误差
func (c *Counter) Count(text string) int {
	if text == "" {
		return 0
	}

	return c.countTokens(text)
}

// CountMessages 计算消息列表的 Token 数
// 除了消息内容外，还计算以下开销：
//   - 每条消息的固定开销（tokensPerMessage）
//   - 消息角色字段的 Token 数
//   - 如有 Name 字段，额外的 Token 数
//   - 回复前导的 3 个 Token（OpenAI 规范）
func (c *Counter) CountMessages(messages []Message) int {
	total := 0

	for _, msg := range messages {
		total += c.tokensPerMessage
		total += c.Count(msg.Role)
		total += c.Count(msg.Content)
		if msg.Name != "" {
			total += c.Count(msg.Name)
			total += c.tokensPerName
		}
	}

	// 每个回复都有一个 "assistant" 前导
	total += 3

	return total
}

// CountPrompt 计算提示词和生成内容的总 Token 数
// 用于预估完整对话的 Token 消耗
func (c *Counter) CountPrompt(prompt, completion string) int {
	return c.Count(prompt) + c.Count(completion)
}

// countTokens 是内部的 Token 计数核心逻辑
// 采用混合策略处理不同类型的字符：
//   - 英文单词和数字：按平均 4 字符/Token 估算
//   - 中文字符：按平均 1.5 字符/Token 估算（中文分词粒度较小）
//   - 特殊标点：按平均 2 字符/Token 估算
func (c *Counter) countTokens(text string) int {

	var tokens float64
	var currentWord strings.Builder
	var chineseCount int
	var englishChars int
	var specialCount int

	for _, r := range text {
		if isChinese(r) {
			// 先处理之前的英文词
			if currentWord.Len() > 0 {
				englishChars += currentWord.Len()
				currentWord.Reset()
			}
			chineseCount++
		} else if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentWord.WriteRune(r)
		} else if unicode.IsSpace(r) {
			if currentWord.Len() > 0 {
				englishChars += currentWord.Len()
				currentWord.Reset()
			}
		} else {
			// 特殊字符
			if currentWord.Len() > 0 {
				englishChars += currentWord.Len()
				currentWord.Reset()
			}
			specialCount++
		}
	}

	// 处理最后的词
	if currentWord.Len() > 0 {
		englishChars += currentWord.Len()
	}

	// 计算 token 数
	// 英文：约 4 字符/token
	tokens += float64(englishChars) / c.charsPerToken
	// 中文：约 1.5 字符/token（更保守的估算）
	tokens += float64(chineseCount) / 1.5
	// 特殊字符：约 2 字符/token
	tokens += float64(specialCount) / 2.0

	// 向上取整
	result := int(tokens + 0.5)
	if result < 1 && (chineseCount > 0 || englishChars > 0 || specialCount > 0) {
		result = 1
	}

	return result
}

// isChinese 判断字符是否为中文（汉字）
// 使用 Unicode Han 范围判断
func isChinese(r rune) bool {
	return unicode.Is(unicode.Han, r)
}

// ============== 便捷函数 ==============
// 以下函数提供快捷方式，无需显式创建 Counter 实例

// 预创建的计数器缓存，避免重复创建
var (
	gpt4Counter   = New(GPT4)
	claudeCounter = New(Claude3)
	geminiCounter = New(Gemini)
)

// CountGPT4 使用 GPT-4 模型参数计算 Token 数
// 适用于 GPT-4、GPT-4 Turbo 等模型
func CountGPT4(text string) int {
	return gpt4Counter.Count(text)
}

// CountClaude 使用 Claude 模型参数计算 Token 数
// 适用于 Claude 3 系列模型
func CountClaude(text string) int {
	return claudeCounter.Count(text)
}

// CountGemini 使用 Gemini 模型参数计算 Token 数
// 适用于 Google Gemini 系列模型
func CountGemini(text string) int {
	return geminiCounter.Count(text)
}

// ============== 模型上下文限制 ==============

// ContextLimit 定义模型的上下文窗口限制
// 用于在发送请求前检查是否超过模型限制
type ContextLimit struct {
	Model           Model // 模型类型
	MaxInputTokens  int   // 最大输入 Token 数
	MaxOutputTokens int   // 最大输出 Token 数
	MaxTotalTokens  int   // 最大总 Token 数（输入+输出）
}

// 预定义的主流模型上下文限制
// 数据来源：各厂商官方文档（2024年数据）
var (
	// GPT4Limit GPT-4 Turbo 的上下文限制
	GPT4Limit = ContextLimit{
		Model:           GPT4,
		MaxInputTokens:  128000,
		MaxOutputTokens: 4096,
		MaxTotalTokens:  128000,
	}

	GPT4oLimit = ContextLimit{
		Model:           GPT4o,
		MaxInputTokens:  128000,
		MaxOutputTokens: 16384,
		MaxTotalTokens:  128000,
	}

	GPT35TurboLimit = ContextLimit{
		Model:           GPT35Turbo,
		MaxInputTokens:  16385,
		MaxOutputTokens: 4096,
		MaxTotalTokens:  16385,
	}

	Claude3OpusLimit = ContextLimit{
		Model:           Claude3,
		MaxInputTokens:  200000,
		MaxOutputTokens: 4096,
		MaxTotalTokens:  200000,
	}

	Claude3SonnetLimit = ContextLimit{
		Model:           Claude3,
		MaxInputTokens:  200000,
		MaxOutputTokens: 4096,
		MaxTotalTokens:  200000,
	}

	Gemini15ProLimit = ContextLimit{
		Model:           Gemini,
		MaxInputTokens:  2097152,
		MaxOutputTokens: 8192,
		MaxTotalTokens:  2097152,
	}

	DeepSeekLimit = ContextLimit{
		Model:           DeepSeek,
		MaxInputTokens:  64000,
		MaxOutputTokens: 4096,
		MaxTotalTokens:  64000,
	}
)

// CheckLimit 检查文本是否超过模型的输入限制
// 返回 true 表示未超过限制，可以安全发送
func (c *Counter) CheckLimit(text string, limit ContextLimit) bool {
	tokens := c.Count(text)
	return tokens <= limit.MaxInputTokens
}

// CheckMessagesLimit 检查消息列表是否超过模型的输入限制
// 返回 true 表示未超过限制
func (c *Counter) CheckMessagesLimit(messages []Message, limit ContextLimit) bool {
	tokens := c.CountMessages(messages)
	return tokens <= limit.MaxInputTokens
}

// TruncateToLimit 将文本截断到指定的 Token 数限制内
// 使用二分查找找到最大的合法截断点
// 返回截断后的文本，保证不超过 maxTokens
func (c *Counter) TruncateToLimit(text string, maxTokens int) string {
	tokens := c.Count(text)
	if tokens <= maxTokens {
		return text
	}

	// 二分查找合适的截断点
	runes := []rune(text)
	low, high := 0, len(runes)

	for low < high {
		mid := (low + high + 1) / 2
		if c.Count(string(runes[:mid])) <= maxTokens {
			low = mid
		} else {
			high = mid - 1
		}
	}

	return string(runes[:low])
}

// ============== 成本估算 ==============

// Pricing 定义模型的 Token 价格
// 价格单位：美元/百万 Token
type Pricing struct {
	Model       Model   // 模型类型
	InputPrice  float64 // 输入 Token 价格（$/1M tokens）
	OutputPrice float64 // 输出 Token 价格（$/1M tokens）
}

// 预定义的主流模型定价
// 数据来源：各厂商官方定价页面（2024年数据）
// 注意：价格可能随时调整，请以官方为准
var (
	GPT4Pricing = Pricing{
		Model:       GPT4,
		InputPrice:  30.0,
		OutputPrice: 60.0,
	}

	GPT4oPricing = Pricing{
		Model:       GPT4o,
		InputPrice:  2.5,
		OutputPrice: 10.0,
	}

	GPT35TurboPricing = Pricing{
		Model:       GPT35Turbo,
		InputPrice:  0.5,
		OutputPrice: 1.5,
	}

	Claude3OpusPricing = Pricing{
		Model:       Claude3,
		InputPrice:  15.0,
		OutputPrice: 75.0,
	}

	Claude3SonnetPricing = Pricing{
		Model:       Claude3,
		InputPrice:  3.0,
		OutputPrice: 15.0,
	}
)

// EstimateCost 根据 Token 数量和定价估算调用成本
// 返回值单位为美元
//
// 示例：
//
//	cost := tokenizer.EstimateCost(1000, 500, tokenizer.GPT4Pricing)
//	fmt.Printf("预估成本: $%.4f\n", cost)
func EstimateCost(inputTokens, outputTokens int, pricing Pricing) float64 {
	inputCost := float64(inputTokens) / 1_000_000 * pricing.InputPrice
	outputCost := float64(outputTokens) / 1_000_000 * pricing.OutputPrice
	return inputCost + outputCost
}

// ============== 工具函数 ==============

// SplitByTokens 将长文本按 Token 数限制分割成多个块
// 用于将超长文本分批处理，避免超过模型限制
// 使用二分查找确保每个块不超过 maxTokensPerChunk
func (c *Counter) SplitByTokens(text string, maxTokensPerChunk int) []string {
	if maxTokensPerChunk <= 0 {
		return nil
	}
	if c.Count(text) <= maxTokensPerChunk {
		return []string{text}
	}

	var chunks []string
	runes := []rune(text)
	start := 0

	for start < len(runes) {
		// 找到不超过限制的最大长度
		end := len(runes)
		for c.Count(string(runes[start:end])) > maxTokensPerChunk && end > start {
			end = (start + end) / 2
		}

		// 确保至少有一个字符
		if end == start {
			end = start + 1
		}

		chunks = append(chunks, string(runes[start:end]))
		start = end
	}

	return chunks
}

// CountRunes 返回文本的 Unicode 字符数（rune 数量）
// 与 len() 返回字节数不同，此函数正确处理多字节字符
func CountRunes(text string) int {
	return utf8.RuneCountInString(text)
}

// CountWords 返回文本的单词数
// 使用空白字符分割，适用于英文文本
func CountWords(text string) int {
	return len(strings.Fields(text))
}

// CountChineseChars 统计文本中的中文字符数量
// 只计算 Unicode Han 范围内的字符
func CountChineseChars(text string) int {
	count := 0
	for _, r := range text {
		if isChinese(r) {
			count++
		}
	}
	return count
}
