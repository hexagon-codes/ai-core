// Package router 提供多 Provider LLM 路由功能
//
// 本文件实现任务分类器，用于自动识别用户请求的任务类型和复杂度。
// 提供两种分类器实现：
// - RuleBasedClassifier: 基于规则的分类器，无需 LLM 调用，速度快
// - LLMClassifier: 基于 LLM 的分类器，准确度高但有额外开销
package router

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/hexagon-codes/ai-core/llm"
)

// TaskClassifier 任务分类器接口
// 用于自动识别请求的任务类型和复杂度
type TaskClassifier interface {
	// Classify 分类任务
	// 输入: CompletionRequest
	// 输出: 任务类型和复杂度
	Classify(ctx context.Context, req llm.CompletionRequest) (TaskType, TaskComplexity)

	// Name 返回分类器名称
	Name() string
}

// ============== 基于规则的分类器 ==============

// RuleBasedClassifier 基于规则的分类器
// 使用关键词匹配和正则表达式进行任务分类，无需 LLM 调用
type RuleBasedClassifier struct {
	// rules 分类规则列表
	rules []ClassificationRule

	// defaultTaskType 默认任务类型
	defaultTaskType TaskType

	// defaultComplexity 默认复杂度
	defaultComplexity TaskComplexity
}

// ClassificationRule 分类规则
type ClassificationRule struct {
	// TaskType 匹配后的任务类型
	TaskType TaskType

	// Keywords 关键词列表（任意匹配）
	Keywords []string

	// Patterns 正则表达式列表（任意匹配）
	Patterns []*regexp.Regexp

	// Priority 优先级（数值越大优先级越高）
	Priority int

	// ComplexityHints 复杂度提示
	// 用于根据匹配内容调整复杂度
	ComplexityHints map[string]TaskComplexity
}

// NewRuleBasedClassifier 创建基于规则的分类器
func NewRuleBasedClassifier() *RuleBasedClassifier {
	c := &RuleBasedClassifier{
		defaultTaskType:   TaskTypeChat,
		defaultComplexity: ComplexityMedium,
	}
	c.rules = c.buildDefaultRules()
	return c
}

// Name 返回分类器名称
func (c *RuleBasedClassifier) Name() string {
	return "rule_based"
}

// Classify 分类任务
func (c *RuleBasedClassifier) Classify(ctx context.Context, req llm.CompletionRequest) (TaskType, TaskComplexity) {
	// 提取文本内容
	text := c.extractText(req)
	if requestHasImage(req) {
		return TaskTypeVision, c.estimateComplexity(text)
	}
	if text == "" {
		return c.defaultTaskType, c.defaultComplexity
	}

	// 转换为小写进行匹配
	lowerText := strings.ToLower(text)

	// 按优先级排序规则（这里假设规则已经按优先级排序）
	var matchedRule *ClassificationRule
	for i := range c.rules {
		rule := &c.rules[i]
		if c.matchRule(lowerText, rule) {
			if matchedRule == nil || rule.Priority > matchedRule.Priority {
				matchedRule = rule
			}
		}
	}

	// 如果没有匹配到规则，使用默认值
	if matchedRule == nil {
		return c.defaultTaskType, c.estimateComplexity(text)
	}

	// 估算复杂度
	complexity := c.estimateComplexityWithHints(text, matchedRule)

	return matchedRule.TaskType, complexity
}

// extractText 从请求中提取用户输入文本；system prompt 不参与任务分类，
// 避免长系统提示词把普通请求误判为复杂编程任务。
func (c *RuleBasedClassifier) extractText(req llm.CompletionRequest) string {
	var texts []string
	for _, msg := range req.Messages {
		if msg.Role != llm.RoleUser {
			continue
		}
		texts = append(texts, messageText(msg))
	}
	return strings.Join(texts, " ")
}

func requestHasImage(req llm.CompletionRequest) bool {
	for _, msg := range req.Messages {
		for _, part := range msg.MultiContent {
			if part.Type == "image_url" {
				return true
			}
		}
	}
	return false
}

func messageText(msg llm.Message) string {
	if len(msg.MultiContent) == 0 {
		return msg.Content
	}
	var parts []string
	for _, part := range msg.MultiContent {
		if part.Type == "text" || part.Type == "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, " ")
}

// matchRule 检查文本是否匹配规则
func (c *RuleBasedClassifier) matchRule(text string, rule *ClassificationRule) bool {
	// 检查关键词
	for _, keyword := range rule.Keywords {
		if strings.Contains(text, strings.ToLower(keyword)) {
			return true
		}
	}

	// 检查正则表达式
	for _, pattern := range rule.Patterns {
		if pattern.MatchString(text) {
			return true
		}
	}

	return false
}

// estimateComplexity 根据文本估算复杂度
func (c *RuleBasedClassifier) estimateComplexity(text string) TaskComplexity {
	// 基于字符数的粗略估算，避免中文字符按 UTF-8 字节数被高估。
	length := utf8.RuneCountInString(text)

	// 基于词汇复杂度的调整
	complexWords := c.countComplexIndicators(text)

	// 综合评估
	if length > 2000 || complexWords > 5 {
		return ComplexityExpert
	}
	if length > 1000 || complexWords > 3 {
		return ComplexityComplex
	}
	if length > 300 || complexWords > 1 {
		return ComplexityMedium
	}
	return ComplexitySimple
}

// estimateComplexityWithHints 根据规则提示估算复杂度；命中多个提示时取最高复杂度，
// 避免 map 遍历顺序导致同一文本得到不同结果。
func (c *RuleBasedClassifier) estimateComplexityWithHints(text string, rule *ClassificationRule) TaskComplexity {
	lowerText := strings.ToLower(text)
	best := TaskComplexity("")
	bestRank := -1
	for hint, complexity := range rule.ComplexityHints {
		if !strings.Contains(lowerText, strings.ToLower(hint)) {
			continue
		}
		rank := complexityRank(complexity)
		if rank > bestRank {
			best, bestRank = complexity, rank
		}
	}
	if best != "" {
		return best
	}

	// 使用通用估算
	return c.estimateComplexity(text)
}

func complexityRank(complexity TaskComplexity) int {
	switch complexity {
	case ComplexityExpert:
		return 3
	case ComplexityComplex:
		return 2
	case ComplexityMedium:
		return 1
	case ComplexitySimple:
		return 0
	default:
		return -1
	}
}

// countComplexIndicators 计算复杂度指标词数量
func (c *RuleBasedClassifier) countComplexIndicators(text string) int {
	lowerText := strings.ToLower(text)
	count := 0

	complexIndicators := []string{
		// 复杂任务指示词
		"详细分析", "深入研究", "全面评估", "系统设计",
		"架构", "优化", "重构", "性能调优",
		"深度学习", "机器学习", "算法设计",
		"comprehensive", "in-depth", "thorough",
		"architecture", "optimization", "refactor",
		"algorithm", "distributed", "scalable",
		// 专家级指示词
		"前沿", "创新", "突破性", "研究级",
		"state-of-the-art", "cutting-edge", "novel",
	}

	for _, indicator := range complexIndicators {
		if strings.Contains(lowerText, indicator) {
			count++
		}
	}

	return count
}

// buildDefaultRules 构建默认规则集
func (c *RuleBasedClassifier) buildDefaultRules() []ClassificationRule {
	return []ClassificationRule{
		// ===== 编程任务 =====
		{
			TaskType: TaskTypeCoding,
			Keywords: []string{
				// 中文关键词
				"代码", "编程", "函数", "方法", "类",
				"实现", "bug", "调试", "修复",
				"重构", "优化代码", "代码审查",
				// 英文关键词
				"code", "program", "function", "method", "class",
				"implement", "debug", "fix", "refactor",
				"coding", "programming", "developer",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)write\s+(a\s+)?(code|function|program|script)`),
				regexp.MustCompile(`(?i)(fix|debug|solve)\s+(this\s+)?(bug|error|issue)`),
				regexp.MustCompile(`(?i)(python|javascript|go|java|rust|typescript|c\+\+)`),
				regexp.MustCompile("```[a-z]*\\n"),
			},
			Priority: 10,
			ComplexityHints: map[string]TaskComplexity{
				"简单":           ComplexitySimple,
				"simple":       ComplexitySimple,
				"架构":           ComplexityExpert,
				"architecture": ComplexityExpert,
				"分布式":          ComplexityExpert,
				"distributed":  ComplexityExpert,
			},
		},

		// ===== 推理任务 =====
		{
			TaskType: TaskTypeReasoning,
			Keywords: []string{
				// 中文关键词
				"推理", "分析原因", "为什么", "怎么解释",
				"逻辑", "论证", "推断", "假设",
				// 英文关键词
				"reason", "why", "explain", "logic",
				"argument", "deduce", "infer", "hypothesis",
				"think through", "step by step",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)let'?s\s+think\s+(about\s+)?(this\s+)?step\s+by\s+step`),
				regexp.MustCompile(`(?i)what\s+(is|are)\s+the\s+reason`),
				regexp.MustCompile(`(?i)(analyze|explain)\s+(why|how)`),
			},
			Priority: 8,
			ComplexityHints: map[string]TaskComplexity{
				"复杂":        ComplexityComplex,
				"complex":   ComplexityComplex,
				"深入":        ComplexityExpert,
				"in-depth":  ComplexityExpert,
				"简单解释":      ComplexitySimple,
				"basically": ComplexitySimple,
			},
		},

		// ===== 数学任务 =====
		{
			TaskType: TaskTypeMath,
			Keywords: []string{
				// 中文关键词
				"计算", "数学", "公式", "方程",
				"求解", "证明", "微积分", "线性代数",
				// 英文关键词
				"calculate", "math", "formula", "equation",
				"solve", "prove", "calculus", "algebra",
				"derivative", "integral", "matrix",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)(solve|calculate|compute)\s+(the\s+)?(equation|expression|formula)`),
				regexp.MustCompile(`\d+\s*[\+\-\*\/\^]\s*\d+`),
				regexp.MustCompile(`(?i)(prove|derive|find)\s+(that|the)`),
			},
			Priority: 9,
			ComplexityHints: map[string]TaskComplexity{
				"简单计算":     ComplexitySimple,
				"basic":    ComplexitySimple,
				"高等数学":     ComplexityComplex,
				"advanced": ComplexityComplex,
				"证明":       ComplexityExpert,
				"prove":    ComplexityExpert,
			},
		},

		// ===== 数据分析任务 =====
		{
			TaskType: TaskTypeAnalysis,
			Keywords: []string{
				// 中文关键词
				"分析", "数据", "统计", "趋势",
				"洞察", "报告", "指标", "评估",
				// 英文关键词
				"analyze", "analysis", "data", "statistics",
				"trend", "insight", "report", "metric",
				"evaluate", "assessment",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)analyze\s+(the\s+)?(data|results|performance)`),
				regexp.MustCompile(`(?i)(what|how)\s+(does|do)\s+(the\s+)?data\s+(show|indicate)`),
			},
			Priority: 7,
			ComplexityHints: map[string]TaskComplexity{
				"简单分析":          ComplexitySimple,
				"overview":      ComplexitySimple,
				"深度分析":          ComplexityComplex,
				"deep":          ComplexityComplex,
				"全面评估":          ComplexityExpert,
				"comprehensive": ComplexityExpert,
			},
		},

		// ===== 摘要任务 =====
		{
			TaskType: TaskTypeSummarization,
			Keywords: []string{
				// 中文关键词
				"摘要", "总结", "概括", "提炼",
				"要点", "简述", "归纳",
				// 英文关键词
				"summary", "summarize", "summarise",
				"brief", "overview", "key points",
				"tldr", "tl;dr",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)(summarize|summarise|give\s+(me\s+)?a\s+summary)`),
				regexp.MustCompile(`(?i)what\s+(are\s+)?the\s+(main|key)\s+points`),
				regexp.MustCompile(`(?i)in\s+(brief|short|summary)`),
			},
			Priority: 6,
			ComplexityHints: map[string]TaskComplexity{
				"简短":       ComplexitySimple,
				"brief":    ComplexitySimple,
				"详细摘要":     ComplexityMedium,
				"detailed": ComplexityMedium,
			},
		},

		// ===== 信息提取任务 =====
		{
			TaskType: TaskTypeExtraction,
			Keywords: []string{
				// 中文关键词
				"提取", "抽取", "识别", "解析",
				"获取", "提炼信息", "实体识别",
				// 英文关键词
				"extract", "parse", "identify",
				"recognize", "get", "retrieve",
				"entity", "ner", "extraction",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)extract\s+(the\s+)?(information|data|entities)`),
				regexp.MustCompile(`(?i)(identify|recognize|find)\s+(all\s+)?(the\s+)?(names|dates|entities)`),
				regexp.MustCompile(`(?i)parse\s+(the\s+)?(text|document|json)`),
			},
			Priority: 7,
			ComplexityHints: map[string]TaskComplexity{
				"简单提取":   ComplexitySimple,
				"simple": ComplexitySimple,
				"复杂结构":   ComplexityComplex,
				"nested": ComplexityComplex,
			},
		},

		// ===== 创意写作任务 =====
		{
			TaskType: TaskTypeCreative,
			Keywords: []string{
				// 中文关键词
				"写", "创作", "文章", "故事",
				"诗", "剧本", "文案", "创意",
				"小说", "散文",
				// 英文关键词
				"write", "compose", "create",
				"story", "poem", "script",
				"creative", "fiction", "essay",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)write\s+(a\s+)?(story|poem|essay|article|script)`),
				regexp.MustCompile(`(?i)(compose|create)\s+(a\s+)?(creative|original)`),
				regexp.MustCompile(`(?i)help\s+me\s+write`),
			},
			Priority: 6,
			ComplexityHints: map[string]TaskComplexity{
				"短文":           ComplexitySimple,
				"short":        ComplexitySimple,
				"长篇":           ComplexityComplex,
				"long":         ComplexityComplex,
				"专业":           ComplexityExpert,
				"professional": ComplexityExpert,
			},
		},

		// ===== 翻译任务 =====
		{
			TaskType: TaskTypeTranslation,
			Keywords: []string{
				// 中文关键词
				"翻译", "译成", "转换语言",
				// 英文关键词
				"translate", "translation",
				"into english", "into chinese",
				"to english", "to chinese",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)translate\s+(this\s+)?(text\s+)?(to|into)\s+`),
				regexp.MustCompile(`(?i)(把|将).*(翻译|译)成`),
			},
			Priority: 8,
			ComplexityHints: map[string]TaskComplexity{
				"简单句子":      ComplexitySimple,
				"sentence":  ComplexitySimple,
				"专业文档":      ComplexityComplex,
				"technical": ComplexityComplex,
				"文学翻译":      ComplexityExpert,
				"literary":  ComplexityExpert,
			},
		},

		// ===== 视觉任务 =====
		{
			TaskType: TaskTypeVision,
			Keywords: []string{
				// 中文关键词
				"图片", "图像", "照片", "看图",
				"图表", "截图", "识别图",
				// 英文关键词
				"image", "picture", "photo",
				"chart", "screenshot", "diagram",
				"look at", "describe the image",
			},
			Patterns: []*regexp.Regexp{
				regexp.MustCompile(`(?i)(describe|explain|analyze)\s+(this\s+)?(image|picture|photo|chart)`),
				regexp.MustCompile(`(?i)what\s+(is|does)\s+(in\s+)?(this\s+)?(image|picture)`),
			},
			Priority: 9,
			ComplexityHints: map[string]TaskComplexity{
				"简单描述":     ComplexitySimple,
				"simple":   ComplexitySimple,
				"详细分析":     ComplexityComplex,
				"detailed": ComplexityComplex,
			},
		},
	}
}

// AddRule 添加自定义规则
func (c *RuleBasedClassifier) AddRule(rule ClassificationRule) {
	c.rules = append(c.rules, rule)
}

// SetDefaultTaskType 设置默认任务类型
func (c *RuleBasedClassifier) SetDefaultTaskType(taskType TaskType) {
	c.defaultTaskType = taskType
}

// SetDefaultComplexity 设置默认复杂度
func (c *RuleBasedClassifier) SetDefaultComplexity(complexity TaskComplexity) {
	c.defaultComplexity = complexity
}

// ============== 基于 LLM 的分类器 ==============

// LLMClassifier 基于 LLM 的分类器
// 使用 LLM 进行任务分类，准确度更高但有额外开销
type LLMClassifier struct {
	// llm LLM Provider
	llm llm.Provider

	// model 使用的模型（建议使用小模型以降低成本）
	model string

	// systemPrompt 系统提示词
	systemPrompt string
}

// LLMClassifierOption LLMClassifier 配置选项
type LLMClassifierOption func(*LLMClassifier)

// WithLLMModel 设置 LLM 模型
func WithLLMModel(model string) LLMClassifierOption {
	return func(c *LLMClassifier) {
		c.model = model
	}
}

// WithSystemPrompt 设置系统提示词
func WithSystemPrompt(prompt string) LLMClassifierOption {
	return func(c *LLMClassifier) {
		c.systemPrompt = prompt
	}
}

// NewLLMClassifier 创建基于 LLM 的分类器
func NewLLMClassifier(provider llm.Provider, opts ...LLMClassifierOption) *LLMClassifier {
	c := &LLMClassifier{
		llm:   provider,
		model: "gpt-4o-mini", // 默认使用小模型
		systemPrompt: `你是一个任务分类专家。分析用户的请求，判断任务类型和复杂度。

任务类型（只能选择一个）：
- chat: 普通对话、闲聊
- reasoning: 逻辑推理、问题分析
- coding: 编程、代码相关
- analysis: 数据分析、评估
- summarization: 摘要、总结
- extraction: 信息提取、数据解析
- creative: 创意写作、内容创作
- translation: 翻译
- math: 数学计算、公式推导
- vision: 图像理解、图表分析

复杂度（只能选择一个）：
- simple: 简单任务
- medium: 中等任务
- complex: 复杂任务
- expert: 专家级任务

请只返回 JSON 格式，不要其他文字：
{"task_type": "类型", "complexity": "复杂度"}`,
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// Name 返回分类器名称
func (c *LLMClassifier) Name() string {
	return "llm_based"
}

// Classify 使用 LLM 分类任务
func (c *LLMClassifier) Classify(ctx context.Context, req llm.CompletionRequest) (TaskType, TaskComplexity) {
	// 提取最后一条用户消息；多轮对话按当前请求而不是历史首条分类。
	userText := ""
	for _, msg := range req.Messages {
		if msg.Role == llm.RoleUser {
			userText = messageText(msg)
		}
	}

	if userText == "" {
		return TaskTypeChat, ComplexityMedium
	}

	// 调用 LLM 进行分类
	classifyReq := llm.CompletionRequest{
		Model: c.model,
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: c.systemPrompt},
			{Role: llm.RoleUser, Content: userText},
		},
		MaxTokens:   100,
		Temperature: func() *float64 { t := 0.0; return &t }(),
	}

	resp, err := c.llm.Complete(ctx, classifyReq)
	if err != nil {
		// 分类失败时返回默认值
		return TaskTypeChat, ComplexityMedium
	}

	// 解析响应
	return c.parseResponse(resp.Content)
}

// parseResponse 解析 LLM 响应
func (c *LLMClassifier) parseResponse(content string) (TaskType, TaskComplexity) {
	var parsed struct {
		TaskType   string `json:"task_type"`
		Complexity string `json:"complexity"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &parsed); err == nil {
		taskType := normalizeTaskType(parsed.TaskType)
		complexity := normalizeTaskComplexity(parsed.Complexity)
		if taskType != "" && complexity != "" {
			return taskType, complexity
		}
	}

	// 兼容模型在 JSON 外包了说明文字的情况：只匹配带键名的精确值，
	// 避免 "complexity" 字段名里的 "complex" 被误判为复杂度。
	lower := strings.ToLower(content)
	taskType := TaskTypeChat
	for _, candidate := range []struct {
		key   string
		value TaskType
	}{
		{"chat", TaskTypeChat},
		{"reasoning", TaskTypeReasoning},
		{"coding", TaskTypeCoding},
		{"analysis", TaskTypeAnalysis},
		{"summarization", TaskTypeSummarization},
		{"extraction", TaskTypeExtraction},
		{"creative", TaskTypeCreative},
		{"translation", TaskTypeTranslation},
		{"math", TaskTypeMath},
		{"vision", TaskTypeVision},
	} {
		if strings.Contains(lower, `"task_type":"`+candidate.key+`"`) ||
			strings.Contains(lower, `"task_type": "`+candidate.key+`"`) {
			taskType = candidate.value
			break
		}
	}

	complexity := ComplexityMedium
	for _, candidate := range []struct {
		key   string
		value TaskComplexity
	}{
		{"simple", ComplexitySimple},
		{"medium", ComplexityMedium},
		{"complex", ComplexityComplex},
		{"expert", ComplexityExpert},
	} {
		if strings.Contains(lower, `"complexity":"`+candidate.key+`"`) ||
			strings.Contains(lower, `"complexity": "`+candidate.key+`"`) {
			complexity = candidate.value
			break
		}
	}
	return taskType, complexity
}

func normalizeTaskType(value string) TaskType {
	switch TaskType(strings.ToLower(strings.TrimSpace(value))) {
	case TaskTypeChat:
		return TaskTypeChat
	case TaskTypeReasoning:
		return TaskTypeReasoning
	case TaskTypeCoding:
		return TaskTypeCoding
	case TaskTypeAnalysis:
		return TaskTypeAnalysis
	case TaskTypeSummarization:
		return TaskTypeSummarization
	case TaskTypeExtraction:
		return TaskTypeExtraction
	case TaskTypeCreative:
		return TaskTypeCreative
	case TaskTypeTranslation:
		return TaskTypeTranslation
	case TaskTypeMath:
		return TaskTypeMath
	case TaskTypeVision:
		return TaskTypeVision
	default:
		return ""
	}
}

func normalizeTaskComplexity(value string) TaskComplexity {
	switch TaskComplexity(strings.ToLower(strings.TrimSpace(value))) {
	case ComplexitySimple:
		return ComplexitySimple
	case ComplexityMedium:
		return ComplexityMedium
	case ComplexityComplex:
		return ComplexityComplex
	case ComplexityExpert:
		return ComplexityExpert
	default:
		return ""
	}
}

// ============== 组合分类器 ==============

// CompositeClassifier 组合分类器
// 先使用规则分类器，置信度低时再使用 LLM 分类器
type CompositeClassifier struct {
	ruleClassifier *RuleBasedClassifier
	llmClassifier  *LLMClassifier

	// confidenceThreshold LLM 分类阈值
	// 当规则分类器置信度低于此值时使用 LLM 分类
	confidenceThreshold float64
}

// NewCompositeClassifier 创建组合分类器
func NewCompositeClassifier(llmProvider llm.Provider) *CompositeClassifier {
	return &CompositeClassifier{
		ruleClassifier:      NewRuleBasedClassifier(),
		llmClassifier:       NewLLMClassifier(llmProvider),
		confidenceThreshold: 0.7,
	}
}

// Name 返回分类器名称
func (c *CompositeClassifier) Name() string {
	return "composite"
}

// Classify 组合分类
func (c *CompositeClassifier) Classify(ctx context.Context, req llm.CompletionRequest) (TaskType, TaskComplexity) {
	// 先使用规则分类
	taskType, complexity := c.ruleClassifier.Classify(ctx, req)

	// 规则命中非 chat 类型视为较高置信度；chat/无命中时按阈值决定是否交给 LLM 复核。
	confidence := 0.9
	if taskType == TaskTypeChat {
		confidence = 0.3
	}
	if confidence < c.confidenceThreshold && c.llmClassifier != nil {
		// 使用 LLM 分类器进行确认
		llmTaskType, llmComplexity := c.llmClassifier.Classify(ctx, req)
		if llmTaskType != TaskTypeChat {
			return llmTaskType, llmComplexity
		}
	}

	return taskType, complexity
}

// SetConfidenceThreshold 设置置信度阈值
func (c *CompositeClassifier) SetConfidenceThreshold(threshold float64) {
	c.confidenceThreshold = threshold
}
