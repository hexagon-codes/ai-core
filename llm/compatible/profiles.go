package compatible

import (
	"sort"

	"github.com/hexagon-codes/ai-core/llm"
)

const (
	ProfileOpenRouter     = "openrouter"
	ProfileGroq           = "groq"
	ProfileTogether       = "together"
	ProfilePerplexity     = "perplexity"
	ProfileXAI            = "xai"
	ProfileMistral        = "mistral"
	ProfileFireworks      = "fireworks"
	ProfileDeepInfra      = "deepinfra"
	ProfileSiliconFlowCN  = "siliconflow-cn"
	ProfileMoonshotCN     = "moonshot-cn"
	ProfileBaichuanCN     = "baichuan-cn"
	ProfileStepFunCN      = "stepfun-cn"
	ProfileModelArkCN     = "modelark-cn"
	ProfileModelArkGlobal = "modelark-global"
	ProfileDoubaoCN       = "doubao-cn"
	ProfileDoubaoGlobal   = "doubao-global"
)

// Profiles returns known OpenAI-compatible provider profiles.
func Profiles() map[string]Profile {
	common := []string{llm.FeatureStreaming, llm.FeatureFunctions, llm.FeatureJSON}
	vision := []string{llm.FeatureStreaming, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureVision}
	return map[string]Profile{
		ProfileOpenRouter: {
			Name:     "openrouter",
			BaseURL:  "https://openrouter.ai/api/v1",
			EnvKey:   "OPENROUTER_API_KEY",
			Model:    "openai/gpt-4o-mini",
			Official: true,
			Stable:   true,
			Source:   "https://openrouter.ai/docs",
			Models: []ModelSpec{
				{ID: "openai/gpt-4o", Name: "GPT-4o via OpenRouter", MaxTokens: 128000, Features: vision},
				{ID: "anthropic/claude-3.5-sonnet", Name: "Claude 3.5 Sonnet via OpenRouter", MaxTokens: 200000, Features: vision},
				{ID: "google/gemini-2.0-flash-001", Name: "Gemini 2.0 Flash via OpenRouter", MaxTokens: 1000000, Features: vision},
				{ID: "deepseek/deepseek-chat", Name: "DeepSeek Chat via OpenRouter", MaxTokens: 64000, Features: common},
			},
		},
		ProfileGroq: {
			Name:     "groq",
			BaseURL:  "https://api.groq.com/openai/v1",
			EnvKey:   "GROQ_API_KEY",
			Model:    "llama-3.3-70b-versatile",
			Official: true,
			Stable:   true,
			Source:   "https://console.groq.com/docs/openai",
			Models: []ModelSpec{
				{ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B Versatile", MaxTokens: 128000, Features: common},
				{ID: "llama-3.1-8b-instant", Name: "Llama 3.1 8B Instant", MaxTokens: 128000, Features: common},
				{ID: "mixtral-8x7b-32768", Name: "Mixtral 8x7B", MaxTokens: 32768, Features: common},
			},
		},
		ProfileTogether: {
			Name:     "together",
			BaseURL:  "https://api.together.xyz/v1",
			EnvKey:   "TOGETHER_API_KEY",
			Model:    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
			Official: true,
			Stable:   true,
			Source:   "https://docs.together.ai/docs/openai-api-compatibility",
			Models: []ModelSpec{
				{ID: "meta-llama/Llama-3.3-70B-Instruct-Turbo", Name: "Llama 3.3 70B Turbo", MaxTokens: 131072, Features: common},
				{ID: "Qwen/Qwen2.5-72B-Instruct-Turbo", Name: "Qwen2.5 72B Turbo", MaxTokens: 32768, Features: common},
				{ID: "deepseek-ai/DeepSeek-V3", Name: "DeepSeek V3", MaxTokens: 64000, Features: common},
			},
		},
		ProfilePerplexity: {
			Name:     "perplexity",
			BaseURL:  "https://api.perplexity.ai",
			EnvKey:   "PERPLEXITY_API_KEY",
			Model:    "sonar-pro",
			Official: true,
			Stable:   true,
			Source:   "https://docs.perplexity.ai",
			Models: []ModelSpec{
				{ID: "sonar-pro", Name: "Sonar Pro", MaxTokens: 200000, Features: common},
				{ID: "sonar", Name: "Sonar", MaxTokens: 128000, Features: common},
			},
		},
		ProfileXAI: {
			Name:     "xai",
			BaseURL:  "https://api.x.ai/v1",
			EnvKey:   "XAI_API_KEY",
			Model:    "grok-2-latest",
			Official: true,
			Stable:   true,
			Source:   "https://docs.x.ai/docs",
			Models: []ModelSpec{
				{ID: "grok-2-latest", Name: "Grok 2", MaxTokens: 131072, Features: common},
				{ID: "grok-2-vision-latest", Name: "Grok 2 Vision", MaxTokens: 32768, Features: vision},
			},
		},
		ProfileMistral: {
			Name:     "mistral",
			BaseURL:  "https://api.mistral.ai/v1",
			EnvKey:   "MISTRAL_API_KEY",
			Model:    "mistral-large-latest",
			Official: true,
			Stable:   true,
			Source:   "https://docs.mistral.ai/api",
			Models: []ModelSpec{
				{ID: "mistral-large-latest", Name: "Mistral Large", MaxTokens: 128000, Features: common},
				{ID: "pixtral-large-latest", Name: "Pixtral Large", MaxTokens: 128000, Features: vision},
				{ID: "codestral-latest", Name: "Codestral", MaxTokens: 256000, Features: common},
			},
		},
		ProfileFireworks: {
			Name:     "fireworks",
			BaseURL:  "https://api.fireworks.ai/inference/v1",
			EnvKey:   "FIREWORKS_API_KEY",
			Model:    "accounts/fireworks/models/llama-v3p3-70b-instruct",
			Official: true,
			Stable:   true,
			Source:   "https://docs.fireworks.ai",
			Models: []ModelSpec{
				{ID: "accounts/fireworks/models/llama-v3p3-70b-instruct", Name: "Llama 3.3 70B Instruct", MaxTokens: 131072, Features: common},
				{ID: "accounts/fireworks/models/deepseek-v3", Name: "DeepSeek V3", MaxTokens: 64000, Features: common},
			},
		},
		ProfileDeepInfra: {
			Name:     "deepinfra",
			BaseURL:  "https://api.deepinfra.com/v1/openai",
			EnvKey:   "DEEPINFRA_API_KEY",
			Model:    "meta-llama/Meta-Llama-3.1-70B-Instruct",
			Official: true,
			Stable:   true,
			Source:   "https://deepinfra.com/docs/openai_api",
			Models: []ModelSpec{
				{ID: "meta-llama/Meta-Llama-3.1-70B-Instruct", Name: "Llama 3.1 70B Instruct", MaxTokens: 131072, Features: common},
				{ID: "Qwen/Qwen2.5-72B-Instruct", Name: "Qwen2.5 72B", MaxTokens: 32768, Features: common},
			},
		},
		ProfileSiliconFlowCN: {
			Name:     "siliconflow-cn",
			BaseURL:  "https://api.siliconflow.cn/v1",
			EnvKey:   "SILICONFLOW_API_KEY",
			Model:    "deepseek-ai/DeepSeek-V3",
			Official: true,
			Stable:   true,
			Region:   "cn",
			Source:   "https://docs.siliconflow.cn",
			Models: []ModelSpec{
				{ID: "deepseek-ai/DeepSeek-V3", Name: "DeepSeek V3", MaxTokens: 64000, Features: common},
				{ID: "Qwen/Qwen2.5-72B-Instruct", Name: "Qwen2.5 72B", MaxTokens: 32768, Features: common},
			},
		},
		ProfileMoonshotCN: {
			Name:     "moonshot-cn",
			BaseURL:  "https://api.moonshot.cn/v1",
			EnvKey:   "MOONSHOT_API_KEY",
			Model:    "moonshot-v1-32k",
			Official: true,
			Stable:   true,
			Region:   "cn",
			Source:   "https://platform.moonshot.cn/docs",
			Models: []ModelSpec{
				{ID: "moonshot-v1-8k", Name: "Moonshot 8K", MaxTokens: 8192, Features: common},
				{ID: "moonshot-v1-32k", Name: "Moonshot 32K", MaxTokens: 32768, Features: common},
				{ID: "moonshot-v1-128k", Name: "Moonshot 128K", MaxTokens: 131072, Features: common},
			},
		},
		ProfileBaichuanCN: {
			Name:     "baichuan-cn",
			BaseURL:  "https://api.baichuan-ai.com/v1",
			EnvKey:   "BAICHUAN_API_KEY",
			Model:    "Baichuan4",
			Official: true,
			Stable:   true,
			Region:   "cn",
			Source:   "https://platform.baichuan-ai.com/docs",
			Models: []ModelSpec{
				{ID: "Baichuan4", Name: "Baichuan 4", MaxTokens: 32768, Features: common},
				{ID: "Baichuan4-Turbo", Name: "Baichuan 4 Turbo", MaxTokens: 32768, Features: common},
			},
		},
		ProfileStepFunCN: {
			Name:     "stepfun-cn",
			BaseURL:  "https://api.stepfun.com/v1",
			EnvKey:   "STEPFUN_API_KEY",
			Model:    "step-2-16k",
			Official: true,
			Stable:   true,
			Region:   "cn",
			Source:   "https://platform.stepfun.com/docs",
			Models: []ModelSpec{
				{ID: "step-2-16k", Name: "Step 2 16K", MaxTokens: 16384, Features: common},
				{ID: "step-1v-32k", Name: "Step 1V 32K", MaxTokens: 32768, Features: vision},
			},
		},
		ProfileModelArkCN:     modelArkProfile("modelark-cn", "cn"),
		ProfileModelArkGlobal: modelArkProfile("modelark-global", "global"),
		ProfileDoubaoCN:       doubaoProfile("doubao-cn", "cn"),
		ProfileDoubaoGlobal:   doubaoProfile("doubao-global", "global"),
	}
}

// ProfileNames returns sorted known profile IDs.
func ProfileNames() []string {
	profiles := Profiles()
	names := make([]string, 0, len(profiles))
	for name := range profiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func modelArkProfile(name, region string) Profile {
	baseURL := "https://ark.cn-beijing.volces.com/api/v3"
	envKey := "ARK_API_KEY"
	if region == "global" {
		baseURL = "https://ark.ap-southeast.bytepluses.com/api/v3"
		envKey = "BYTEPLUS_ARK_API_KEY"
	}
	return Profile{
		Name:     name,
		BaseURL:  baseURL,
		EnvKey:   envKey,
		Model:    "doubao-seed-1-6",
		Official: true,
		Stable:   true,
		Region:   region,
		Source:   "https://docs.byteplus.com/en/docs/ModelArk/1330626",
		Models: []ModelSpec{
			{ID: "doubao-seed-1-6", Name: "Doubao Seed 1.6", MaxTokens: 256000, Features: []string{llm.FeatureStreaming, llm.FeatureFunctions, llm.FeatureJSON, llm.FeatureVision}},
			{ID: "doubao-seed-1-6-thinking", Name: "Doubao Seed 1.6 Thinking", MaxTokens: 256000, Features: []string{llm.FeatureStreaming, llm.FeatureJSON, llm.FeatureVision}},
			{ID: "doubao-pro-32k", Name: "Doubao Pro 32K", MaxTokens: 32768, Features: []string{llm.FeatureStreaming, llm.FeatureFunctions, llm.FeatureJSON}},
			{ID: "doubao-pro-128k", Name: "Doubao Pro 128K", MaxTokens: 131072, Features: []string{llm.FeatureStreaming, llm.FeatureFunctions, llm.FeatureJSON}},
		},
	}
}

func doubaoProfile(name, region string) Profile {
	p := modelArkProfile(name, region)
	p.Model = "doubao-seed-1-6"
	return p
}
