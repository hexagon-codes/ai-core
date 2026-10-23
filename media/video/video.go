// Package videogen 提供文本到视频生成能力。
//
// 视频生成与图像生成的关键差异：
//   - 长任务（智谱 CogVideoX 单次 30s-2min，OpenAI Sora 几分钟）
//   - 普遍走"提交 → 轮询 → 结果"两阶段 API（不能像 DALL-E 一样同步返回）
//
// 因此 Provider 接口分两步：Submit → Poll。前端调用方负责轮询节奏。
package video

import (
	"context"
	"encoding/base64"
	"fmt"
	"sort"
	"strings"

	"github.com/hexagon-codes/ai-core/media"
)

// Provider 文生视频 Provider 接口。
type Provider interface {
	// Name 返回 Provider 名称（如 "zhipu-cogvideox"）
	Name() string
	// Submit 提交生成任务，返回任务 ID（异步 Provider）。
	// 同步 Provider 可在此处直接完成生成，返回 ID 供 Poll 立即返回 done=true。
	Submit(ctx context.Context, req Request) (taskID string, err error)
	// Poll 查询任务状态。done=true 时 Result 应有可用结果。
	Poll(ctx context.Context, taskID string) (status TaskStatus, err error)
	// SupportedModels 返回支持的模型 ID 列表
	SupportedModels() []string
}

// Request 视频生成请求。
type Request struct {
	Model          string         `json:"model"`                     // 模型 ID（如 "cogvideox-2"）
	Prompt         string         `json:"prompt"`                    // 文本提示词
	Negative       string         `json:"negative,omitempty"`        // 反向提示词（部分 Provider 支持）
	ImageURL       string         `json:"image_url,omitempty"`       // 首帧图（图生视频）
	EndImageURL    string         `json:"end_image_url,omitempty"`   // 尾帧图（部分模型支持）
	Quality        string         `json:"quality,omitempty"`         // "speed" / "quality"（CogVideoX）
	WithAudio      bool           `json:"with_audio,omitempty"`      // 是否生成音轨
	Duration       int            `json:"duration,omitempty"`        // 时长秒（部分模型支持）
	FPS            int            `json:"fps,omitempty"`             // 帧率
	Size           string         `json:"size,omitempty"`            // 分辨率 1280x720
	Ratio          string         `json:"ratio,omitempty"`           // 画幅，如 16:9 / 9:16 / 1:1
	Seed           int64          `json:"seed,omitempty"`            // 随机种子，0=不指定
	UserID         string         `json:"user_id,omitempty"`         // 终端用户标识
	IdempotencyKey string         `json:"idempotency_key,omitempty"` // 上游幂等键（若 Provider 支持）
	CallbackURL    string         `json:"callback_url,omitempty"`    // 上游回调 URL（若 Provider 支持）
	Extra          map[string]any `json:"extra,omitempty"`           // Provider 特有参数
}

// TaskStatus 任务状态。
type TaskStatus struct {
	TaskID        string          `json:"task_id"`
	Provider      string          `json:"provider"`
	Model         string          `json:"model"`
	Status        string          `json:"status"`                    // queueing / running / success / failed
	State         media.TaskState `json:"state,omitempty"`           // queued/running/succeeded/failed/cancelled
	Done          bool            `json:"done"`                      // status==success / failed
	VideoURL      string          `json:"video_url,omitempty"`       // Provider 给的临时 URL（24h 过期）
	VideoFilePath string          `json:"video_file_path,omitempty"` // 持久化路径（成功后由 handler 落盘）
	CoverURL      string          `json:"cover_url,omitempty"`       // 封面图 URL（部分 Provider 提供）
	CoverFilePath string          `json:"cover_file_path,omitempty"` // 封面图持久化路径
	Error         string          `json:"error,omitempty"`
	RequestID     string          `json:"request_id,omitempty"` // 上游 request id / trace id
	Billed        *bool           `json:"billed,omitempty"`     // 上游是否计费；nil=unknown
	Progress      float64         `json:"progress,omitempty"`
	UsageMs       int64           `json:"usage_ms,omitempty"`
	RawStatus     string          `json:"raw_status,omitempty"` // 上游原始状态
}

// Service 视频生成服务（多 Provider 路由）。
type Service struct {
	providers       map[string]Provider
	modelIndex      map[string]string
	defaultProvider string
}

// NewService 创建视频生成服务。
func NewService(providers map[string]Provider, defaultName string) *Service {
	s := &Service{
		providers:       make(map[string]Provider, len(providers)),
		modelIndex:      make(map[string]string),
		defaultProvider: defaultName,
	}
	// 先按 Provider 名（map key）字典序排序，保证 modelIndex 的构建顺序
	// 是确定的。否则 Go map 迭代顺序随机，配合下方 "first-wins" 语义会导致
	// 同一 model 被多个 Provider 声明时，路由目标在不同进程/构造间漂移。
	//
	// 冲突仲裁规则（确定性优先级）：同名 model 由 key 字典序最小的 Provider 胜出。
	names := make([]string, 0, len(providers))
	for k := range providers {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, k := range names {
		p := providers[k]
		if p == nil {
			continue
		}
		s.providers[k] = p
		for _, m := range p.SupportedModels() {
			if _, exists := s.modelIndex[m]; !exists {
				s.modelIndex[m] = k
			}
		}
	}
	return s
}

func (s *Service) resolveProvider(name, model string) Provider {
	_, p := s.resolveRoute(name, model)
	return p
}

// resolveRoute 同时保留注册身份，不通过供应商名称反查实例。
func (s *Service) resolveRoute(name, model string) (string, Provider) {
	if name != "" {
		if p, ok := s.providers[name]; ok {
			return name, p
		}
	}
	if model != "" {
		if key, ok := s.modelIndex[model]; ok {
			return key, s.providers[key]
		}
	}
	if s.defaultProvider != "" {
		return s.defaultProvider, s.providers[s.defaultProvider]
	}
	return "", nil
}

const taskReferencePrefix = "aicore-video-v1:"

// wrapTaskID 对别名和歧义名称使用无分隔符冲突的版本化引用。
func (s *Service) wrapTaskID(key string, p Provider, raw string) string {
	legacy := key == p.Name() && !strings.Contains(key, "::")
	candidate := key + "::" + raw
	for otherKey, other := range s.providers {
		if otherKey != key && (strings.HasPrefix(candidate, otherKey+"::") || strings.HasPrefix(candidate, other.Name()+"::")) {
			legacy = false
		}
	}
	if legacy {
		return candidate
	}
	return taskReferencePrefix + base64.RawURLEncoding.EncodeToString([]byte(key)) + ":" + base64.RawURLEncoding.EncodeToString([]byte(raw))
}

// resolveByTaskID 反查任务所属 Provider。
//
// 新任务使用无歧义的旧格式或版本化引用；历史引用仅在唯一匹配时解析。
// Poll 时解析回原 Provider，调用方应将任务引用作为不透明字符串保存。
func (s *Service) resolveByTaskID(taskID string) (Provider, string, error) {
	if strings.HasPrefix(taskID, taskReferencePrefix) && !strings.Contains(taskID, "::") {
		keyPart, rawPart, ok := strings.Cut(strings.TrimPrefix(taskID, taskReferencePrefix), ":")
		key, keyErr := base64.RawURLEncoding.DecodeString(keyPart)
		raw, rawErr := base64.RawURLEncoding.DecodeString(rawPart)
		if !ok || keyErr != nil || rawErr != nil || len(key) == 0 || len(raw) == 0 {
			return nil, "", fmt.Errorf("invalid video task reference")
		}
		p, found := s.providers[string(key)]
		if !found {
			return nil, "", fmt.Errorf("unknown video provider instance %q", key)
		}
		return p, string(raw), nil
	}
	if !strings.Contains(taskID, "::") {
		return nil, "", fmt.Errorf("任务 ID 格式错误（缺少 provider 前缀）")
	}
	var found Provider
	var rawID string
	for key, p := range s.providers {
		seenPrefix := make(map[string]bool, 2)
		for _, prefix := range []string{key, p.Name()} {
			if seenPrefix[prefix] {
				continue
			}
			seenPrefix[prefix] = true
			if raw, ok := strings.CutPrefix(taskID, prefix+"::"); ok {
				if found != nil {
					return nil, "", fmt.Errorf("ambiguous legacy video task reference")
				}
				found, rawID = p, raw
			}
		}
	}
	if found == nil {
		return nil, "", fmt.Errorf("unknown video provider in task reference")
	}
	return found, rawID, nil
}

// Submit 提交生成任务，返回绑定注册实例的不透明任务引用，供 Poll 原样使用。
func (s *Service) Submit(ctx context.Context, providerName string, req Request) (string, error) {
	// 规范化输入：纯空白字符（含 Tab/换行/全角空格 U+3000）的 prompt 视同空，
	// 不能绕过校验向上游发请求（浪费配额、可能触发风控）。
	// 仅当 prompt 规范化后为空【且】未提供首帧图时才拒绝——图生视频允许空 prompt。
	if strings.TrimSpace(req.Prompt) == "" && strings.TrimSpace(req.ImageURL) == "" {
		return "", fmt.Errorf("prompt 与 image_url 至少需要一个有效（非空白）值")
	}
	key, p := s.resolveRoute(providerName, req.Model)
	if p == nil {
		return "", fmt.Errorf("未找到可用的视频生成 Provider（请求 provider=%q model=%q）", providerName, req.Model)
	}
	rawID, err := p.Submit(ctx, req)
	if err != nil {
		return "", err
	}
	return s.wrapTaskID(key, p, rawID), nil
}

// Poll 查询任务状态。
func (s *Service) Poll(ctx context.Context, taskID string) (TaskStatus, error) {
	p, raw, err := s.resolveByTaskID(taskID)
	if err != nil {
		return TaskStatus{}, err
	}
	st, err := p.Poll(ctx, raw)
	if err != nil {
		return TaskStatus{}, err
	}
	st.TaskID = taskID // 返回带前缀 ID 给前端透传
	return st, nil
}

// HasProvider 是否注册了任意 Provider。
func (s *Service) HasProvider() bool { return len(s.providers) > 0 }

// Providers 返回 Provider 名列表。
func (s *Service) Providers() []string {
	names := make([]string, 0, len(s.providers))
	for k := range s.providers {
		names = append(names, k)
	}
	return names
}

// Models 返回所有支持的模型 ID。
func (s *Service) Models() []string {
	models := make([]string, 0, len(s.modelIndex))
	for m := range s.modelIndex {
		models = append(models, m)
	}
	return models
}
