package llm

import (
	"fmt"
	"strings"
)

const (
	// ReasoningCapabilityMetadataKey 保存由可信配置提供的显式推理能力。
	ReasoningCapabilityMetadataKey = "reasoning_capability_v1"
	// ReasoningReceiptObserverMetadataKey 用于向调用方逐步发布执行证据。
	ReasoningReceiptObserverMetadataKey = "reasoning_receipt_observer_v1"
)

// ReasoningSupport 表示模型推理能力的精确三态。
type ReasoningSupport string

const (
	ReasoningSupported   ReasoningSupport = "supported"
	ReasoningUnsupported ReasoningSupport = "unsupported"
	ReasoningUnknown     ReasoningSupport = "unknown"
)

// ReasoningDialect 表示上游请求使用的显式控制字段。
type ReasoningDialect string

const (
	ReasoningDialectEffort         ReasoningDialect = "reasoning_effort"
	ReasoningDialectEnableThinking ReasoningDialect = "enable_thinking"
	ReasoningDialectThink          ReasoningDialect = "think"
	ReasoningDialectThinking       ReasoningDialect = "thinking"
)

// ReasoningApplication 表示有证据支持的请求执行结论。
type ReasoningApplication string

const (
	ReasoningApplicationApplied  ReasoningApplication = "applied"
	ReasoningApplicationIgnored  ReasoningApplication = "ignored"
	ReasoningApplicationRejected ReasoningApplication = "rejected"
	ReasoningApplicationUnknown  ReasoningApplication = "unknown"
)

// ReasoningCapability 描述一个模型经过配置确认的推理控制契约。
type ReasoningCapability struct {
	Support  ReasoningSupport `json:"support"`
	Dialect  ReasoningDialect `json:"dialect,omitempty"`
	OnValue  any              `json:"on,omitempty"`
	OffValue any              `json:"off,omitempty"`
}

// ReasoningPlanRequest 是生成上游推理控制字段所需的输入。
type ReasoningPlanRequest struct {
	Enabled    bool
	Model      string
	BaseURL    string
	Capability ReasoningCapability
}

// ReasoningPlan 同时保留请求字段和初始执行回执。
type ReasoningPlan struct {
	Wire    map[string]any
	Receipt ReasoningReceipt
}

// ReasoningReceipt 分离请求发送、上游接受和可观察执行证据。
type ReasoningReceipt struct {
	Version         int                  `json:"version"`
	Enabled         bool                 `json:"enabled"`
	Support         ReasoningSupport     `json:"support"`
	Dialect         ReasoningDialect     `json:"dialect,omitempty"`
	Sent            bool                 `json:"sent"`
	Accepted        bool                 `json:"accepted"`
	Observed        bool                 `json:"observed"`
	ReasoningTokens int                  `json:"reasoning_tokens,omitempty"`
	Applied         bool                 `json:"applied"`
	Application     ReasoningApplication `json:"application"`
}

// WithEvidence 返回合并上游接受状态和可观察推理证据后的回执。
func (r ReasoningReceipt) WithEvidence(accepted, observedDelta bool, reasoningTokens int) ReasoningReceipt {
	if r.Version == 0 {
		r.Version = 1
	}
	r.Accepted = r.Sent && accepted
	if reasoningTokens > r.ReasoningTokens {
		r.ReasoningTokens = reasoningTokens
	}
	r.Observed = r.Observed || observedDelta || r.ReasoningTokens > 0
	r.Applied = r.Enabled && r.Observed
	switch {
	case r.Applied:
		r.Application = ReasoningApplicationApplied
	case !r.Enabled && r.Observed:
		r.Application = ReasoningApplicationIgnored
	case r.Application == "":
		r.Application = ReasoningApplicationUnknown
	}
	return r
}

// ReasoningUnsupportedError 表示明确不支持推理的模型拒绝了开启请求。
type ReasoningUnsupportedError struct {
	Model string
}

func (e *ReasoningUnsupportedError) Error() string {
	if strings.TrimSpace(e.Model) == "" {
		return "reasoning is not supported"
	}
	return fmt.Sprintf("reasoning is not supported by model %q", e.Model)
}

func planReasoning(req ReasoningPlanRequest) (ReasoningPlan, error) {
	support := normalizeReasoningSupport(req.Capability.Support)
	plan := ReasoningPlan{
		Wire: map[string]any{},
		Receipt: ReasoningReceipt{
			Version:     1,
			Enabled:     req.Enabled,
			Support:     support,
			Dialect:     req.Capability.Dialect,
			Application: ReasoningApplicationUnknown,
		},
	}

	switch support {
	case ReasoningUnknown:
		return plan, nil
	case ReasoningUnsupported:
		if !req.Enabled {
			return plan, nil
		}
		plan.Receipt.Application = ReasoningApplicationRejected
		return plan, &ReasoningUnsupportedError{Model: req.Model}
	}

	if !isReasoningDialect(req.Capability.Dialect) {
		return plan, fmt.Errorf("invalid reasoning dialect %q", req.Capability.Dialect)
	}
	value := req.Capability.OffValue
	if req.Enabled {
		value = req.Capability.OnValue
	}
	if value == nil {
		return plan, fmt.Errorf("reasoning dialect %q has no configured value", req.Capability.Dialect)
	}
	plan.Wire[string(req.Capability.Dialect)] = value
	plan.Receipt.Sent = true
	return plan, nil
}

// PlanReasoning 生成不依赖模型名或 BaseURL 的显式推理控制计划。
func PlanReasoning(req ReasoningPlanRequest) (ReasoningPlan, error) {
	return planReasoning(req)
}

// PlanReasoningFromMetadata 从版本化能力元数据生成推理控制计划。
func PlanReasoningFromMetadata(metadata map[string]any, model, baseURL string) (ReasoningPlan, bool, error) {
	raw, explicit := metadata[ReasoningCapabilityMetadataKey]
	if !explicit {
		return ReasoningPlan{}, false, nil
	}

	capability, err := reasoningCapabilityFromValue(raw)
	if err != nil {
		return ReasoningPlan{}, true, err
	}
	enabled, err := reasoningIntentFromMetadata(metadata)
	if err != nil {
		return ReasoningPlan{}, true, err
	}
	plan, err := planReasoning(ReasoningPlanRequest{
		Enabled:    enabled,
		Model:      model,
		BaseURL:    baseURL,
		Capability: capability,
	})
	return plan, true, err
}

// PublishReasoningReceipt 将内部证据发布给可信调用方，不影响 Provider 返回值。
func PublishReasoningReceipt(metadata map[string]any, receipt ReasoningReceipt) {
	observer, ok := metadata[ReasoningReceiptObserverMetadataKey]
	if !ok || observer == nil {
		return
	}
	defer func() {
		_ = recover()
	}()
	switch notify := observer.(type) {
	case func(ReasoningReceipt):
		notify(receipt)
	case func(map[string]any):
		notify(receiptMap(receipt))
	}
}

func reasoningCapabilityFromValue(raw any) (ReasoningCapability, error) {
	if capability, ok := raw.(ReasoningCapability); ok {
		capability.Support = normalizeReasoningSupport(capability.Support)
		return capability, nil
	}
	values, ok := raw.(map[string]any)
	if !ok {
		return ReasoningCapability{}, fmt.Errorf("reasoning capability must be an object")
	}
	support, ok := values["support"].(string)
	if !ok {
		return ReasoningCapability{}, fmt.Errorf("reasoning capability support is required")
	}
	capability := ReasoningCapability{
		Support: ReasoningSupport(strings.ToLower(strings.TrimSpace(support))),
	}
	if dialect, ok := values["dialect"].(string); ok {
		capability.Dialect = ReasoningDialect(strings.ToLower(strings.TrimSpace(dialect)))
	}
	capability.OnValue = values["on"]
	capability.OffValue = values["off"]
	if !isReasoningSupport(capability.Support) {
		return ReasoningCapability{}, fmt.Errorf("invalid reasoning support %q", capability.Support)
	}
	return capability, nil
}

func reasoningIntentFromMetadata(metadata map[string]any) (bool, error) {
	raw, ok := metadata["thinking"]
	if !ok {
		return false, fmt.Errorf("reasoning intent is required")
	}
	switch value := raw.(type) {
	case bool:
		return value, nil
	case string:
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "on", "true", "1", "yes", "enabled":
			return true, nil
		case "off", "false", "0", "no", "disabled":
			return false, nil
		}
	}
	return false, fmt.Errorf("invalid reasoning intent")
}

func normalizeReasoningSupport(support ReasoningSupport) ReasoningSupport {
	if isReasoningSupport(support) {
		return support
	}
	return ReasoningUnknown
}

func isReasoningSupport(support ReasoningSupport) bool {
	switch support {
	case ReasoningSupported, ReasoningUnsupported, ReasoningUnknown:
		return true
	default:
		return false
	}
}

func isReasoningDialect(dialect ReasoningDialect) bool {
	switch dialect {
	case ReasoningDialectEffort, ReasoningDialectEnableThinking, ReasoningDialectThink, ReasoningDialectThinking:
		return true
	default:
		return false
	}
}

func receiptMap(receipt ReasoningReceipt) map[string]any {
	return map[string]any{
		"version":          receipt.Version,
		"enabled":          receipt.Enabled,
		"support":          string(receipt.Support),
		"dialect":          string(receipt.Dialect),
		"sent":             receipt.Sent,
		"accepted":         receipt.Accepted,
		"observed":         receipt.Observed,
		"reasoning_tokens": receipt.ReasoningTokens,
		"applied":          receipt.Applied,
		"application":      string(receipt.Application),
	}
}

// ReasoningEvidenceMap 返回可跨包传递但不得直接作为产品公共回执的内部事实。
func ReasoningEvidenceMap(receipt ReasoningReceipt) map[string]any {
	return receiptMap(receipt)
}
