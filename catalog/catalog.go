// Package catalog provides a provider/model capability registry shared by
// LLM and media adapters.
package catalog

import (
	"sort"
	"strings"
	"sync"
)

// Modality identifies the kind of AI capability exposed by a model.
type Modality string

const (
	ModalityText         Modality = "text"
	ModalityEmbedding    Modality = "embedding"
	ModalityRerank       Modality = "rerank"
	ModalityImage        Modality = "image"
	ModalityVideo        Modality = "video"
	ModalitySpeechToText Modality = "speech_to_text"
	ModalityTextToSpeech Modality = "text_to_speech"
	ModalityVoiceChat    Modality = "voice_chat"
	ModalityModeration   Modality = "moderation"
)

// Feature names intentionally mirror llm feature constants where possible,
// while also covering media-specific model capabilities.
const (
	FeatureStreaming      = "streaming"
	FeatureFunctions      = "functions"
	FeatureJSON           = "json_mode"
	FeatureVision         = "vision"
	FeatureImageInput     = "image_input"
	FeatureAudioInput     = "audio_input"
	FeatureAudioOutput    = "audio_output"
	FeatureAsyncTask      = "async_task"
	FeatureSeed           = "seed"
	FeatureNegativePrompt = "negative_prompt"
	FeatureReferenceImage = "reference_image"
	FeatureCallback       = "callback"
)

// Cost stores normalized model pricing. Values are intentionally optional:
// many media vendors price by resolution/duration rather than token.
type Cost struct {
	InputPerMillionTokens  float64 `json:"input_per_million_tokens,omitempty"`
	OutputPerMillionTokens float64 `json:"output_per_million_tokens,omitempty"`
	Unit                   string  `json:"unit,omitempty"`
	Amount                 float64 `json:"amount,omitempty"`
	Currency               string  `json:"currency,omitempty"`
}

// Capability describes a single provider/model capability row.
type Capability struct {
	Provider string   `json:"provider"`
	Model    string   `json:"model"`
	Name     string   `json:"name,omitempty"`
	Modality Modality `json:"modality"`

	Features []string `json:"features,omitempty"`

	ContextWindow   int `json:"context_window,omitempty"`
	MaxInputTokens  int `json:"max_input_tokens,omitempty"`
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`

	Async       bool   `json:"async,omitempty"`
	Stable      bool   `json:"stable,omitempty"`
	OfficialAPI bool   `json:"official_api,omitempty"`
	Source      string `json:"source,omitempty"`

	Cost Cost `json:"cost,omitempty"`
}

// Provider describes adapters that can report their own capability matrix.
type Provider interface {
	Capabilities() []Capability
}

// HasFeature reports whether a capability includes a feature.
func (c Capability) HasFeature(feature string) bool {
	for _, f := range c.Features {
		if f == feature {
			return true
		}
	}
	return false
}

// Registry is an in-memory provider/model capability index.
//
// The zero value is ready to use.
type Registry struct {
	mu   sync.RWMutex
	rows []Capability
}

// NewRegistry constructs a registry and registers any initial rows.
func NewRegistry(rows ...Capability) *Registry {
	r := &Registry{}
	r.Register(rows...)
	return r
}

// Register appends capability rows. Empty provider/model rows are ignored.
func (r *Registry) Register(rows ...Capability) {
	if len(rows) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, row := range rows {
		row.Provider = strings.TrimSpace(row.Provider)
		row.Model = strings.TrimSpace(row.Model)
		if row.Provider == "" || row.Model == "" || row.Modality == "" {
			continue
		}
		row.Features = normalizeStrings(row.Features)
		r.rows = append(r.rows, row)
	}
	sortCapabilities(r.rows)
}

// RegisterProvider appends all capabilities reported by p.
func (r *Registry) RegisterProvider(p Provider) {
	if p == nil {
		return
	}
	r.Register(p.Capabilities()...)
}

// All returns a sorted snapshot of all capability rows.
func (r *Registry) All() []Capability {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := append([]Capability(nil), r.rows...)
	sortCapabilities(out)
	return out
}

// Find returns all rows matching the non-zero fields in q.
func (r *Registry) Find(q Query) []Capability {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Capability, 0)
	for _, row := range r.rows {
		if q.Provider != "" && row.Provider != q.Provider {
			continue
		}
		if q.Model != "" && row.Model != q.Model {
			continue
		}
		if q.Modality != "" && row.Modality != q.Modality {
			continue
		}
		if q.Feature != "" && !row.HasFeature(q.Feature) {
			continue
		}
		out = append(out, row)
	}
	sortCapabilities(out)
	return out
}

// Query filters registry rows.
type Query struct {
	Provider string
	Model    string
	Modality Modality
	Feature  string
}

func normalizeStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sortCapabilities(rows []Capability) {
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Provider != rows[j].Provider {
			return rows[i].Provider < rows[j].Provider
		}
		if rows[i].Modality != rows[j].Modality {
			return rows[i].Modality < rows[j].Modality
		}
		return rows[i].Model < rows[j].Model
	})
}
