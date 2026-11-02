package catalog

import (
	"net/url"
	"strings"
)

const (
	// FeatureComputerUse 表示模型支持原生计算机操作工具，不由图像输入能力推断。
	FeatureComputerUse = "computer_use"
	// FeatureResponses 表示模型支持 Responses API，不表示调用方必须选择该协议。
	FeatureResponses = "responses"
)

// LookupOfficialModel 返回已核实的官方端点与精确型号能力。
// 未收录的型号、别名及兼容端点返回 false；目录事实不代表当前账号已通过实际调用。
// 返回值独立持有 Features，调用方可以投影到自己的配置而不修改共享目录。
func LookupOfficialModel(endpoint, model string) (Capability, bool) {
	u, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Port() != "" && u.Port() != "443") {
		return Capability{}, false
	}
	path := strings.TrimRight(u.EscapedPath(), "/")
	switch strings.ToLower(u.Hostname()) {
	case "api.deepseek.com":
		if (path == "" || path == "/v1") && (model == "deepseek-flash" || model == "deepseek-v4-flash") {
			return Capability{
				Provider: "deepseek", Model: model, Modality: ModalityText,
				Features:    []string{FeatureVision, FeatureImageInput},
				OfficialAPI: true,
				Source:      "https://api-docs.deepseek.com/guides/vision/",
			}, true
		}
	case "api.openai.com":
		if path == "/v1" {
			switch model {
			case "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
				"gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra":
				sourceModel := model
				if model == "gpt-5.6" {
					sourceModel = "gpt-5.6-sol"
				}
				return Capability{
					Provider: "openai", Model: model, Modality: ModalityText,
					Features:    []string{FeatureVision, FeatureImageInput, FeatureComputerUse, FeatureResponses},
					OfficialAPI: true,
					Source:      "https://developers.openai.com/api/docs/models/" + sourceModel,
				}, true
			}
		}
	}
	return Capability{}, false
}
