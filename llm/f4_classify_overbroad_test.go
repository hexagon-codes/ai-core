package llm

// 回归锁定（F4）：failover ClassifyError 曾用裸 "无效" 判 FailInvalidKey，过宽 ——
// 中文"无效的请求参数 / 无效的模型 / 无效的分辨率"等非凭证错误被误判为凭证无效
// （HandleFailover → Retry=false + "API 凭证无效"）。现已收窄为凭证语境关键词，
// 非凭证错误归 FailUnknown。本用例断言非凭证错误不再判为 FailInvalidKey。

import (
	"errors"
	"testing"
)

func TestF4_InvalidKeywordOverbroadMisclassifies(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"中文无效请求参数", errors.New("无效的请求参数")},
		{"中文无效模型名", errors.New("无效的模型名称")},
		{"中文无效分辨率", errors.New("image_generation: 无效的分辨率")},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ClassifyError(c.err, 0, "")
			t.Logf("ClassifyError(%q, 0, \"\") = %v", c.err, got)
			if got == FailInvalidKey {
				t.Fatalf("F4 证实：%q 被分类为 FailInvalidKey（不重试 + 凭证无效提示），"+
					"但它是请求/参数错误，不是凭证错误。裸 \"无效\" 关键词匹配过宽。", c.err)
			}
		})
	}
}
