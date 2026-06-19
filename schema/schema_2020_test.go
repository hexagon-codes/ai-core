package schema

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestStrict_ClosedObject 验证 Strict() 把对象标记为封闭对象：
// additionalProperties=false 且全部属性进入 required（排序确定）。
func TestStrict_ClosedObject(t *testing.T) {
	obj := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"name": String("姓名"),
			"age":  Integer("年龄"),
		},
	}
	obj.Strict()

	if obj.AdditionalProperties != false {
		t.Fatalf("Strict 后 AdditionalProperties 应为 bool false, got %#v", obj.AdditionalProperties)
	}
	if len(obj.Required) != 2 || obj.Required[0] != "age" || obj.Required[1] != "name" {
		t.Fatalf("Strict 后 Required 应为排序后的全部属性 [age name], got %v", obj.Required)
	}

	// 序列化应输出 "additionalProperties":false
	b, err := json.Marshal(obj)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"additionalProperties":false`) {
		t.Errorf("序列化缺少 additionalProperties:false, got %s", b)
	}
}

// TestStrict_NonObject 验证 Strict() 对非对象类型不做改动。
func TestStrict_NonObject(t *testing.T) {
	s := String("文本")
	s.Strict()
	if s.AdditionalProperties != nil {
		t.Errorf("非对象 Strict 不应设置 AdditionalProperties, got %#v", s.AdditionalProperties)
	}
}

// TestAdditionalProperties_Schema 验证 additionalProperties 仍可承载 *Schema（约束额外属性）。
func TestAdditionalProperties_Schema(t *testing.T) {
	s := &Schema{
		Type:                 "object",
		AdditionalProperties: Integer("额外整数"),
	}
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"additionalProperties":{`) {
		t.Errorf("additionalProperties 为 *Schema 时应序列化为对象, got %s", b)
	}
}

// TestWithDialect 验证 $schema 方言声明的 opt-in 输出（默认不输出）。
func TestWithDialect(t *testing.T) {
	plain := Object("无方言")
	if b, _ := json.Marshal(plain); strings.Contains(string(b), "$schema") {
		t.Errorf("默认不应输出 $schema, got %s", b)
	}

	withDialect := Object("带方言").WithDialect(Draft2020_12)
	b, _ := json.Marshal(withDialect)
	if !strings.Contains(string(b), Draft2020_12) {
		t.Errorf("WithDialect 后应输出 $schema=%s, got %s", Draft2020_12, b)
	}
}
