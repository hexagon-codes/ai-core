package schema

import (
	"encoding/json"
	"testing"
)

func TestOf_BasicTypes(t *testing.T) {
	tests := []struct {
		name     string
		schema   *Schema
		wantType string
	}{
		{"string", Of[string](), "string"},
		{"int", Of[int](), "integer"},
		{"int64", Of[int64](), "integer"},
		{"float64", Of[float64](), "number"},
		{"bool", Of[bool](), "boolean"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.schema.Type != tt.wantType {
				t.Errorf("Of[%s]().Type = %q, want %q", tt.name, tt.schema.Type, tt.wantType)
			}
		})
	}
}

func TestOf_Slice(t *testing.T) {
	schema := Of[[]string]()

	if schema.Type != "array" {
		t.Errorf("Type = %q, want %q", schema.Type, "array")
	}
	if schema.Items == nil {
		t.Fatal("Items is nil")
	}
	if schema.Items.Type != "string" {
		t.Errorf("Items.Type = %q, want %q", schema.Items.Type, "string")
	}
}

func TestOf_Pointer(t *testing.T) {
	schema := Of[*string]()

	if schema.Type != "string" {
		t.Errorf("Type = %q, want %q", schema.Type, "string")
	}
}

type SimpleStruct struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestOf_SimpleStruct(t *testing.T) {
	schema := Of[SimpleStruct]()

	if schema.Type != "object" {
		t.Errorf("Type = %q, want %q", schema.Type, "object")
	}
	if len(schema.Properties) != 2 {
		t.Errorf("len(Properties) = %d, want %d", len(schema.Properties), 2)
	}
	if schema.Properties["name"] == nil {
		t.Error("Properties[name] is nil")
	}
	if schema.Properties["name"].Type != "string" {
		t.Errorf("Properties[name].Type = %q, want %q", schema.Properties["name"].Type, "string")
	}
	if schema.Properties["age"] == nil {
		t.Error("Properties[age] is nil")
	}
	if schema.Properties["age"].Type != "integer" {
		t.Errorf("Properties[age].Type = %q, want %q", schema.Properties["age"].Type, "integer")
	}
}

type AnnotatedStruct struct {
	Name  string  `json:"name" desc:"用户名称" required:"true"`
	Email string  `json:"email" format:"email"`
	Age   int     `json:"age" min:"0" max:"150"`
	Role  string  `json:"role" enum:"admin,user,guest"`
	Score float64 `json:"score" default:"0.0"`
}

func TestOf_AnnotatedStruct(t *testing.T) {
	schema := Of[AnnotatedStruct]()

	// 检查 required
	if len(schema.Required) != 1 || schema.Required[0] != "name" {
		t.Errorf("Required = %v, want [name]", schema.Required)
	}

	// 检查 description
	if schema.Properties["name"].Description != "用户名称" {
		t.Errorf("Properties[name].Description = %q, want %q", schema.Properties["name"].Description, "用户名称")
	}

	// 检查 format
	if schema.Properties["email"].Format != "email" {
		t.Errorf("Properties[email].Format = %q, want %q", schema.Properties["email"].Format, "email")
	}

	// 检查 min/max
	if schema.Properties["age"].Minimum == nil || *schema.Properties["age"].Minimum != 0 {
		t.Error("Properties[age].Minimum != 0")
	}
	if schema.Properties["age"].Maximum == nil || *schema.Properties["age"].Maximum != 150 {
		t.Error("Properties[age].Maximum != 150")
	}

	// 检查 enum
	if len(schema.Properties["role"].Enum) != 3 {
		t.Errorf("len(Properties[role].Enum) = %d, want 3", len(schema.Properties["role"].Enum))
	}

	// 检查 default（default 值根据字段类型自动转换：number 类型的 "0.0" 转为 float64(0)）
	if schema.Properties["score"].Default != float64(0) {
		t.Errorf("Properties[score].Default = %v, want %v", schema.Properties["score"].Default, float64(0))
	}
}

type NestedStruct struct {
	User    SimpleStruct   `json:"user"`
	Friends []SimpleStruct `json:"friends"`
}

func TestOf_NestedStruct(t *testing.T) {
	schema := Of[NestedStruct]()

	if schema.Type != "object" {
		t.Errorf("Type = %q, want %q", schema.Type, "object")
	}

	// 检查嵌套对象
	userSchema := schema.Properties["user"]
	if userSchema == nil {
		t.Fatal("Properties[user] is nil")
	}
	if userSchema.Type != "object" {
		t.Errorf("Properties[user].Type = %q, want %q", userSchema.Type, "object")
	}

	// 检查嵌套数组
	friendsSchema := schema.Properties["friends"]
	if friendsSchema == nil {
		t.Fatal("Properties[friends] is nil")
	}
	if friendsSchema.Type != "array" {
		t.Errorf("Properties[friends].Type = %q, want %q", friendsSchema.Type, "array")
	}
	if friendsSchema.Items == nil || friendsSchema.Items.Type != "object" {
		t.Error("Properties[friends].Items is not object")
	}
}

func TestOf_IgnoreUnexported(t *testing.T) {
	type MixedStruct struct {
		Public  string `json:"public"`
		private string
	}
	// 构造包含未导出字段的值，确保 fixture 自身被使用；Of 仍应忽略该字段。
	_ = MixedStruct{Public: "visible", private: "hidden"}

	schema := Of[MixedStruct]()

	if len(schema.Properties) != 1 {
		t.Errorf("len(Properties) = %d, want 1", len(schema.Properties))
	}
	if schema.Properties["public"] == nil {
		t.Error("Properties[public] is nil")
	}
}

func TestOf_IgnoreJsonDash(t *testing.T) {
	type IgnoreStruct struct {
		Include string `json:"include"`
		Ignore  string `json:"-"`
	}

	schema := Of[IgnoreStruct]()

	if len(schema.Properties) != 1 {
		t.Errorf("len(Properties) = %d, want 1", len(schema.Properties))
	}
	if schema.Properties["include"] == nil {
		t.Error("Properties[include] is nil")
	}
}

func TestBuilder(t *testing.T) {
	schema := NewBuilder().
		Type("object").
		Description("用户信息").
		Property("name", String("用户名"), true).
		Property("age", Integer("年龄"), false).
		Build()

	if schema.Type != "object" {
		t.Errorf("Type = %q, want %q", schema.Type, "object")
	}
	if schema.Description != "用户信息" {
		t.Errorf("Description = %q, want %q", schema.Description, "用户信息")
	}
	if len(schema.Properties) != 2 {
		t.Errorf("len(Properties) = %d, want 2", len(schema.Properties))
	}
	if len(schema.Required) != 1 || schema.Required[0] != "name" {
		t.Errorf("Required = %v, want [name]", schema.Required)
	}
}

func TestBuilder_MinMax(t *testing.T) {
	schema := NewBuilder().
		Type("number").
		Min(0).
		Max(100).
		Build()

	if schema.Minimum == nil || *schema.Minimum != 0 {
		t.Error("Minimum != 0")
	}
	if schema.Maximum == nil || *schema.Maximum != 100 {
		t.Error("Maximum != 100")
	}
}

func TestBuilder_StringConstraints(t *testing.T) {
	schema := NewBuilder().
		Type("string").
		MinLength(1).
		MaxLength(100).
		Pattern("^[a-z]+$").
		Format("email").
		Build()

	if schema.MinLength == nil || *schema.MinLength != 1 {
		t.Error("MinLength != 1")
	}
	if schema.MaxLength == nil || *schema.MaxLength != 100 {
		t.Error("MaxLength != 100")
	}
	if schema.Pattern != "^[a-z]+$" {
		t.Errorf("Pattern = %q, want %q", schema.Pattern, "^[a-z]+$")
	}
	if schema.Format != "email" {
		t.Errorf("Format = %q, want %q", schema.Format, "email")
	}
}

func TestBuilder_Enum(t *testing.T) {
	schema := NewBuilder().
		Type("string").
		Enum("a", "b", "c").
		Build()

	if len(schema.Enum) != 3 {
		t.Errorf("len(Enum) = %d, want 3", len(schema.Enum))
	}
}

func TestConvenienceFunctions(t *testing.T) {
	tests := []struct {
		name   string
		schema *Schema
		typ    string
		desc   string
	}{
		{"String", String("desc"), "string", "desc"},
		{"Integer", Integer("desc"), "integer", "desc"},
		{"Number", Number("desc"), "number", "desc"},
		{"Boolean", Boolean("desc"), "boolean", "desc"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.schema.Type != tt.typ {
				t.Errorf("Type = %q, want %q", tt.schema.Type, tt.typ)
			}
			if tt.schema.Description != tt.desc {
				t.Errorf("Description = %q, want %q", tt.schema.Description, tt.desc)
			}
		})
	}
}

func TestArray(t *testing.T) {
	schema := Array("字符串数组", String("元素"))

	if schema.Type != "array" {
		t.Errorf("Type = %q, want %q", schema.Type, "array")
	}
	if schema.Description != "字符串数组" {
		t.Errorf("Description = %q, want %q", schema.Description, "字符串数组")
	}
	if schema.Items == nil || schema.Items.Type != "string" {
		t.Error("Items is not string type")
	}
}

func TestStringEnum(t *testing.T) {
	schema := StringEnum("状态", "active", "inactive", "pending")

	if schema.Type != "string" {
		t.Errorf("Type = %q, want %q", schema.Type, "string")
	}
	if len(schema.Enum) != 3 {
		t.Errorf("len(Enum) = %d, want 3", len(schema.Enum))
	}
}

func TestSchema_String(t *testing.T) {
	schema := Of[SimpleStruct]()
	str := schema.String()

	// 验证是有效的 JSON
	var parsed map[string]any
	if err := json.Unmarshal([]byte(str), &parsed); err != nil {
		t.Errorf("String() is not valid JSON: %v", err)
	}
}

func TestFromValue(t *testing.T) {
	// nil
	schema := FromValue(nil)
	if schema.Type != "null" {
		t.Errorf("FromValue(nil).Type = %q, want %q", schema.Type, "null")
	}

	// struct value
	schema = FromValue(SimpleStruct{})
	if schema.Type != "object" {
		t.Errorf("FromValue(struct).Type = %q, want %q", schema.Type, "object")
	}
}

func TestOf_Map(t *testing.T) {
	schema := Of[map[string]int]()

	if schema.Type != "object" {
		t.Errorf("Type = %q, want %q", schema.Type, "object")
	}
}

func TestOf_Interface(t *testing.T) {
	schema := Of[any]()

	// interface{}/any 的零值是 nil，所以类型是 null
	if schema.Type != "null" {
		t.Errorf("Type = %q, want %q", schema.Type, "null")
	}
}
