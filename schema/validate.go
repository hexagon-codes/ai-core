package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Validator 持有已解析的不可变 Schema，可并发校验多个 JSON 值。
// 默认方言为 Draft 2020-12，format 仅为注释；本地 $ref/$defs 由解析器处理。
// 不隐式获取远程引用，宿主应提供包含所需定义的完整文档。
type Validator struct{ resolved *jsonschema.Schema }

// NewValidator 编译 Schema 对象、JSON 文档或布尔 Schema 的独立快照。
// 编译后修改原对象不改变验证合同；非法 Schema 或未解析引用在模型调用前报错。
func NewValidator(document any) (*Validator, error) {
	data, err := json.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("encode schema: %w", err)
	}
	raw := bytes.TrimSpace(data)
	if len(raw) == 0 || raw[0] != '{' && !bytes.Equal(raw, []byte("true")) && !bytes.Equal(raw, []byte("false")) {
		return nil, fmt.Errorf("schema document must be an object or boolean")
	}
	definition, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode schema: %w", err)
	}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	// 基础校验只处理传入文档及其内部引用，不在编译期间隐式读取其他文件或网络。
	compiler.UseLoader(jsonschema.SchemeURLLoader{})
	const root = "urn:hexagon:schema:root"
	if err := compiler.AddResource(root, definition); err != nil {
		return nil, fmt.Errorf("register schema: %w", err)
	}
	resolved, err := compiler.Compile(root)
	if err != nil {
		return nil, fmt.Errorf("resolve schema: %w", err)
	}
	return &Validator{resolved: resolved}, nil
}

// Validate 校验任意可编码的 Go 值；复杂类型按其实际 JSON 表示校验。
func (v *Validator) Validate(value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode schema instance: %w", err)
	}
	return v.ValidateJSON(data)
}

// ValidateJSON 校验原始 JSON，保留大整数精度和所有实际字段，不先投影到目标 struct。
func (v *Validator) ValidateJSON(data []byte) error {
	if v == nil || v.resolved == nil {
		return fmt.Errorf("schema validator is not initialized")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return fmt.Errorf("decode schema instance: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("schema instance contains multiple JSON values")
		}
		return fmt.Errorf("decode schema instance suffix: %w", err)
	}
	if err := v.resolved.Validate(value); err != nil {
		return fmt.Errorf("schema validation: %w", err)
	}
	return nil
}
