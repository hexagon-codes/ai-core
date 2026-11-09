package schema

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"strconv"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"golang.org/x/text/message"
)

// Validator 持有已解析的不可变 Schema，可并发校验多个 JSON 值。
// 默认方言为 Draft 2020-12，format 仅为注释；本地 $ref/$defs 由解析器处理。
// 不隐式获取远程引用，宿主应提供包含所需定义的完整文档。
type Validator struct{ resolved *jsonschema.Schema }

// NewValidator 编译 Schema 对象、JSON 文档或布尔 Schema 的独立快照。
// 编译后修改原对象不改变验证合同；非法 Schema 或未解析引用在模型调用前报错。
func NewValidator(document any) (*Validator, error) {
	resolved, err := compileStandardSchema(document)
	if err != nil {
		return nil, err
	}
	return &Validator{resolved: resolved}, nil
}

func compileStandardSchema(document any) (*jsonschema.Schema, error) {
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
	// 显式数值字符串约束参与同一次编译，引用、条件和组合由标准引擎统一决议。
	compiler.RegisterVocabulary(&jsonschema.Vocabulary{
		URL: "urn:hexagon:schema:string-number",
		Compile: func(_ *jsonschema.CompilerContext, object map[string]any) (jsonschema.SchemaExt, error) {
			if !hasNumericStringMarker(object) {
				return nil, nil
			}
			minimum, hasMinimum := numericBound(object["minimum"])
			maximum, hasMaximum := numericBound(object["maximum"])
			if !hasMinimum && !hasMaximum {
				return nil, nil
			}
			return &numericStringBounds{minimum: minimum, maximum: maximum}, nil
		},
	})
	compiler.AssertVocabs()
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
	return resolved, nil
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

// numericStringBounds 持有本次编译固定的精确边界，可供多个验证调用只读共享。
type numericStringBounds struct{ minimum, maximum *big.Rat }

func (bounds *numericStringBounds) Validate(ctx *jsonschema.ValidatorContext, value any) {
	text, ok := value.(string)
	if !ok {
		return
	}
	number, err := parseNumberString(text)
	if err == nil && bounds.minimum != nil && number.Cmp(bounds.minimum) < 0 {
		err = fmt.Errorf("string-encoded number %q is below minimum %s", text, bounds.minimum.RatString())
	}
	if err == nil && bounds.maximum != nil && number.Cmp(bounds.maximum) > 0 {
		err = fmt.Errorf("string-encoded number %q exceeds maximum %s", text, bounds.maximum.RatString())
	}
	if err != nil {
		ctx.AddError(&numericStringError{err: err})
	}
}

type numericStringError struct{ err error }

func (*numericStringError) KeywordPath() []string { return []string{numericStringKeyword} }
func (e *numericStringError) LocalizedString(printer *message.Printer) string {
	return printer.Sprintf("%s", e.err)
}

func hasNumericStringMarker(schema map[string]any) bool {
	marker, _ := schema[numericStringKeyword].(bool)
	if !marker {
		return false
	}
	if typ, ok := schema["type"].(string); ok && typ != "string" {
		return false
	}
	return true
}

func numericBound(value any) (*big.Rat, bool) {
	switch number := value.(type) {
	case json.Number:
		bound, ok := new(big.Rat).SetString(number.String())
		return bound, ok
	case float64:
		bound := new(big.Rat).SetFloat64(number)
		return bound, bound != nil
	default:
		return nil, false
	}
}

func parseNumberString(value string) (*big.Rat, error) {
	// 显式十进制保留前导零和符号，避免大数解析器按八进制解释整数。
	digits := strings.TrimPrefix(strings.TrimPrefix(value, "-"), "+")
	if digits != "" && strings.IndexFunc(digits, func(r rune) bool { return r < '0' || r > '9' }) < 0 {
		if integer, ok := new(big.Int).SetString(value, 10); ok {
			return new(big.Rat).SetInt(integer), nil
		}
	}
	// Go 浮点字符串允许前导加号和零；范围比较仍用原字面量，不取 float64 近似。
	if _, err := strconv.ParseFloat(value, 64); err != nil && !errors.Is(err, strconv.ErrRange) {
		return nil, fmt.Errorf("string-encoded number %q is not a valid number", value)
	}
	bound, ok := new(big.Rat).SetString(value)
	if !ok {
		return nil, fmt.Errorf("string-encoded number %q cannot be compared", value)
	}
	return bound, nil
}
