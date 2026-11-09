package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"sort"
	"strings"
	"sync"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// Validator 持有已解析的不可变 Schema，可并发校验多个 JSON 值。
// 默认方言为 Draft 2020-12，format 仅为注释；本地 $ref/$defs 由解析器处理。
// 不隐式获取远程引用，宿主应提供包含所需定义的完整文档。
type Validator struct {
	resolved *jsonschema.Schema
	document any
	// branchCache 保存 anyOf/oneOf 分支的标准验证器，避免自定义标记误伤未匹配分支。
	branchCache sync.Map
}

// NewValidator 编译 Schema 对象、JSON 文档或布尔 Schema 的独立快照。
// 编译后修改原对象不改变验证合同；非法 Schema 或未解析引用在模型调用前报错。
func NewValidator(document any) (*Validator, error) {
	resolved, err := compileStandardSchema(document)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(document)
	if err != nil {
		return nil, fmt.Errorf("encode schema: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var decodedDocument any
	if err := decoder.Decode(&decodedDocument); err != nil {
		return nil, fmt.Errorf("decode schema document: %w", err)
	}
	return &Validator{resolved: resolved, document: decodedDocument}, nil
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
	if err := v.validateNumericStrings(v.document, value, "#"); err != nil {
		return fmt.Errorf("schema validation: %w", err)
	}
	return nil
}

// validateNumericStrings 补充标准 JSON Schema 对字符串实例不生效的数值边界。
// 只处理生成器显式标记为 json:",string" 数值的字段；普通字符串 Schema 上的
// minimum/maximum 仍按标准仅作为 number 约束。组合分支、prefixItems 与
// $anchor 引用必须继续传播该标记，不能因递归边界而静默放行。
func (v *Validator) validateNumericStrings(schema, instance any, path string) error {
	return v.validateNumericStringsDepth(schema, instance, path, 0, make(map[string]bool))
}

func (v *Validator) validateNumericStringsDepth(schema, instance any, path string, depth int, seen map[string]bool) error {
	if depth > 128 {
		return fmt.Errorf("at %s: schema recursion too deep", path)
	}
	s, ok := schema.(map[string]any)
	if !ok {
		return nil
	}
	if ref, ok := s["$ref"].(string); ok {
		target, found := v.resolveRef(ref)
		if !found {
			return fmt.Errorf("at %s: unresolved schema reference %q", path, ref)
		}
		// $ref 在 2020-12 允许与兄弟关键字同时出现：先校验引用目标，
		// 再继续处理本层兄弟约束；seen 只拦截当前递归链上的自引用。
		if !seen[ref] {
			seen[ref] = true
			err := v.validateNumericStringsDepth(target, instance, path, depth+1, seen)
			delete(seen, ref)
			if err != nil {
				return err
			}
		}
	}
	if ref, ok := s["$dynamicRef"].(string); ok {
		// 本地 $dynamicRef 先按静态锚点解析；无法静态解析时交给标准验证器，
		// 不在这里猜测动态作用域，也不误拒绝整体 Schema。
		if target, found := v.resolveRef(ref); found && !seen[ref] {
			seen[ref] = true
			err := v.validateNumericStringsDepth(target, instance, path, depth+1, seen)
			delete(seen, ref)
			if err != nil {
				return err
			}
		}
	}
	if branches, ok := s["allOf"].([]any); ok {
		for _, branch := range branches {
			if err := v.validateNumericStringsDepth(branch, instance, path, depth+1, seen); err != nil {
				return err
			}
		}
	}
	for _, keyword := range []string{"anyOf", "oneOf"} {
		branches, ok := s[keyword].([]any)
		if !ok {
			continue
		}
		// 只对标准验证器实际匹配的分支执行自定义字符串数值约束。
		matched := make([]any, 0, len(branches))
		for _, branch := range branches {
			if v.branchMatches(branch, instance) {
				matched = append(matched, branch)
			}
		}
		switch keyword {
		case "anyOf":
			// 任一匹配分支通过自定义约束即可满足 anyOf；全部失败才返回错误。
			if len(matched) == 0 {
				continue
			}
			var firstErr error
			passed := false
			for _, branch := range matched {
				if err := v.validateNumericStringsDepth(branch, instance, path, depth+1, seen); err != nil {
					if firstErr == nil {
						firstErr = err
					}
					continue
				}
				passed = true
				break
			}
			if !passed {
				return firstErr
			}
		case "oneOf":
			// 标准 oneOf 已保证只有一个匹配；该分支仍须通过自定义约束。
			if len(matched) == 1 {
				if err := v.validateNumericStringsDepth(matched[0], instance, path, depth+1, seen); err != nil {
					return err
				}
			}
		}
	}
	if condition, ok := s["if"]; ok {
		if matched := v.branchMatches(condition, instance); matched {
			if thenSchema, exists := s["then"]; exists {
				if err := v.validateNumericStringsDepth(thenSchema, instance, path, depth+1, seen); err != nil {
					return err
				}
			}
		} else if elseSchema, exists := s["else"]; exists {
			if err := v.validateNumericStringsDepth(elseSchema, instance, path, depth+1, seen); err != nil {
				return err
			}
		}
	}
	if str, ok := instance.(string); ok && hasNumericStringMarker(s) {
		if err := v.checkStringNumber(s, str, path); err != nil {
			return err
		}
	}
	switch value := instance.(type) {
	case map[string]any:
		properties, _ := s["properties"].(map[string]any)
		for name, property := range properties {
			if item, exists := value[name]; exists {
				if err := v.validateNumericStringsDepth(property, item, path+"/"+name, depth+1, make(map[string]bool)); err != nil {
					return err
				}
			}
		}
		if additional, ok := s["additionalProperties"].(map[string]any); ok {
			for name, item := range value {
				if _, declared := properties[name]; declared {
					continue
				}
				if err := v.validateNumericStringsDepth(additional, item, path+"/"+name, depth+1, make(map[string]bool)); err != nil {
					return err
				}
			}
		}
	case []any:
		prefix, _ := s["prefixItems"].([]any)
		for index, item := range value {
			itemPath := fmt.Sprintf("%s/%d", path, index)
			if index < len(prefix) {
				if err := v.validateNumericStringsDepth(prefix[index], item, itemPath, depth+1, make(map[string]bool)); err != nil {
					return err
				}
				continue
			}
			switch items := s["items"].(type) {
			case map[string]any:
				if err := v.validateNumericStringsDepth(items, item, itemPath, depth+1, make(map[string]bool)); err != nil {
					return err
				}
			case []any:
				if index < len(items) {
					if err := v.validateNumericStringsDepth(items[index], item, itemPath, depth+1, make(map[string]bool)); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func (v *Validator) checkStringNumber(schema map[string]any, value, path string) error {
	minimum, hasMinimum := numericBound(schema["minimum"])
	maximum, hasMaximum := numericBound(schema["maximum"])
	if !hasMinimum && !hasMaximum {
		return nil
	}
	number, err := parseNumberString(value)
	if err != nil {
		return fmt.Errorf("at %s: %w", path, err)
	}
	if hasMinimum && number.Cmp(minimum) < 0 {
		return fmt.Errorf("at %s: string-encoded number %q is below minimum %s", path, value, minimum.RatString())
	}
	if hasMaximum && number.Cmp(maximum) > 0 {
		return fmt.Errorf("at %s: string-encoded number %q exceeds maximum %s", path, value, maximum.RatString())
	}
	return nil
}

// branchMatches 用标准验证器判断 anyOf/oneOf 分支是否真正匹配实例。
// 无法独立编译的分支（例如依赖外部文档）不执行自定义标记，避免误拒绝；
// 整体 Schema 仍已由标准验证器校验，不会因此放宽标准约束。
func (v *Validator) branchMatches(branch, instance any) bool {
	data, err := json.Marshal(branch)
	if err != nil {
		return false
	}
	key := string(data)
	if cached, ok := v.branchCache.Load(key); ok {
		return cached.(*jsonschema.Schema).Validate(instance) == nil
	}
	compiled, err := compileStandardSchema(branch)
	if err != nil {
		if rootCompiled, ok := v.compileBranchWithRoot(branch); ok {
			compiled = rootCompiled
			err = nil
		} else if root, ok := v.document.(map[string]any); ok {
			// 含 $id 的文档跳过根副本重写，但仍可解析纯 $defs 引用。
			if defs, exists := root["$defs"]; exists {
				compiled, err = compileStandardSchema(map[string]any{"$defs": defs, "allOf": []any{branch}})
			}
		}
	}
	if err != nil {
		return false
	}
	v.branchCache.Store(key, compiled)
	return compiled.Validate(instance) == nil
}

// compileBranchWithRoot 在分支无法独立编译时，把完整根文档挂到合成
// $defs 下，并把分支中的本地 JSON Pointer 引用重写到该副本，从而让
// #/properties/... 等非 $defs 引用仍能参与分支匹配。含 $id 的文档会
// 改变引用基址，保守跳过而不是猜测。
func (v *Validator) compileBranchWithRoot(branch any) (*jsonschema.Schema, bool) {
	root, ok := v.document.(map[string]any)
	if !ok || schemaHasID(root) || schemaHasID(branch) {
		return nil, false
	}
	rewritten, ok := rewriteLocalRefs(branch, "#/$defs/__hexagon_root__")
	if !ok {
		return nil, false
	}
	compiled, err := compileStandardSchema(map[string]any{
		"$defs": map[string]any{"__hexagon_root__": root},
		"allOf": []any{rewritten},
	})
	if err != nil {
		return nil, false
	}
	return compiled, true
}

func schemaDataKeyword(key string) bool {
	switch key {
	case "const", "enum", "default", "examples", "example", "metadata", "discriminator", "xml", "externalDocs":
		return true
	default:
		return false
	}
}

func schemaHasID(value any) bool {
	switch node := value.(type) {
	case map[string]any:
		if _, ok := node["$id"]; ok {
			return true
		}
		for _, item := range node {
			if schemaHasID(item) {
				return true
			}
		}
	case []any:
		for _, item := range node {
			if schemaHasID(item) {
				return true
			}
		}
	}
	return false
}

// rewriteLocalRefs 只改写 # 与 #/...；#anchor 仍由同一资源内的锚点解析。
func rewriteLocalRefs(value any, prefix string) (any, bool) {
	switch node := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(node))
		for key, item := range node {
			if key == "$ref" {
				ref, ok := item.(string)
				if !ok {
					return nil, false
				}
				switch {
				case ref == "#":
					out[key] = prefix
				case strings.HasPrefix(ref, "#/"):
					out[key] = prefix + strings.TrimPrefix(ref, "#")
				default:
					out[key] = ref
				}
				continue
			}
			// 注解/数据关键字不是子 Schema，不能把其中的 $ref 当引用改写。
			if schemaDataKeyword(key) {
				out[key] = item
				continue
			}
			rewritten, ok := rewriteLocalRefs(item, prefix)
			if !ok {
				return nil, false
			}
			out[key] = rewritten
		}
		return out, true
	case []any:
		out := make([]any, len(node))
		for index, item := range node {
			rewritten, ok := rewriteLocalRefs(item, prefix)
			if !ok {
				return nil, false
			}
			out[index] = rewritten
		}
		return out, true
	default:
		return value, true
	}
}

func (v *Validator) resolveRef(ref string) (any, bool) {
	if ref == "#" {
		return v.document, v.document != nil
	}
	if strings.HasPrefix(ref, "#/") {
		current := v.document
		for _, token := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
			token = strings.ReplaceAll(strings.ReplaceAll(token, "~1", "/"), "~0", "~")
			object, ok := current.(map[string]any)
			if !ok {
				return nil, false
			}
			current, ok = object[token]
			if !ok {
				return nil, false
			}
		}
		return current, true
	}
	if strings.HasPrefix(ref, "#") {
		return findAnchor(v.document, strings.TrimPrefix(ref, "#"))
	}
	return nil, false
}

// findAnchor 在完整文档中定位 $anchor。只用于本地、已由标准验证器解析的引用，
// 不发起网络读取；重复 anchor 属于非法 Schema，按排序后的确定性顺序返回第一个。
func findAnchor(value any, name string) (any, bool) {
	if name == "" {
		return nil, false
	}
	switch node := value.(type) {
	case map[string]any:
		if anchor, ok := node["$anchor"].(string); ok && anchor == name {
			return node, true
		}
		keys := make([]string, 0, len(node))
		for key := range node {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if found, ok := findAnchor(node[key], name); ok {
				return found, true
			}
		}
	case []any:
		for _, child := range node {
			if found, ok := findAnchor(child, name); ok {
				return found, true
			}
		}
	}
	return nil, false
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
	// encoding/json 的整数字符串允许前导零，不能直接按裸 JSON 数字拒绝。
	digits := strings.TrimPrefix(value, "-")
	if digits != "" && strings.IndexFunc(digits, func(r rune) bool { return r < '0' || r > '9' }) < 0 {
		if integer, ok := new(big.Int).SetString(value, 10); ok {
			return new(big.Rat).SetInt(integer), nil
		}
	}
	decoder := json.NewDecoder(strings.NewReader(value))
	decoder.UseNumber()
	var parsed any
	if err := decoder.Decode(&parsed); err != nil {
		return nil, fmt.Errorf("string-encoded number %q is not a JSON number", value)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("string-encoded number %q is not a single JSON number", value)
	}
	number, ok := parsed.(json.Number)
	if !ok {
		return nil, fmt.Errorf("string-encoded value %q is not a JSON number", value)
	}
	bound, ok := new(big.Rat).SetString(number.String())
	if !ok {
		return nil, fmt.Errorf("string-encoded number %q cannot be compared", value)
	}
	return bound, nil
}
