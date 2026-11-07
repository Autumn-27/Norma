package tool

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"unicode/utf8"
)

// ValidateInput checks a raw JSON input against a minimal subset of JSON Schema
// sufficient for tool inputs: types, required properties, enum, inclusive numeric
// bounds, array lengths and string lengths. It never
// panics; a non-nil error means the input is invalid (FR-04.5, NFR-07).
func ValidateInput(schema map[string]any, raw json.RawMessage) error {
	if schema == nil {
		return nil
	}
	// Accept both Go-authored schemas ([]string, int, typed maps) and schemas
	// decoded from JSON ([]any, float64). Compare JSON values consistently.
	encoded, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("invalid input schema: %w", err)
	}
	var normalized map[string]any
	if err := json.Unmarshal(encoded, &normalized); err != nil {
		return fmt.Errorf("invalid input schema: %w", err)
	}
	var v any
	if len(raw) == 0 {
		v = map[string]any{}
	} else if err := json.Unmarshal(raw, &v); err != nil {
		return fmt.Errorf("input is not valid JSON: %w", err)
	}
	return validateValue(normalized, v, "")
}

func validateValue(schema map[string]any, v any, path string) error {
	if values, ok := schema["enum"].([]any); ok {
		found := false
		for _, value := range values {
			if reflect.DeepEqual(value, v) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%s: value is not in enum", at(path))
		}
	}
	typ, _ := schema["type"].(string)
	switch typ {
	case "object", "":
		m, ok := v.(map[string]any)
		if !ok {
			if typ == "" {
				break
			}
			return fmt.Errorf("%s: expected object", at(path))
		}
		for _, req := range toStrings(schema["required"]) {
			if _, present := m[req]; !present {
				return fmt.Errorf("%s: missing required property %q", at(path), req)
			}
		}
		props, _ := schema["properties"].(map[string]any)
		for key, val := range m {
			ps, ok := props[key].(map[string]any)
			if !ok {
				continue // unknown property — tolerate
			}
			if err := validateValue(ps, val, join(path, key)); err != nil {
				return err
			}
		}
	case "string":
		if _, ok := v.(string); !ok {
			return fmt.Errorf("%s: expected string", at(path))
		}
	case "integer":
		if f, ok := v.(float64); !ok || math.Trunc(f) != f {
			return fmt.Errorf("%s: expected integer", at(path))
		}
	case "number":
		if _, ok := v.(float64); !ok {
			return fmt.Errorf("%s: expected number", at(path))
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return fmt.Errorf("%s: expected boolean", at(path))
		}
	case "array":
		arr, ok := v.([]any)
		if !ok {
			return fmt.Errorf("%s: expected array", at(path))
		}
		if err := validateBounds(schema, "minItems", "maxItems", float64(len(arr)), path); err != nil {
			return err
		}
		if items, ok := schema["items"].(map[string]any); ok {
			for i, el := range arr {
				if err := validateValue(items, el, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	}
	if f, ok := v.(float64); ok {
		if err := validateBounds(schema, "minimum", "maximum", f, path); err != nil {
			return err
		}
	}
	if s, ok := v.(string); ok {
		return validateBounds(schema, "minLength", "maxLength", float64(utf8.RuneCountInString(s)), path)
	}
	return nil
}

func validateBounds(schema map[string]any, minKey, maxKey string, value float64, path string) error {
	if bound, ok := schema[minKey].(float64); ok && value < bound {
		return fmt.Errorf("%s: %s is %g", at(path), minKey, bound)
	}
	if bound, ok := schema[maxKey].(float64); ok && value > bound {
		return fmt.Errorf("%s: %s is %g", at(path), maxKey, bound)
	}
	return nil
}

func toStrings(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, e := range arr {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func join(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

func at(path string) string {
	if path == "" {
		return "input"
	}
	return strings.TrimPrefix(path, ".")
}
