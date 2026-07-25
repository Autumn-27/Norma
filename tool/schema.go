package tool

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ValidateInput checks a raw JSON input against a minimal subset of JSON Schema
// sufficient for tool inputs: object type, required properties, and per-property
// primitive type checks (string/number/integer/boolean/array/object). It never
// panics; a non-nil error means the input is invalid (FR-04.5, NFR-07).
func ValidateInput(schema map[string]any, raw json.RawMessage) error {
	if schema == nil {
		return nil
	}
	var v any
	if len(raw) == 0 {
		v = map[string]any{}
	} else if err := json.Unmarshal(raw, &v); err != nil {
		return fmt.Errorf("input is not valid JSON: %w", err)
	}
	return validateValue(schema, v, "")
}

func validateValue(schema map[string]any, v any, path string) error {
	typ, _ := schema["type"].(string)
	switch typ {
	case "object", "":
		m, ok := v.(map[string]any)
		if !ok {
			if typ == "" {
				return nil
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
		if f, ok := v.(float64); !ok || f != float64(int64(f)) {
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
		if items, ok := schema["items"].(map[string]any); ok {
			for i, el := range arr {
				if err := validateValue(items, el, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
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
