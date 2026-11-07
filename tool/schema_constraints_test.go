package tool

import (
	"encoding/json"
	"sync"
	"testing"
)

func TestValidateInputDoesNotMutateSharedSchema(t *testing.T) {
	schema := map[string]any{"type": "object", "required": []string{"x"}}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := ValidateInput(schema, []byte(`{"x":1}`)); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if _, ok := schema["required"].([]string); !ok {
		t.Fatal("caller schema was mutated")
	}
}

func TestSchemaConstraints(t *testing.T) {
	for _, test := range []struct {
		name   string
		schema map[string]any
		input  string
		valid  bool
	}{
		{"Go required missing", map[string]any{"type": "object", "required": []string{"x"}}, `{}`, false},
		{"Go required present", map[string]any{"type": "object", "required": []string{"x"}}, `{"x":0}`, true},
		{"JSON required missing", map[string]any{"type": "object", "required": []any{"x"}}, `{}`, false},
		{"enum rejects", map[string]any{"type": "string", "enum": []string{"pending", "completed"}}, `"invalid"`, false},
		{"enum accepts", map[string]any{"type": "string", "enum": []string{"pending", "completed"}}, `"pending"`, true},
		{"numeric enum", map[string]any{"enum": []any{1, 2}}, `1.0`, true},
		{"object enum", map[string]any{"enum": []any{map[string]any{"x": 1}}}, `{"x":1}`, true},
		{"minimum inclusive", map[string]any{"type": "integer", "minimum": 1, "maximum": 20}, `1`, true},
		{"maximum inclusive", map[string]any{"type": "integer", "minimum": 1, "maximum": 20}, `20`, true},
		{"below minimum", map[string]any{"type": "integer", "minimum": 1}, `0`, false},
		{"above maximum", map[string]any{"type": "number", "maximum": 20}, `20.5`, false},
		{"fractional integer", map[string]any{"type": "integer"}, `1.5`, false},
		{"large integer", map[string]any{"type": "integer"}, `1e20`, true},
		{"minItems", map[string]any{"type": "array", "minItems": 2}, `[1]`, false},
		{"maxItems", map[string]any{"type": "array", "maxItems": 2}, `[1,2,3]`, false},
		{"array boundary", map[string]any{"type": "array", "minItems": 2, "maxItems": 2}, `[1,2]`, true},
		{"unicode length", map[string]any{"type": "string", "minLength": 2, "maxLength": 2}, `"你好"`, true},
		{"too long", map[string]any{"type": "string", "maxLength": 2}, `"你好啊"`, false},
		{"too short", map[string]any{"type": "string", "minLength": 1}, `""`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Both Go-authored and wire-decoded schemas must behave identically.
			b, _ := json.Marshal(test.schema)
			var decoded map[string]any
			if err := json.Unmarshal(b, &decoded); err != nil {
				t.Fatal(err)
			}
			for _, schema := range []map[string]any{test.schema, decoded} {
				err := ValidateInput(schema, json.RawMessage(test.input))
				if (err == nil) != test.valid {
					t.Fatalf("valid=%v err=%v", test.valid, err)
				}
			}
		})
	}
}

func TestBuiltinSchemaRejectsInvalidArguments(t *testing.T) {
	search, err := NewWebSearch(WebSearchConfig{})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		tool  CoreTool
		input string
	}{
		{NewSleep(), `{}`},
		{NewTodoStore().Tool(), `{"todos":[{"content":"x","status":"bogus"}]}`},
		{search, `{"query":"x","limit":21}`},
		{NewAskUserQuestion(nil), `{"questions":[]}`},
		{NewAskUserQuestion(nil), `{"questions":[{"question":"x","options":[{"label":"a"}]}]}`},
		{NewMultiEdit(), `{"file_path":"x","edits":[]}`},
	} {
		if err := ValidateInput(test.tool.InputSchema(), []byte(test.input)); err == nil {
			t.Errorf("%s accepted %s", test.tool.Name(), test.input)
		}
	}
}
