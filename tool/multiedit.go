package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// NewMultiEdit builds the MultiEdit tool: applies a sequence of exact-string
// replacements to one file atomically. Edits apply in order to the in-memory
// content; if any fails to match, nothing is written.
func NewMultiEdit() CoreTool {
	return Build(Spec{
		Name:        "MultiEdit",
		Description: "Applies several exact string replacements to a single file in one atomic operation. Edits are applied in order; if any old_string fails to match (or isn't unique without replace_all), the whole operation aborts and the file is left unchanged.",
		Prompt:      "Use this instead of multiple Edit calls on the same file. Each edit's old_string must match the file as modified by the preceding edits. Read the file first.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path": map[string]any{"type": "string", "description": "Path of the file to edit."},
				"edits": map[string]any{
					"type":        "array",
					"description": "Edits to apply in order.",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"old_string":  map[string]any{"type": "string", "description": "Exact text to replace."},
							"new_string":  map[string]any{"type": "string", "description": "Replacement text."},
							"replace_all": map[string]any{"type": "boolean", "description": "Replace all occurrences."},
						},
						"required": []any{"old_string", "new_string"},
					},
				},
			},
			"required": []any{"file_path", "edits"},
		},
		Run: runMultiEdit,
	})
}

func runMultiEdit(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath string `json:"file_path"`
		Edits    []struct {
			OldString  string `json:"old_string"`
			NewString  string `json:"new_string"`
			ReplaceAll bool   `json:"replace_all"`
		} `json:"edits"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	if len(in.Edits) == 0 {
		return Errorf("Error: no edits provided"), nil
	}
	path := resolvePath(tc, in.FilePath)
	data, err := os.ReadFile(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	content := string(data)
	total := 0
	for i, e := range in.Edits {
		if e.OldString == e.NewString {
			return Errorf(fmt.Sprintf("Error in edit %d: old_string and new_string are identical", i+1)), nil
		}
		count := strings.Count(content, e.OldString)
		if count == 0 {
			return Errorf(fmt.Sprintf("Error in edit %d: old_string not found", i+1)), nil
		}
		if count > 1 && !e.ReplaceAll {
			return Errorf(fmt.Sprintf("Error in edit %d: old_string is not unique (%d matches); add context or set replace_all", i+1, count)), nil
		}
		if e.ReplaceAll {
			content = strings.ReplaceAll(content, e.OldString, e.NewString)
		} else {
			content = strings.Replace(content, e.OldString, e.NewString, 1)
		}
		total += count
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return Errorf(fmt.Sprintf("Error writing %s: %v", in.FilePath, err)), nil
	}
	return Text(fmt.Sprintf("Applied %d edits to %s (%d replacement(s))", len(in.Edits), in.FilePath, total)), nil
}
