package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// NewRead builds the Read tool: reads a file with cat -n style line numbers and
// optional offset/limit.
func NewRead() CoreTool {
	return Build(Spec{
		Name:        "Read",
		Description: "Reads a file from the local filesystem and returns its contents with line numbers. Use offset/limit for large files. Always read a file before editing it.",
		Prompt:      "file_path may be absolute or relative to the working directory. Output lines are prefixed with their 1-based line number.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path": map[string]any{"type": "string", "description": "Path to the file to read."},
				"offset":    map[string]any{"type": "integer", "description": "1-based line to start from."},
				"limit":     map[string]any{"type": "integer", "description": "Max lines to read (default 2000)."},
			},
			"required": []any{"file_path"},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runRead,
	})
}

const maxLineLen = 2000

func runRead(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath string `json:"file_path"`
		Offset   int    `json:"offset"`
		Limit    int    `json:"limit"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	path := resolvePath(tc, in.FilePath)
	info, err := os.Stat(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	if info.IsDir() {
		return Errorf(fmt.Sprintf("Error: %s is a directory, not a file. Use LS or Glob to list its contents.", in.FilePath)), nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	if isBinary(data) {
		return Text(fmt.Sprintf("[binary file %s not shown — %d bytes]", in.FilePath, len(data))), nil
	}
	lines := strings.Split(string(data), "\n")
	start := 0
	if in.Offset > 0 {
		start = in.Offset - 1
	}
	if start > len(lines) {
		start = len(lines)
	}
	limit := in.Limit
	if limit <= 0 {
		limit = 2000
	}
	end := start + limit
	if end > len(lines) {
		end = len(lines)
	}
	var b strings.Builder
	for i := start; i < end; i++ {
		line := lines[i]
		if len(line) > maxLineLen {
			line = line[:maxLineLen] + "… [line truncated]"
		}
		fmt.Fprintf(&b, "%6d\t%s\n", i+1, line)
	}
	if b.Len() == 0 {
		if len(data) == 0 {
			return Text("(file is empty)"), nil
		}
		return Text("(no lines in the requested range)"), nil
	}
	return Text(readCapture(tc, b.String(), start)), nil
}

// readOutputBonus lifts the Read tool's output budget above the shared default
// (ToolContext.MaxOutputChars): Read gets a little more room before truncating.
const readOutputBonus = 1000

// readCapture bounds Read's line-numbered output. When it exceeds the Read budget
// (MaxToolOutputChars + readOutputBonus) it cuts at a line boundary and tells the
// model the exact offset to continue from — the source file is already on disk, so
// there is no need to spill a copy: the model just pages on with Read(offset=…).
// start is the 0-based line the output began at.
func readCapture(tc *ToolContext, s string, start int) string {
	max := maxOut(tc) + readOutputBonus
	if len(s) <= max {
		return s
	}
	head, headLines := headWithinBudget(s, max)
	nextOffset := start + headLines + 1 // 1-based file line to continue from
	return head + fmt.Sprintf(
		"\n... [Read truncated to fit the output limit. Continue reading the file with Read offset=%d (and an optional limit), or narrow with Grep/`sed -n`.] ...",
		nextOffset)
}

// headWithinBudget returns the largest whole-line prefix of s within max bytes and
// the number of lines it contains. Falls back to a hard cut if a single line
// already exceeds the budget (no newline within reach).
func headWithinBudget(s string, max int) (string, int) {
	if len(s) <= max {
		return s, strings.Count(s, "\n")
	}
	cut := strings.LastIndexByte(s[:max], '\n')
	if cut < 0 {
		return s[:max], 0
	}
	head := s[:cut+1]
	return head, strings.Count(head, "\n")
}

// isBinary reports whether data looks like a binary (non-text) file, using the
// common heuristic of a NUL byte in the first chunk.
func isBinary(data []byte) bool {
	n := len(data)
	if n > 8000 {
		n = 8000
	}
	for i := 0; i < n; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}

// NewWrite builds the Write tool: creates or overwrites a file.
func NewWrite() CoreTool {
	return Build(Spec{
		Name:        "Write",
		Description: "Writes a file to the local filesystem, creating parent directories and overwriting any existing file. Prefer Edit for modifying existing files.",
		Prompt:      "Provide the complete file contents — Write replaces the whole file.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path": map[string]any{"type": "string", "description": "Path of the file to write."},
				"content":   map[string]any{"type": "string", "description": "Full contents to write."},
			},
			"required": []any{"file_path", "content"},
		},
		Run: runWrite,
	})
}

func runWrite(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	path := resolvePath(tc, in.FilePath)
	existed := false
	if info, err := os.Stat(path); err == nil {
		if info.IsDir() {
			return Errorf(fmt.Sprintf("Error: %s is an existing directory, not a file.", in.FilePath)), nil
		}
		existed = true
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Errorf(fmt.Sprintf("Error creating directory: %v", err)), nil
		}
	}
	if err := os.WriteFile(path, []byte(in.Content), 0o644); err != nil {
		return Errorf(fmt.Sprintf("Error writing %s: %v", in.FilePath, err)), nil
	}
	verb := "Created"
	if existed {
		verb = "Overwrote"
	}
	return Text(fmt.Sprintf("%s %s (%d bytes)", verb, in.FilePath, len(in.Content))), nil
}

// NewEdit builds the Edit tool: exact string replacement with uniqueness check.
func NewEdit() CoreTool {
	return Build(Spec{
		Name:        "Edit",
		Description: "Performs an exact string replacement in a file. old_string must match exactly once unless replace_all is set. Read the file first so old_string matches precisely.",
		Prompt:      "old_string must match the file exactly (whitespace included) and be unique unless replace_all is true. new_string must differ from old_string.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path":   map[string]any{"type": "string", "description": "Path of the file to edit."},
				"old_string":  map[string]any{"type": "string", "description": "Exact text to replace."},
				"new_string":  map[string]any{"type": "string", "description": "Replacement text."},
				"replace_all": map[string]any{"type": "boolean", "description": "Replace all occurrences."},
			},
			"required": []any{"file_path", "old_string", "new_string"},
		},
		Run: runEdit,
	})
}

func runEdit(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath   string `json:"file_path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	if in.OldString == in.NewString {
		return Errorf("Error: old_string and new_string are identical"), nil
	}
	path := resolvePath(tc, in.FilePath)
	data, err := os.ReadFile(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	content := string(data)
	count := strings.Count(content, in.OldString)
	if count == 0 {
		return Errorf(fmt.Sprintf("Error: old_string not found in %s", in.FilePath)), nil
	}
	if count > 1 && !in.ReplaceAll {
		return Errorf(fmt.Sprintf("Error: old_string is not unique in %s (%d matches); add context or set replace_all", in.FilePath, count)), nil
	}
	var updated string
	if in.ReplaceAll {
		updated = strings.ReplaceAll(content, in.OldString, in.NewString)
	} else {
		updated = strings.Replace(content, in.OldString, in.NewString, 1)
	}
	if err := os.WriteFile(path, []byte(updated), 0o644); err != nil {
		return Errorf(fmt.Sprintf("Error writing %s: %v", in.FilePath, err)), nil
	}
	return Text(fmt.Sprintf("Edited %s (%d replacement(s))", in.FilePath, count)), nil
}
