package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// NewLS builds the LS tool: lists the entries of a directory.
func NewLS() CoreTool {
	return Build(Spec{
		Name:        "LS",
		Description: "Lists the files and subdirectories of a directory (non-recursive). Use Glob to match by pattern across depth, or Grep to search contents.",
		Prompt:      "path may be absolute or relative to the working directory; defaults to the working directory.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "description": "Directory to list (default working directory)."},
			},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runLS,
	})
}

func runLS(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	dir := tc.WorkingDir
	if in.Path != "" {
		dir = resolvePath(tc, in.Path)
	}
	if dir == "" {
		dir = "."
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return Errorf(fmt.Sprintf("Error listing %s: %v", in.Path, err)), nil
	}
	if len(entries) == 0 {
		return Text("(empty directory)"), nil
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].IsDir() != entries[j].IsDir() {
			return entries[i].IsDir() // directories first
		}
		return entries[i].Name() < entries[j].Name()
	})
	var b strings.Builder
	for _, e := range entries {
		if e.IsDir() {
			fmt.Fprintf(&b, "%s/\n", e.Name())
			continue
		}
		var size int64
		if info, err := e.Info(); err == nil {
			size = info.Size()
		}
		fmt.Fprintf(&b, "%s\t%d bytes\n", e.Name(), size)
	}
	return Text(Capture(tc, b.String())), nil
}
