package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// NewTaskList builds the TaskList tool: enumerate this session's background tasks.
func NewTaskList() CoreTool {
	return Build(Spec{
		Name:        "TaskList",
		Description: "Lists the background tasks started in this session with their status, exit code, and command.",
		Schema:      map[string]any{"type": "object", "properties": map[string]any{}},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runTaskList,
	})
}

func runTaskList(_ context.Context, _ json.RawMessage, tc *ToolContext) (Result, error) {
	if tc == nil || tc.Tasks == nil {
		return Errorf("Error: background tasks are not enabled"), nil
	}
	tasks := tc.Tasks.List()
	if len(tasks) == 0 {
		return Text("No background tasks."), nil
	}
	var b strings.Builder
	for _, t := range tasks {
		fmt.Fprintf(&b, "%s [%s] %s", t.ID, t.Kind, t.Status)
		if t.Status == TaskCompleted || t.Status == TaskFailed {
			fmt.Fprintf(&b, " exit=%d", t.ExitCode)
		}
		fmt.Fprintf(&b, "  %s\n", short(t.Command))
	}
	return Text(b.String()), nil
}
