package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// NewMonitor builds the Monitor tool: start a long-running streaming process in
// the background (tail -f, watch, a dev server, a polling loop) and return
// immediately. Inspect it with Read/TaskOutput and end it with TaskStop.
func NewMonitor() CoreTool {
	return Build(Spec{
		Name:        "Monitor",
		Description: "Starts a long-running streaming process in the background (e.g. tail -f, watch, a dev server, a polling loop) and returns immediately with a task id and output-file path. Use Read or TaskOutput to inspect it and TaskStop to end it.",
		Prompt:      "Provide a command that streams output over time. It always runs in the background; you are notified if it exits on its own.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command":     map[string]any{"type": "string", "description": "The streaming command to run."},
				"description": map[string]any{"type": "string", "description": "Short description of what is being monitored."},
			},
			"required": []any{"command"},
		},
		// Same catastrophic-command safety floor as Bash.
		Permissions: bashPermissions,
		Run:         runMonitor,
	})
}

func runMonitor(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		Command     string `json:"command"`
		Description string `json:"description"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	if strings.TrimSpace(in.Command) == "" {
		return Errorf("Error: command is required"), nil
	}
	if tc == nil || tc.Tasks == nil {
		return Errorf("Error: background tasks are not enabled"), nil
	}
	t, err := tc.Tasks.Spawn(SpawnSpec{
		Command:     in.Command,
		Description: in.Description,
		Kind:        KindMonitor,
		WorkingDir:  tc.WorkingDir,
	})
	if err != nil {
		return Errorf("Error: failed to start monitor: " + err.Error()), nil
	}
	tc.Tasks.Background(t.ID)
	return Text(fmt.Sprintf(
		"Started monitor %s. Output file: %s\nUse TaskOutput{task_id:%q} or Read to inspect it; TaskStop{task_id:%q} to end it.",
		t.ID, t.OutputPath, t.ID, t.ID)), nil
}
