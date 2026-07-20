package tool

import (
	"context"
	"encoding/json"
	"fmt"
)

// NewTaskStop builds the TaskStop tool: kill a running background task (or all).
func NewTaskStop() CoreTool {
	return Build(Spec{
		Name:        "TaskStop",
		Description: "Stops a running background task, killing its process tree. Pass a task_id, or omit it to stop all running tasks.",
		Prompt:      "Provide task_id to stop one task, or leave it empty to stop all running tasks.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"task_id": map[string]any{"type": "string", "description": "The id of the task to stop. Empty stops all running tasks."},
			},
		},
		// Stopping a process the agent itself launched is a safe control op.
		Permissions: allowReadOnly,
		Run:         runTaskStop,
	})
}

func runTaskStop(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		TaskID string `json:"task_id"`
	}
	_ = json.Unmarshal(input, &in)
	if tc == nil || tc.Tasks == nil {
		return Errorf("Error: background tasks are not enabled"), nil
	}
	if in.TaskID == "" {
		n := tc.Tasks.KillAll()
		return Text(fmt.Sprintf("Stopped %d running task(s).", n)), nil
	}
	if tc.Tasks.Kill(in.TaskID) {
		return Text(fmt.Sprintf("Stopped task %s.", in.TaskID)), nil
	}
	return Errorf(fmt.Sprintf("Error: unknown task %q", in.TaskID)), nil
}
