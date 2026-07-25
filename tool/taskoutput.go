package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// NewTaskOutput builds the TaskOutput tool: read the captured output of a
// background task (from Bash run_in_background or Monitor), optionally blocking
// until it finishes.
func NewTaskOutput() CoreTool {
	return Build(Spec{
		Name:        "TaskOutput",
		Description: "Reads the output of a background task started by Bash (run_in_background) or Monitor. You can also Read the task's output-file path directly. Set block=true to wait for the task to finish first.",
		Prompt:      "Provide the task_id. With block=true the call waits until the task reaches a terminal state (up to timeout_ms) before returning its output.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"task_id":    map[string]any{"type": "string", "description": "The task id (e.g. task_1)."},
				"max_bytes":  map[string]any{"type": "integer", "description": "Max bytes of output to return from the tail (default 30000)."},
				"block":      map[string]any{"type": "boolean", "description": "Wait until the task finishes before returning (default false)."},
				"timeout_ms": map[string]any{"type": "integer", "description": "Max wait when block=true (default 30000)."},
			},
			"required": []any{"task_id"},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runTaskOutput,
	})
}

func runTaskOutput(ctx context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		TaskID    string `json:"task_id"`
		MaxBytes  int    `json:"max_bytes"`
		Block     bool   `json:"block"`
		TimeoutMS int    `json:"timeout_ms"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	if tc == nil || tc.Tasks == nil {
		return Errorf("Error: background tasks are not enabled"), nil
	}
	if in.TaskID == "" {
		return Errorf("Error: task_id is required"), nil
	}
	info, ok := tc.Tasks.Get(in.TaskID)
	if !ok {
		return Errorf(fmt.Sprintf("Error: unknown task %q", in.TaskID)), nil
	}
	if in.Block && !info.Status.Terminal() {
		timeout := 30 * time.Second
		if in.TimeoutMS > 0 {
			timeout = time.Duration(in.TimeoutMS) * time.Millisecond
		}
		info, _ = tc.Tasks.Wait(ctx, in.TaskID, timeout)
	}
	out, err := tc.Tasks.Output(in.TaskID, in.MaxBytes)
	if err != nil {
		return Errorf("Error: " + err.Error()), nil
	}
	header := fmt.Sprintf("[task %s | %s", info.ID, info.Status)
	if info.Status == TaskCompleted || info.Status == TaskFailed {
		header += fmt.Sprintf(" | exit %d", info.ExitCode)
	}
	header += "]\n"
	return Text(Capture(tc, header+out)), nil
}
