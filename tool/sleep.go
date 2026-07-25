package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// NewSleep returns a tool that waits N seconds, interruptible via context. It pairs
// with background task execution: after spawning long-running work (e.g. a Bash
// command with run_in_background), an agent can sleep, then read partial output with
// TaskOutput. Part of DefaultTools — every agent gets it.
//
// The wait is cancelled the moment ctx is done (pause/stop/timeout), returning
// promptly rather than blocking out the deadline.
func NewSleep() CoreTool {
	return Build(Spec{
		Name:        "sleep",
		Description: "Waits for N seconds, then returns. Use to let a backgrounded command make progress before reading its partial output with TaskOutput. Interrupted immediately if the run is paused or stopped.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"seconds": map[string]any{"type": "integer", "description": "Seconds to wait, 1-3600."},
			},
			"required": []string{"seconds"},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runSleep,
	})
}

func runSleep(ctx context.Context, in json.RawMessage, _ *ToolContext) (Result, error) {
	var a struct {
		Seconds int `json:"seconds"`
	}
	_ = json.Unmarshal(in, &a)
	if a.Seconds < 1 {
		a.Seconds = 1
	}
	if a.Seconds > 3600 {
		a.Seconds = 3600
	}
	select {
	case <-time.After(time.Duration(a.Seconds) * time.Second):
		return Text(fmt.Sprintf("slept %ds", a.Seconds)), nil
	case <-ctx.Done():
		return Text("sleep interrupted"), nil
	}
}
