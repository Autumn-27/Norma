package agentcore

import (
	"context"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// TestPlanModeGatesThenAllowsWrites proves the plan-mode lifecycle: a write is
// denied while planning, and the same write succeeds after ExitPlanMode approval
// flips the permission mode mid-loop.
func TestPlanModeGatesThenAllowsWrites(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "out.txt")

	turns := [][]llm.StreamEvent{
		toolTurn("w1", "Write", `{"file_path":"`+jsonEsc(target)+`","content":"first"}`),  // denied (plan mode)
		toolTurn("p1", "ExitPlanMode", `{"plan":"write the file"}`),                       // approved → mode flips
		toolTurn("w2", "Write", `{"file_path":"`+jsonEsc(target)+`","content":"second"}`), // now allowed
		textTurn("done"),
	}
	var n int
	callModel := func(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		evs := turns[n]
		n++
		return func(yield func(llm.StreamEvent, error) bool) {
			for _, e := range evs {
				if !yield(e, nil) {
					return
				}
			}
		}
	}

	sess := NewSession(Options{
		SystemPrompt:   []string{"plan first"},
		Tools:          []tool.CoreTool{tool.NewWrite()},
		PermissionMode: permission.ModeDefault,
		WorkingDir:     dir,
		Plan: &PlanOptions{
			StartInPlanMode: true,
			Approver:        func(_ context.Context, _ string) (bool, string) { return true, "" },
			TargetMode:      permission.ModeAcceptEdits,
		},
		Deps: harness.QueryDeps{CallModel: callModel},
	})

	var results []llm.ContentBlock
	for ev, err := range sess.Prompt(context.Background(), "create the file") {
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if ev.Kind == harness.KindToolResult {
			results = append(results, *ev.ToolResult)
		}
	}

	// First Write denied in plan mode.
	if !results[0].IsError || !strings.Contains(results[0].Content[0].Text, "not permitted") {
		t.Fatalf("first write should be denied: %+v", results[0])
	}
	// Second Write (after approval) succeeded.
	if results[2].IsError {
		t.Fatalf("second write should succeed: %+v", results[2])
	}
	if b, _ := os.ReadFile(target); string(b) != "second" {
		t.Fatalf("file content=%q, want %q", b, "second")
	}
}

func jsonEsc(s string) string { return strings.ReplaceAll(s, `\`, `\\`) }

func textTurn(s string) []llm.StreamEvent {
	return []llm.StreamEvent{{Type: llm.SETextDelta, Text: s}, {Type: llm.SEMessageDelta, StopReason: "end_turn"}, {Type: llm.SEMessageStop}}
}

func toolTurn(id, name, input string) []llm.StreamEvent {
	return []llm.StreamEvent{
		{Type: llm.SEToolUseStart, ToolID: id, ToolName: name},
		{Type: llm.SEToolInputJSON, Text: input},
		{Type: llm.SEMessageDelta, StopReason: "tool_use"},
		{Type: llm.SEMessageStop},
	}
}
