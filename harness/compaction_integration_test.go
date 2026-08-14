package harness

import (
	"context"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// recordingCompactor captures the lastInputTokens the loop passes to each Pre.
type recordingCompactor struct{ seen []int }

func (c *recordingCompactor) Pre(_ context.Context, msgs []llm.Message, last int) []llm.Message {
	c.seen = append(c.seen, last)
	return msgs
}
func (c *recordingCompactor) Reactive(_ context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	return msgs, false
}
func (c *recordingCompactor) IsOverflow(error) bool { return false }

func usageToolTurn(id, name, input string, in, out int) []llm.StreamEvent {
	return []llm.StreamEvent{
		{Type: llm.SEMessageStart, Usage: llm.Usage{InputTokens: in}},
		{Type: llm.SEToolUseStart, ToolID: id, ToolName: name},
		{Type: llm.SEToolInputJSON, Text: input},
		{Type: llm.SEMessageDelta, StopReason: "tool_use", Usage: llm.Usage{OutputTokens: out}},
		{Type: llm.SEMessageStop},
	}
}

func usageTextTurn(s string, in, out int) []llm.StreamEvent {
	return []llm.StreamEvent{
		{Type: llm.SEMessageStart, Usage: llm.Usage{InputTokens: in}},
		{Type: llm.SETextDelta, Text: s},
		{Type: llm.SEMessageDelta, StopReason: "end_turn", Usage: llm.Usage{OutputTokens: out}},
		{Type: llm.SEMessageStop},
	}
}

// The loop must pass the previous response's total token size to the compactor's
// Pre: 0 on the first turn (no prior response), then input+output of the turn
// that just completed.
func TestCompactorReceivesLastInputTokens(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		usageToolTurn("c1", "Bash", `{"command":"echo hi"}`, 100, 20),
		usageTextTurn("done", 500, 30),
	}}
	rc := &recordingCompactor{}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewBash()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		Compactor:      rc,
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonCompleted {
		t.Fatalf("reason=%v", term.Reason)
	}
	if len(rc.seen) != 2 {
		t.Fatalf("want 2 Pre calls, got %d: %v", len(rc.seen), rc.seen)
	}
	if rc.seen[0] != 0 {
		t.Fatalf("first Pre should see 0 (no prior response), got %d", rc.seen[0])
	}
	if rc.seen[1] != 120 {
		t.Fatalf("second Pre should see turn-1 tokens (100+20), got %d", rc.seen[1])
	}
}
