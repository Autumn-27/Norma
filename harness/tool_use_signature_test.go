package harness

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// recordingTool captures the raw input every invocation receives, so a test can
// assert on what the tool actually ran with rather than on what the accumulated
// message claims it ran with — the two disagreeing is the bug under test.
func recordingTool(got *[]string) tool.CoreTool {
	return tool.Build(tool.Spec{
		Name: "probe",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"command": map[string]any{"type": "string"}},
		},
		ReadOnly: func(json.RawMessage) bool { return true },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			*got = append(*got, string(in))
			return tool.Text("ok"), nil
		},
	})
}

// TestToolUseSignatureDoesNotFinalizeToolBlock guards the streaming loop against
// a tool_use block being cut short by an event that belongs inside it.
//
// Every event other than SEToolInputJSON falls through to finalizeTool, which
// emits KindToolUse and dispatches the tool immediately. SEToolUseSignature
// (Gemini's thought_signature) is emitted *within* a tool call's event run, so
// if it reached that boundary the tool would be dispatched with raw == "{}" and
// every later argument delta would be silently discarded — the tool runs with
// no command while the recorded history shows the real one.
//
// Gemini's current framing trails the signature after the arguments, where the
// premature dispatch happens to be harmless. This scripts the hostile order a
// gateway is free to produce — signature attached to the chunk that only
// declares id/name, arguments still to come.
func TestToolUseSignatureDoesNotFinalizeToolBlock(t *testing.T) {
	var got []string
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		{
			{Type: llm.SEToolUseStart, ToolID: "c1", ToolName: "probe"},
			// Signature BEFORE the arguments land.
			{Type: llm.SEToolUseSignature, Text: "sig-abc"},
			{Type: llm.SEToolInputJSON, Text: `{"command":`},
			{Type: llm.SEToolInputJSON, Text: `"echo hi"}`},
			{Type: llm.SEMessageDelta, StopReason: "tool_use"},
			{Type: llm.SEMessageStop},
		},
		textTurn("done"),
	}}

	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(recordingTool(&got)),
		PermissionMode: permission.ModeBypass,
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})

	if len(got) != 1 {
		t.Fatalf("tool ran %d times, want 1: %q", len(got), got)
	}
	if want := `{"command":"echo hi"}`; got[0] != want {
		t.Errorf("tool ran with %s, want %s — the signature event finalized the block early and the argument deltas were dropped", got[0], want)
	}

	// The signature must still reach the replayed history, or the next turn
	// against Gemini is rejected for the missing thought_signature.
	var found bool
	for _, msg := range term.Messages {
		for _, b := range msg.Content {
			if b.Type == llm.BlockToolUse && b.ID == "c1" {
				found = true
				if b.ThoughtSignature != "sig-abc" {
					t.Errorf("replayed tool_use carries signature %q, want %q", b.ThoughtSignature, "sig-abc")
				}
				if string(b.Input) != `{"command":"echo hi"}` {
					t.Errorf("replayed tool_use input = %s, want the full arguments", b.Input)
				}
			}
		}
	}
	if !found {
		t.Error("no tool_use block for c1 in the conversation snapshot")
	}
}
