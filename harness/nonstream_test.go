package harness

import (
	"context"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// scriptedSyncModel drives the loop's non-streaming path (QueryInput.NonStreaming
// + QueryDeps.CallModelSync): one preset whole-message reply per call.
type scriptedSyncModel struct {
	turns []syncTurn
	calls int
}

type syncTurn struct {
	msg   llm.Message
	stop  string
	usage llm.Usage
}

func (m *scriptedSyncModel) call(_ context.Context, _ llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	t := m.turns[m.calls]
	m.calls++
	return t.msg, t.stop, t.usage, nil
}

func syncToolTurn(id, name, input string) syncTurn {
	return syncTurn{
		msg:  llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockToolUse, ID: id, Name: name, Input: []byte(input)}}},
		stop: "tool_use",
	}
}

func syncTextTurn(s string) syncTurn {
	return syncTurn{
		msg:   llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock(s)}},
		stop:  "end_turn",
		usage: llm.Usage{InputTokens: 5, OutputTokens: 2},
	}
}

// TestNonStreamingRunsToolThenCompletes mirrors TestLoopRunsToolThenCompletes but
// over the non-streaming path: the loop must still execute the tool, feed back
// the result, and complete — producing the same conversation shape and events.
func TestNonStreamingRunsToolThenCompletes(t *testing.T) {
	m := &scriptedSyncModel{turns: []syncTurn{
		syncToolTurn("c1", "Bash", `{"command":"echo hi"}`),
		syncTextTurn("done"),
	}}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewBash()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		NonStreaming:   true,
	}

	var sawText, sawToolUse, sawToolResult bool
	var term *Terminal
	for ev, err := range Query(context.Background(), in, QueryDeps{CallModelSync: m.call}) {
		if err != nil {
			t.Fatalf("event err: %v", err)
		}
		switch ev.Kind {
		case KindText:
			if ev.Text == "done" {
				sawText = true
			}
		case KindToolUse:
			sawToolUse = true
		case KindToolResult:
			sawToolResult = true
		case KindResult:
			term = ev.Terminal
		}
	}
	if term == nil {
		t.Fatal("no terminal event")
	}
	if term.Reason != ReasonCompleted || term.Text != "done" {
		t.Fatalf("terminal=%+v", term)
	}
	if !sawText || !sawToolUse || !sawToolResult {
		t.Fatalf("missing host events: text=%v toolUse=%v toolResult=%v", sawText, sawToolUse, sawToolResult)
	}
	// Same conversation shape as the streaming test: user, assistant(tool_use),
	// user(tool_result), assistant(text).
	if len(term.Messages) != 4 {
		t.Fatalf("messages=%d", len(term.Messages))
	}
	res := term.Messages[2].Content[0]
	if res.ToolUseID != "c1" || !strings.Contains(res.Content[0].Text, "hi") {
		t.Fatalf("tool_result wrong: %+v", res)
	}
	if m.calls != 2 {
		t.Fatalf("model calls=%d", m.calls)
	}
	// Usage must accumulate across turns exactly like the streaming accumulator.
	if term.Usage.InputTokens != 5 || term.Usage.OutputTokens != 2 {
		t.Fatalf("usage=%+v", term.Usage)
	}
}
