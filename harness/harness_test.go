package harness

import (
	"context"
	"iter"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// scriptedModel returns one preset event sequence per call, driving the loop
// offline through QueryDeps.CallModel.
type scriptedModel struct {
	turns [][]llm.StreamEvent
	calls int
}

func (m *scriptedModel) call(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	evs := m.turns[m.calls]
	m.calls++
	return func(yield func(llm.StreamEvent, error) bool) {
		for _, ev := range evs {
			if !yield(ev, nil) {
				return
			}
		}
	}
}

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

func drain(t *testing.T, in QueryInput, deps QueryDeps) *Terminal {
	t.Helper()
	var term *Terminal
	for ev, err := range Query(context.Background(), in, deps) {
		if err != nil {
			t.Fatalf("event err: %v", err)
		}
		if ev.Kind == KindResult {
			term = ev.Terminal
		}
	}
	if term == nil {
		t.Fatal("no terminal event")
	}
	return term
}

func TestLoopRunsToolThenCompletes(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("c1", "Bash", `{"command":"echo hi"}`),
		textTurn("done"),
	}}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewBash()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonCompleted || term.Text != "done" {
		t.Fatalf("terminal=%+v", term)
	}
	// user, assistant(tool_use), user(tool_result), assistant(text)
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
}

func TestPermissionDenyPairsSyntheticResult(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("c1", "Write", `{"file_path":"x","content":"y"}`),
		textTurn("ok"),
	}}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewWrite()),
		PermissionMode: permission.ModePlan, // denies writes
		WorkingDir:     t.TempDir(),
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	res := term.Messages[2].Content[0]
	if !res.IsError || !strings.Contains(res.Content[0].Text, "not permitted") {
		t.Fatalf("expected permission-denied result, got %+v", res)
	}
}

func TestMaxTurns(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("a", "Bash", `{"command":"echo 1"}`),
		toolTurn("b", "Bash", `{"command":"echo 2"}`),
		toolTurn("c", "Bash", `{"command":"echo 3"}`),
	}}
	in := QueryInput{
		Tools:          tool.NewRegistry(tool.NewBash()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		MaxTurns:       2,
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonMaxTurns {
		t.Fatalf("reason=%v", term.Reason)
	}
}

func TestParallelReadOnlyExecution(t *testing.T) {
	// Two concurrent-safe reads in one turn must both run and pair results.
	dir := t.TempDir()
	in := QueryInput{Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(tool.NewGlob()), PermissionMode: permission.ModeDefault, WorkingDir: dir}
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		{
			{Type: llm.SEToolUseStart, ToolID: "g1", ToolName: "Glob"},
			{Type: llm.SEToolInputJSON, Text: `{"pattern":"*.go"}`},
			{Type: llm.SEToolUseStart, ToolID: "g2", ToolName: "Glob"},
			{Type: llm.SEToolInputJSON, Text: `{"pattern":"*.txt"}`},
			{Type: llm.SEMessageDelta, StopReason: "tool_use"},
			{Type: llm.SEMessageStop},
		},
		textTurn("done"),
	}}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	results := term.Messages[2].Content
	if len(results) != 2 || results[0].ToolUseID != "g1" || results[1].ToolUseID != "g2" {
		t.Fatalf("parallel results mis-ordered: %+v", results)
	}
}

// fakeCompactor simulates autoCompact once: it inserts a boundary + summary just
// before the last message (the kept tail), so the post-boundary working set is
// [boundary, summary, tail] — the shape recordNewBoundary must persist.
type fakeCompactor struct{ done bool }

func (f *fakeCompactor) Pre(_ context.Context, msgs []llm.Message) []llm.Message {
	if f.done || len(msgs) < 2 {
		return msgs
	}
	f.done = true
	tailStart := len(msgs) - 1
	boundary := llm.BoundaryMessage(llm.BoundaryMeta{Trigger: "auto", PreTokens: 999, MessagesSummarized: tailStart})
	summary := llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextBlock("[summary] DIGEST")}}
	out := append([]llm.Message{}, msgs[:tailStart]...)
	out = append(out, boundary, summary)
	return append(out, msgs[tailStart:]...)
}
func (f *fakeCompactor) Reactive(_ context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	return msgs, false
}
func (f *fakeCompactor) IsOverflow(error) bool { return false }

// capRecorder captures everything the harness persists.
type capRecorder struct {
	msgs       []llm.Message
	boundaries int
}

func (r *capRecorder) RecordMessage(m llm.Message, _ llm.Usage) { r.msgs = append(r.msgs, m) }
func (r *capRecorder) RecordBoundary(_ llm.BoundaryMeta)        { r.boundaries++ }

// TestCompactionWorkingSetIsPersisted verifies that when compaction inserts a
// boundary, recordNewBoundary persists the WHOLE post-boundary working set
// (boundary marker + summary + kept tail) as durable messages — so a cold-start
// resume can restore the compacted view instead of re-summarizing.
func TestCompactionWorkingSetIsPersisted(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{textTurn("done")}}
	rec := &capRecorder{}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("m1"), llm.UserText("m2"), llm.UserText("TAILMSG")},
		Tools:          tool.NewRegistry(),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		Compactor:      &fakeCompactor{},
		Recorder:       rec,
	}
	drain(t, in, QueryDeps{CallModel: m.call})

	if rec.boundaries != 1 {
		t.Fatalf("RecordBoundary calls=%d, want 1 (observability marker)", rec.boundaries)
	}
	// Find the persisted boundary marker; the summary and the tail must follow it.
	bi := -1
	for i, msg := range rec.msgs {
		if llm.IsBoundaryMarker(msg) {
			bi = i
			break
		}
	}
	if bi < 0 {
		t.Fatal("boundary marker was not persisted as a message")
	}
	post := rec.msgs[bi:]
	if len(post) < 3 {
		t.Fatalf("post-boundary persisted = %d msgs, want >=3 (boundary+summary+tail)", len(post))
	}
	if !strings.Contains(post[1].Text(), "DIGEST") {
		t.Fatalf("message after boundary should be the summary, got %q", post[1].Text())
	}
	if post[2].Text() != "TAILMSG" {
		t.Fatalf("kept tail not re-recorded after the boundary, got %q", post[2].Text())
	}
	// The persisted boundary reconstructs to the compacted API view: it opens with
	// [summary, tail] (the assistant's reply, recorded after, trails it).
	if api := llm.MessagesForAPI(post); len(api) < 2 || !strings.Contains(api[0].Text(), "DIGEST") || api[1].Text() != "TAILMSG" {
		t.Fatalf("MessagesForAPI(persisted working set) = %+v, want to open with [summary, TAILMSG]", api)
	}
}
