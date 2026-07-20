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

func maxTokensTurn(text string) []llm.StreamEvent {
	return []llm.StreamEvent{
		{Type: llm.SETextDelta, Text: text},
		{Type: llm.SEMessageDelta, StopReason: "max_tokens"},
		{Type: llm.SEMessageStop},
	}
}

// alwaysModel returns the same turn forever (with a hard cap via MaxTurns).
func alwaysModel(turn []llm.StreamEvent) func(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		return func(yield func(llm.StreamEvent, error) bool) {
			for _, e := range turn {
				if !yield(e, nil) {
					return
				}
			}
		}
	}
}

func TestMaxOutputTokensRecovery(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		maxTokensTurn("partial output"), // truncated
		textTurn("finished"),            // resumes and completes
	}}
	in := QueryInput{Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(), PermissionMode: permission.ModeBypass}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonCompleted || term.Text != "finished" {
		t.Fatalf("terminal=%+v", term)
	}
	// user, asst(partial), user(resume), asst(finished)
	if len(term.Messages) != 4 || !strings.Contains(term.Messages[2].Text(), "Resume directly") {
		t.Fatalf("resume message not injected: %+v", term.Messages)
	}
}

func TestMaxOutputTokensRecoveryLimit(t *testing.T) {
	// Always truncates → must stop after 3 recoveries (no infinite loop).
	in := QueryInput{Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(), PermissionMode: permission.ModeBypass, MaxTurns: 20}
	term := drain(t, in, QueryDeps{CallModel: alwaysModel(maxTokensTurn("x"))})
	if term.Reason != ReasonCompleted {
		t.Fatalf("should complete after recovery limit, got %v", term.Reason)
	}
	resumes := 0
	for _, msg := range term.Messages {
		if strings.Contains(msg.Text(), "Resume directly") {
			resumes++
		}
	}
	if resumes != 3 {
		t.Fatalf("expected 3 recovery injections, got %d", resumes)
	}
}

func TestTokenBudgetContinuation(t *testing.T) {
	in := QueryInput{
		Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(),
		PermissionMode: permission.ModeBypass, TokenBudget: 100000, MaxTurns: 20,
	}
	term := drain(t, in, QueryDeps{CallModel: alwaysModel(textTurn("done"))})
	if term.Reason != ReasonCompleted {
		t.Fatalf("reason=%v", term.Reason)
	}
	nudges := 0
	for _, msg := range term.Messages {
		if strings.Contains(msg.Text(), "token budget") {
			nudges++
		}
	}
	// continuationCount reaches 3 (with zero-delta diminishing) then stops.
	if nudges != 3 {
		t.Fatalf("expected 3 budget nudges then diminishing-returns stop, got %d", nudges)
	}
}

func TestRecoveriesDoNotCountTowardMaxTurns(t *testing.T) {
	// Two max_tokens recoveries then completion, with MaxTurns=1. Since recoveries
	// are retries of the same turn (not tool-turns), they must NOT trip max_turns.
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		maxTokensTurn("partial 1"),
		maxTokensTurn("partial 2"),
		textTurn("done"),
	}}
	in := QueryInput{
		Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(),
		PermissionMode: permission.ModeBypass, MaxTurns: 1,
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonCompleted || term.Text != "done" {
		t.Fatalf("recoveries should not count as turns; terminal=%+v", term)
	}
	if term.Turns != 1 {
		t.Fatalf("turnCount should stay 1 (no tool-turns), got %d", term.Turns)
	}
}

func TestToolTurnsCountTowardMaxTurns(t *testing.T) {
	// Each tool-turn advances turnCount; MaxTurns bounds them.
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("a", "Bash", `{"command":"echo 1"}`),
		toolTurn("b", "Bash", `{"command":"echo 2"}`),
		toolTurn("c", "Bash", `{"command":"echo 3"}`),
	}}
	in := QueryInput{
		Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(tool.NewBash()),
		PermissionMode: permission.ModeBypass, WorkingDir: t.TempDir(), MaxTurns: 2,
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonMaxTurns {
		t.Fatalf("reason=%v, want max_turns", term.Reason)
	}
}

type fakeHooks struct {
	stop func(calls int) (bool, []string, string)
	n    int
}

func (h *fakeHooks) PreToolUse(context.Context, string, []byte) (bool, string, []byte) {
	return false, "", nil
}
func (h *fakeHooks) PostToolUse(context.Context, string, []byte, []byte, bool) {}
func (h *fakeHooks) Stop(context.Context, []llm.Message) (bool, []string, string) {
	h.n++
	return h.stop(h.n)
}

func TestStopHookPreventTerminates(t *testing.T) {
	hooks := &fakeHooks{stop: func(int) (bool, []string, string) { return true, nil, "halt" }}
	in := QueryInput{Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(), Hooks: hooks}
	term := drain(t, in, QueryDeps{CallModel: alwaysModel(textTurn("done"))})
	if term.Reason != ReasonStopHookPrevented || term.Text != "halt" {
		t.Fatalf("terminal=%+v", term)
	}
}

func TestStopHookBlockingContinues(t *testing.T) {
	hooks := &fakeHooks{stop: func(n int) (bool, []string, string) {
		if n == 1 {
			return false, []string{"keep going"}, ""
		}
		return false, nil, ""
	}}
	m := &scriptedModel{turns: [][]llm.StreamEvent{textTurn("first"), textTurn("really done")}}
	in := QueryInput{Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(), Hooks: hooks}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if term.Reason != ReasonCompleted || term.Text != "really done" {
		t.Fatalf("terminal=%+v", term)
	}
	found := false
	for _, msg := range term.Messages {
		if strings.Contains(msg.Text(), "keep going") {
			found = true
		}
	}
	if !found {
		t.Fatal("blocking error not injected")
	}
}

func TestMessagesForAPIExcludesPreBoundary(t *testing.T) {
	seeded := []llm.Message{
		llm.UserText("ancient question"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("ancient answer")}},
		llm.BoundaryMessage(llm.BoundaryMeta{PreTokens: 1234}),
		llm.UserText("[summary] earlier work digest"),
		llm.UserText("recent question"),
	}
	var captured []llm.Message
	capture := func(_ context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		captured = append([]llm.Message(nil), req.Messages...)
		return alwaysModel(textTurn("ok"))(context.Background(), req)
	}
	in := QueryInput{Messages: seeded, Tools: tool.NewRegistry(), PermissionMode: permission.ModeBypass}
	drain(t, in, QueryDeps{CallModel: capture})
	// API view = summary + recent (boundary stripped, pre-boundary excluded).
	if len(captured) != 2 || captured[0].Text() != "[summary] earlier work digest" || captured[1].Text() != "recent question" {
		t.Fatalf("API view wrong: %+v", captured)
	}
}
