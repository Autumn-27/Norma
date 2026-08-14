package agentcore

import (
	"context"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// TestBackgroundTaskEndToEnd proves the full Session path: a backgrounded Bash
// command is launched in one prompt, its completion notification is surfaced at
// the start of the next prompt, and Close removes the session's task directory.
func TestBackgroundTaskEndToEnd(t *testing.T) {
	dir := t.TempDir()

	turns := [][]llm.StreamEvent{
		toolTurn("b1", "Bash", `{"command":"sleep 1; echo e2e-out","run_in_background":true}`),
		textTurn("launched"), // completes prompt #1 while the task is still running
		textTurn("ack"),      // prompt #2; the notification is drained before this turn
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
		SystemPrompt:   []string{"go"},
		Tools:          []tool.CoreTool{tool.NewBash()},
		PermissionMode: permission.ModeBypass,
		WorkingDir:     dir,
		Deps:           harness.QueryDeps{CallModel: callModel},
	})

	// The task tools are auto-registered alongside Bash.
	for _, name := range []string{"TaskOutput", "TaskStop", "TaskList", "Monitor"} {
		if _, ok := sess.registry.Get(name); !ok {
			t.Fatalf("expected %s tool to be registered", name)
		}
	}

	// Prompt #1: launch the background command.
	var launchText string
	for ev := range mustEvents(t, sess, "start the scan") {
		if ev.Kind == harness.KindToolResult {
			launchText = ev.ToolResult.Content[0].Text
		}
	}
	if !strings.Contains(launchText, "background") {
		t.Fatalf("expected background launch result, got %q", launchText)
	}

	// The session's task temp dir exists while the task runs.
	taskDir := filepath.Join(os.TempDir(), "norma", sess.SessionID())
	if _, err := os.Stat(taskDir); err != nil {
		t.Fatalf("task dir should exist: %v", err)
	}

	// Wait for the command to finish so its notification is queued.
	time.Sleep(1500 * time.Millisecond)

	// Prompt #2: the completion notification is injected before the new turn.
	for range mustEvents(t, sess, "anything new?") {
	}
	var found bool
	for _, m := range sess.Messages() {
		if strings.Contains(m.Text(), "<task-notification>") && strings.Contains(m.Text(), "completed") {
			found = true
		}
	}
	if !found {
		t.Fatal("background completion notification was not surfaced into the conversation")
	}

	// Close removes the task directory.
	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(taskDir); !os.IsNotExist(err) {
		t.Fatalf("task dir should be removed after Close: %v", err)
	}
}

func mustEvents(t *testing.T, s *Session, input string) iter.Seq[harness.Event] {
	t.Helper()
	return func(yield func(harness.Event) bool) {
		for ev, err := range s.Prompt(context.Background(), input) {
			if err != nil {
				t.Fatalf("prompt err: %v", err)
			}
			if !yield(ev) {
				return
			}
		}
	}
}
