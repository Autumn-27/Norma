package harness

import (
	"strings"
	"testing"
	"time"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// A pending background-task completion is drained into the conversation and the
// loop takes one extra turn so the model can react to it.
func TestTaskNotificationInjected(t *testing.T) {
	mgr, err := tool.NewManager("harness-notify-test")
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	defer mgr.Cleanup()

	task, err := mgr.Spawn(tool.SpawnSpec{Command: "echo hi"})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	mgr.Background(task.ID)

	// Wait until the task finishes and its notification is queued.
	deadline := time.Now().Add(5 * time.Second)
	for {
		if info, ok := mgr.Get(task.ID); ok && info.Status.Terminal() {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("task did not finish")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Model: first turn produces no tools (loop should inject the notification and
	// continue); second turn completes.
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		textTurn("first"),
		textTurn("done"),
	}}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewBash()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		Tasks:          mgr,
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})

	if term.Reason != ReasonCompleted || term.Text != "done" {
		t.Fatalf("terminal=%+v", term)
	}
	if m.calls != 2 {
		t.Fatalf("expected the loop to take an extra turn for the notification, calls=%d", m.calls)
	}
	var found bool
	for _, msg := range term.Messages {
		if strings.Contains(msg.Text(), "<task-notification>") && strings.Contains(msg.Text(), task.OutputPath) {
			found = true
		}
	}
	if !found {
		t.Fatal("no task-notification injected into the conversation")
	}
}
