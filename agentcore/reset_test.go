package agentcore

import (
	"testing"

	"github.com/Autumn-27/norma/tool"
)

// A background task outlives the turn that started it, so a host that resets to
// begin a new conversation would otherwise keep the old one's processes running
// and get their completion notifications injected into the new conversation.
func TestResetStopsBackgroundTasks(t *testing.T) {
	if tool.BackgroundTasksDisabled() {
		t.Skip("background tasks are disabled in this environment")
	}
	s := NewSession(Options{Provider: usageProvider{in: 1, out: 1}})
	if s.tasks == nil {
		t.Skip("no task manager in this environment")
	}
	spec := tool.SpawnSpec{Command: "sleep 30", WorkingDir: t.TempDir()}
	tk, err := s.tasks.Spawn(spec)
	if err != nil {
		t.Fatalf("spawn: %v", err)
	}

	s.Reset()

	info, ok := s.tasks.Get(tk.ID)
	if !ok {
		t.Fatalf("task %s vanished", tk.ID)
	}
	if info.Status != tool.TaskKilled {
		t.Fatalf("task status = %s after Reset, want %s", info.Status, tool.TaskKilled)
	}
	if s.Messages() != nil {
		t.Fatalf("messages not cleared: %v", s.Messages())
	}
}
