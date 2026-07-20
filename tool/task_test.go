package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func runTC(t *testing.T, tl CoreTool, tc *ToolContext, in any) Result {
	t.Helper()
	raw, _ := json.Marshal(in)
	if err := ValidateInput(tl.InputSchema(), raw); err != nil {
		t.Fatalf("%s schema: %v", tl.Name(), err)
	}
	res, err := tl.Call(context.Background(), raw, tc)
	if err != nil {
		t.Fatalf("%s: %v", tl.Name(), err)
	}
	return res
}

func newTestManager(t *testing.T) *Manager {
	t.Helper()
	m, err := NewManager("test-" + t.Name())
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	t.Cleanup(m.Cleanup)
	return m
}

func waitTerminal(t *testing.T, m *Manager, id string) TaskInfo {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if info, ok := m.Get(id); ok && info.Status.Terminal() {
			return info
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("task %s did not reach a terminal state", id)
	return TaskInfo{}
}

// A backgrounded task records its exit and enqueues a completion notification
// carrying the output-file path (not the output itself).
func TestManagerSpawnCompleteNotifies(t *testing.T) {
	m := newTestManager(t)
	// Output ("42") deliberately differs from the command text so we can assert the
	// notification carries the path, not the output.
	task, err := m.Spawn(SpawnSpec{Command: `expr 6 \* 7`})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminal(t, m, task.ID)
	if info.Status != TaskCompleted || info.ExitCode != 0 {
		t.Fatalf("status=%s exit=%d", info.Status, info.ExitCode)
	}
	out, err := m.Output(task.ID, 0)
	if err != nil || !strings.Contains(out, "42") {
		t.Fatalf("output=%q err=%v", out, err)
	}
	notes := m.DrainNotifications()
	if len(notes) != 1 || notes[0].Status != TaskCompleted {
		t.Fatalf("notifications=%+v", notes)
	}
	if !strings.Contains(notes[0].String(), task.OutputPath) {
		t.Fatalf("notification should carry the output path: %s", notes[0].String())
	}
	if strings.Contains(notes[0].String(), "42") {
		t.Fatalf("notification must NOT carry the output content: %s", notes[0].String())
	}
	// Drained once; nothing left.
	if again := m.DrainNotifications(); len(again) != 0 {
		t.Fatalf("notifications not cleared after drain: %+v", again)
	}
}

// A failing command records a non-zero exit code.
func TestManagerFailedExitCode(t *testing.T) {
	m := newTestManager(t)
	task, _ := m.Spawn(SpawnSpec{Command: "exit 3"})
	m.Background(task.ID)
	info := waitTerminal(t, m, task.ID)
	if info.Status != TaskFailed || info.ExitCode != 3 {
		t.Fatalf("status=%s exit=%d", info.Status, info.ExitCode)
	}
}

// Bash run_in_background returns immediately and the task is later readable.
func TestBashExplicitBackground(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	res := runTC(t, NewBash(), tc, map[string]any{"command": "echo bg-out", "run_in_background": true})
	if res.IsError || !strings.Contains(res.Flatten(), "background") {
		t.Fatalf("expected background launch: %q (err=%v)", res.Flatten(), res.IsError)
	}
	tasks := m.List()
	if len(tasks) != 1 {
		t.Fatalf("expected 1 task, got %d", len(tasks))
	}
	id := tasks[0].ID
	waitTerminal(t, m, id)
	out := runTC(t, NewTaskOutput(), tc, map[string]any{"task_id": id}).Flatten()
	if !strings.Contains(out, "bg-out") {
		t.Fatalf("TaskOutput=%q", out)
	}
}

// A foreground command returns its output inline and leaves no task behind.
func TestBashForegroundInline(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	res := runTC(t, NewBash(), tc, map[string]any{"command": "echo fg-out"})
	if res.IsError || !strings.Contains(res.Flatten(), "fg-out") {
		t.Fatalf("foreground output=%q err=%v", res.Flatten(), res.IsError)
	}
	if tasks := m.List(); len(tasks) != 0 {
		t.Fatalf("foreground task should be forgotten, got %d", len(tasks))
	}
}

// A long-running command that exceeds its timeout is moved to the background.
func TestBashAutoBackgroundOnTimeout(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	res := runTC(t, NewBash(), tc, map[string]any{
		"command":    "while true; do echo tick; sleep 0.1; done",
		"timeout_ms": 300,
	})
	if res.IsError || !strings.Contains(res.Flatten(), "background") {
		t.Fatalf("expected auto-background: %q (err=%v)", res.Flatten(), res.IsError)
	}
	tasks := m.List()
	if len(tasks) != 1 || tasks[0].Status != TaskRunning {
		t.Fatalf("expected 1 running task, got %+v", tasks)
	}
	// Stop it.
	if r := runTC(t, NewTaskStop(), tc, map[string]any{"task_id": tasks[0].ID}); r.IsError {
		t.Fatalf("TaskStop: %q", r.Flatten())
	}
	info := waitTerminal(t, m, tasks[0].ID)
	if info.Status != TaskKilled {
		t.Fatalf("expected killed, got %s", info.Status)
	}
}

// sleep is on the auto-background denylist: a timed-out sleep is killed, not
// backgrounded, and surfaces a timeout error.
func TestBashSleepTimesOutNotBackgrounded(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	res := runTC(t, NewBash(), tc, map[string]any{"command": "sleep 5", "timeout_ms": 200})
	if !res.IsError || !strings.Contains(res.Flatten(), "timed out") {
		t.Fatalf("expected timeout error: %q (err=%v)", res.Flatten(), res.IsError)
	}
	if tasks := m.List(); len(tasks) != 0 {
		t.Fatalf("timed-out sleep should be forgotten, got %d", len(tasks))
	}
}

func TestTaskListAndStopAll(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	for i := 0; i < 2; i++ {
		task, _ := m.Spawn(SpawnSpec{Command: "sleep 5"})
		m.Background(task.ID)
	}
	list := runTC(t, NewTaskList(), tc, map[string]any{}).Flatten()
	if strings.Count(list, "task_") != 2 {
		t.Fatalf("TaskList=%q", list)
	}
	if r := runTC(t, NewTaskStop(), tc, map[string]any{}); !strings.Contains(r.Flatten(), "Stopped 2") {
		t.Fatalf("TaskStop all=%q", r.Flatten())
	}
}

// Monitor always backgrounds and reports an output file.
func TestMonitorBackgrounds(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	res := runTC(t, NewMonitor(), tc, map[string]any{"command": "sleep 5", "description": "watcher"})
	if res.IsError || !strings.Contains(res.Flatten(), "monitor") {
		t.Fatalf("Monitor=%q err=%v", res.Flatten(), res.IsError)
	}
	tasks := m.List()
	if len(tasks) != 1 || tasks[0].Kind != KindMonitor || tasks[0].Status != TaskRunning {
		t.Fatalf("monitor task=%+v", tasks)
	}
}

// TaskOutput block=true waits for completion.
func TestTaskOutputBlock(t *testing.T) {
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	task, _ := m.Spawn(SpawnSpec{Command: "sleep 0.3; echo done"})
	m.Background(task.ID)
	out := runTC(t, NewTaskOutput(), tc, map[string]any{"task_id": task.ID, "block": true, "timeout_ms": 5000}).Flatten()
	if !strings.Contains(out, "done") || !strings.Contains(out, "completed") {
		t.Fatalf("blocking TaskOutput=%q", out)
	}
}

// Tools degrade gracefully when no manager is configured.
func TestToolsWithoutManager(t *testing.T) {
	tc := &ToolContext{}
	if r := runTC(t, NewTaskOutput(), tc, map[string]any{"task_id": "task_1"}); !strings.Contains(r.Flatten(), "not enabled") {
		t.Fatalf("TaskOutput without manager=%q", r.Flatten())
	}
	if r := runTC(t, NewMonitor(), tc, map[string]any{"command": "echo x"}); !strings.Contains(r.Flatten(), "not enabled") {
		t.Fatalf("Monitor without manager=%q", r.Flatten())
	}
}

func TestTailFileOmitsEarlier(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out")
	var b strings.Builder
	for i := 0; i < 5000; i++ {
		b.WriteString("0123456789\n")
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := tailFile(path, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "earlier output omitted") {
		t.Fatalf("expected omission notice: %q", out[:80])
	}
	if len(out) > 1000+200 {
		t.Fatalf("tail too large: %d", len(out))
	}
}

func TestCleanupRemovesDir(t *testing.T) {
	m, err := NewManager("cleanup-test")
	if err != nil {
		t.Fatal(err)
	}
	task, _ := m.Spawn(SpawnSpec{Command: "sleep 5"})
	dir := filepath.Dir(m.dir)
	m.Cleanup()
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("session dir not removed: %v", err)
	}
	// The spawned process is killed by Cleanup.
	info, _ := m.Get(task.ID)
	if info.Status == TaskRunning {
		t.Fatalf("task still running after cleanup")
	}
}
