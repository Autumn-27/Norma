//go:build linux

package tool

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// growAndHold grows a bash variable well past any small cap, then holds that
// allocation in a sleep. The hold is what makes the test deterministic: the
// watchdog ticks every 2s, so the kill must land while the memory is still
// resident no matter how fast (or slow) the machine doubles the string.
const growAndHold = "x=a; while [ ${#x} -lt 200000000 ]; do x=\"$x$x\"; done; sleep 30"

func waitTerminalIn(t *testing.T, m *Manager, id string, d time.Duration) TaskInfo {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if info, ok := m.Get(id); ok && info.Status.Terminal() {
			return info
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("task %s did not reach a terminal state within %s", id, d)
	return TaskInfo{}
}

// The incident this guard exists for: a command that accumulates in a loop
// (an unbounded scanner re-appending its own results) must die at the cap and
// the tool error must NAME the cap — a bare "[exit code -1]" is exactly the
// failure signal that never reached the model in the original OOM loop.
func TestMemGuardKillsForegroundCommand(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "16m")
	m := newTestManager(t)
	tc := &ToolContext{Tasks: m}
	res := runTC(t, NewBash(), tc, map[string]any{
		"command":    growAndHold,
		"timeout_ms": 30000,
	})
	if !res.IsError {
		t.Fatalf("expected a memory-cap error, got success: %q", res.Flatten())
	}
	if out := res.Flatten(); !strings.Contains(out, "memory cap") || !strings.Contains(out, "16MiB") {
		t.Fatalf("error must name the cap that fired: %q", out)
	}
	if tasks := m.List(); len(tasks) != 0 {
		t.Fatalf("finished foreground task must be forgotten: %+v", tasks)
	}
}

// The kill takes the whole process group, not just the direct shell: a
// backgrounded descendant spawned by the command must die with it, or it
// outlives the guard and keeps its memory resident.
func TestMemGuardKillsDescendants(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "32m")
	dir := t.TempDir()
	m := newTestManager(t)
	cmd := fmt.Sprintf("(sleep 300 & echo $! > %s); %s", filepath.Join(dir, "child.pid"), growAndHold)
	task, err := m.Spawn(SpawnSpec{Command: cmd})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminalIn(t, m, task.ID, 30*time.Second)
	if info.Status != TaskFailed || info.Reason != terminateMemory {
		t.Fatalf("status=%s reason=%q, want failed/memory", info.Status, info.Reason)
	}
	b, err := os.ReadFile(filepath.Join(dir, "child.pid"))
	if err != nil {
		t.Fatalf("descendant pid file: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		t.Fatalf("bad pid %q: %v", b, err)
	}
	// The killed orphan is reparented to the container's pid 1, which may not
	// reap it — a zombie is dead, a running process is not.
	if st := procStatState(pid); st != "" && st != "Z" {
		t.Fatalf("descendant %d still alive, state=%q — group kill must take the whole tree", pid, st)
	}
}

// A backgrounded task that blows the cap notifies with the cap named, so the
// model learns WHY the task died on the next drain.
func TestMemGuardBackgroundNotification(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "32m")
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: growAndHold})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminalIn(t, m, task.ID, 30*time.Second)
	if info.Status != TaskFailed || info.Reason != terminateMemory {
		t.Fatalf("status=%s reason=%q, want failed/memory", info.Status, info.Reason)
	}
	notes := m.DrainNotifications()
	if len(notes) != 1 {
		t.Fatalf("expected 1 notification, got %+v", notes)
	}
	if !strings.Contains(notes[0].Summary, "32MiB") {
		t.Fatalf("summary must name the cap: %q", notes[0].Summary)
	}
}

// Control: a command that stays under the cap completes normally with an empty
// reason — the guard must not start firing on ordinary work.
func TestMemGuardLeavesCommandsUnderCap(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "512m")
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: "echo under-cap; sleep 1"})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminal(t, m, task.ID)
	if info.Status != TaskCompleted || info.Reason != "" {
		t.Fatalf("status=%s reason=%q, want completed/empty", info.Status, info.Reason)
	}
}

// "off" disables the guard: a command that would exceed a real cap must run to
// completion instead of being killed.
func TestMemGuardDisabledByEnv(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "off")
	m := newTestManager(t)
	// ~64 MiB allocation — well over the 16m cap used by the kill test above.
	task, err := m.Spawn(SpawnSpec{Command: "x=a; while [ ${#x} -lt 67108864 ]; do x=\"$x$x\"; done; echo done"})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminalIn(t, m, task.ID, 30*time.Second)
	if info.Status != TaskCompleted || info.Reason != "" {
		t.Fatalf("status=%s reason=%q, want completed/empty with the guard off", info.Status, info.Reason)
	}
	out, _ := m.Output(task.ID, 0)
	if !strings.Contains(out, "done") {
		t.Fatalf("output=%q", out)
	}
}

// procStatState returns the /proc/<pid>/stat state field ("" when the process
// is gone), parsing past the parenthesized comm the same way procPpid does.
func procStatState(pid int) string {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return ""
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return ""
	}
	fields := strings.Fields(string(b[i+1:]))
	if len(fields) < 1 {
		return ""
	}
	return fields[0]
}
