package tool

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Background task execution.
//
// A Manager owns the set of background processes started during a Session. Each
// task runs detached from the per-turn context (so it outlives the agent turn
// that launched it), streams its combined stdout/stderr to a per-task file on
// disk, and — when it finishes — enqueues a notification the harness drains into
// the conversation. The model is told only the task id, status, and output-file
// path; it reads the actual output on demand (Read or the TaskOutput tool).

// TaskStatus is the lifecycle state of a background task.
type TaskStatus string

const (
	TaskRunning   TaskStatus = "running"
	TaskCompleted TaskStatus = "completed"
	TaskFailed    TaskStatus = "failed"
	TaskKilled    TaskStatus = "killed"
)

// Terminal reports whether the status is final (the process has exited).
func (s TaskStatus) Terminal() bool {
	return s == TaskCompleted || s == TaskFailed || s == TaskKilled
}

// TaskKind distinguishes a one-shot background command from a long-lived monitor.
type TaskKind string

const (
	KindBash    TaskKind = "bash"
	KindMonitor TaskKind = "monitor"
)

// Tunables for the on-disk output store.
const (
	// maxTaskOutputBytes caps a single task's output file; a task that exceeds it
	// is terminated to keep a runaway process from filling the disk.
	maxTaskOutputBytes = 50 << 20 // 50 MiB
	// staleSessionTTL bounds how long another session's task directory survives
	// before it is garbage-collected on the next Manager creation.
	staleSessionTTL = 24 * time.Hour
)

// Task is a single background process tracked by a Manager. Its mutable fields
// are guarded by Manager.mu; read snapshots through Manager methods (which copy
// into TaskInfo).
type Task struct {
	ID          string
	Kind        TaskKind
	Command     string
	Description string
	Status      TaskStatus
	ExitCode    int
	StartTime   time.Time
	EndTime     time.Time
	OutputPath  string

	// internal, guarded by Manager.mu
	backgrounded bool          // surfaced as a background task: notify on exit
	notified     bool          // completion already enqueued into pending
	oversize     bool          // terminated by the size watchdog
	cmd          *exec.Cmd
	done         chan struct{} // closed by watch when the process exits
}

func (t *Task) info() TaskInfo {
	return TaskInfo{
		ID: t.ID, Kind: t.Kind, Command: t.Command, Description: t.Description,
		Status: t.Status, ExitCode: t.ExitCode,
		StartTime: t.StartTime, EndTime: t.EndTime,
		OutputPath: t.OutputPath, Backgrounded: t.backgrounded,
	}
}

// TaskInfo is an immutable snapshot of a task for safe reads outside the lock.
type TaskInfo struct {
	ID          string
	Kind        TaskKind
	Command     string
	Description string
	Status      TaskStatus
	ExitCode    int
	StartTime   time.Time
	EndTime     time.Time
	OutputPath  string
	Backgrounded bool
}

// Notification is surfaced to the model when a backgrounded task exits.
type Notification struct {
	TaskID     string
	Kind       TaskKind
	Status     TaskStatus
	OutputPath string
	Summary    string
}

// String renders the notification as the <task-notification> block injected into
// the conversation. It deliberately carries only the path, not the output.
func (n Notification) String() string {
	return fmt.Sprintf("<task-notification>\n"+
		"  <task-id>%s</task-id>\n"+
		"  <task-type>%s</task-type>\n"+
		"  <status>%s</status>\n"+
		"  <output-file>%s</output-file>\n"+
		"  <summary>%s</summary>\n"+
		"</task-notification>",
		n.TaskID, n.Kind, n.Status, n.OutputPath, n.Summary)
}

// SpawnSpec describes a process to launch in the background.
type SpawnSpec struct {
	Command     string
	Description string
	Kind        TaskKind
	WorkingDir  string
	// Env holds extra "KEY=VALUE" entries layered onto the process environment for
	// this command (see ToolContext.Env). Empty = inherit unchanged.
	Env []string
}

// withEnv returns the current process environment extended with extra
// "KEY=VALUE" entries, or nil when extra is empty (child inherits unchanged).
// os/exec dedups by key — case-insensitive on Windows, keeping the LAST
// occurrence — so the appended entries override inherited ones on every OS.
func withEnv(extra []string) []string {
	if len(extra) == 0 {
		return nil
	}
	return append(os.Environ(), extra...)
}

// Manager owns the background tasks of one Session.
type Manager struct {
	mu      sync.Mutex
	tasks   map[string]*Task
	order   []string
	pending []Notification
	dir     string
	seq     int

	rootCtx    context.Context
	rootCancel context.CancelFunc

	// interactive PTY shell sessions (see shell_session.go). Separate map + mutex
	// from the file-based tasks above: sessions are persistent, stdin-driven, and
	// keep an in-memory ring buffer rather than streaming to a file.
	sessMu   sync.Mutex
	sessions map[string]*ptySession
	sessSeq  int
}

func baseTaskDir() string { return filepath.Join(os.TempDir(), "norma") }

// NewManager creates the per-session task directory and garbage-collects stale
// directories left by previous sessions.
func NewManager(sessionID string) (*Manager, error) {
	if sessionID == "" {
		sessionID = "default"
	}
	dir := filepath.Join(baseTaskDir(), sessionID, "tasks")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	cleanStaleSessions(baseTaskDir(), sessionID)
	ctx, cancel := context.WithCancel(context.Background())
	return &Manager{
		tasks:      map[string]*Task{},
		sessions:   map[string]*ptySession{},
		dir:        dir,
		rootCtx:    ctx,
		rootCancel: cancel,
	}, nil
}

// cleanStaleSessions removes sibling session directories older than the TTL.
func cleanStaleSessions(base, current string) {
	entries, err := os.ReadDir(base)
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-staleSessionTTL)
	for _, e := range entries {
		if !e.IsDir() || e.Name() == current {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.ModTime().Before(cutoff) {
			_ = os.RemoveAll(filepath.Join(base, e.Name()))
		}
	}
}

// Spawn launches a process detached from the per-turn context. Its output streams
// to a file; a watch goroutine records the exit status and (if backgrounded) a
// completion notification.
func (m *Manager) Spawn(spec SpawnSpec) (*Task, error) {
	m.mu.Lock()
	m.seq++
	id := fmt.Sprintf("task_%d", m.seq)
	m.mu.Unlock()

	outPath := filepath.Join(m.dir, id+".output")
	f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}

	kind := spec.Kind
	if kind == "" {
		kind = KindBash
	}
	shell, flags := shellCmd()
	cmd := exec.CommandContext(m.rootCtx, shell, append(flags, spec.Command)...)
	cmd.Dir = spec.WorkingDir
	if env := withEnv(spec.Env); env != nil {
		cmd.Env = env
	}
	cmd.Stdout = f
	cmd.Stderr = f
	setDetached(cmd)

	if err := cmd.Start(); err != nil {
		_ = f.Close()
		_ = os.Remove(outPath)
		return nil, err
	}

	t := &Task{
		ID:          id,
		Kind:        kind,
		Command:     spec.Command,
		Description: spec.Description,
		Status:      TaskRunning,
		StartTime:   time.Now(),
		OutputPath:  outPath,
		cmd:         cmd,
		done:        make(chan struct{}),
	}
	m.mu.Lock()
	m.tasks[id] = t
	m.order = append(m.order, id)
	m.mu.Unlock()

	go m.watch(t, f)
	return t, nil
}

func (m *Manager) watch(t *Task, f *os.File) {
	stop := make(chan struct{})
	go m.watchSize(t, stop)

	err := t.cmd.Wait()
	close(stop)
	_ = f.Close()

	m.mu.Lock()
	if t.Status == TaskRunning { // not pre-empted by Kill / watchdog
		if err == nil {
			t.Status = TaskCompleted
			t.ExitCode = 0
		} else if ee, ok := err.(*exec.ExitError); ok {
			t.Status = TaskFailed
			t.ExitCode = ee.ExitCode()
		} else {
			t.Status = TaskFailed
			t.ExitCode = -1
		}
	}
	t.EndTime = time.Now()
	close(t.done)
	if t.backgrounded && !t.notified && (t.Status == TaskCompleted || t.Status == TaskFailed) {
		t.notified = true
		m.pending = append(m.pending, m.notifFor(t))
	}
	m.mu.Unlock()
}

// watchSize terminates a task whose output file grows past the cap.
func (m *Manager) watchSize(t *Task, stop chan struct{}) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			if fi, err := os.Stat(t.OutputPath); err == nil && fi.Size() > maxTaskOutputBytes {
				m.terminate(t, TaskFailed, true)
				return
			}
		}
	}
}

// terminate kills a still-running task, recording the given terminal status. The
// watch goroutine observes the exit and finalizes EndTime / notifications.
func (m *Manager) terminate(t *Task, status TaskStatus, oversize bool) {
	m.mu.Lock()
	if t.Status != TaskRunning {
		m.mu.Unlock()
		return
	}
	t.Status = status
	t.oversize = oversize
	proc := t.cmd.Process
	m.mu.Unlock()
	if proc != nil {
		treeKill(proc)
	}
}

func (m *Manager) notifFor(t *Task) Notification {
	var sum string
	switch {
	case t.oversize:
		sum = fmt.Sprintf("Task %q 输出超过上限已被终止 (%s)。", short(t.Command), t.Kind)
	case t.Status == TaskCompleted:
		sum = fmt.Sprintf("Task %q 已完成 (exit %d)。用 TaskOutput 或 Read 读取输出文件。", short(t.Command), t.ExitCode)
	case t.Status == TaskFailed:
		sum = fmt.Sprintf("Task %q 失败 (exit %d)。用 TaskOutput 或 Read 读取输出文件。", short(t.Command), t.ExitCode)
	default:
		sum = fmt.Sprintf("Task %q %s。", short(t.Command), t.Status)
	}
	return Notification{TaskID: t.ID, Kind: t.Kind, Status: t.Status, OutputPath: t.OutputPath, Summary: sum}
}

// Background marks a task as backgrounded: its completion will be surfaced to the
// model. If the task already finished (a race against a fast command), the
// notification is enqueued immediately.
func (m *Manager) Background(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	t, ok := m.tasks[id]
	if !ok {
		return
	}
	t.backgrounded = true
	if t.Status.Terminal() && !t.notified && (t.Status == TaskCompleted || t.Status == TaskFailed) {
		t.notified = true
		m.pending = append(m.pending, m.notifFor(t))
	}
}

// Forget removes a task from the registry and deletes its output file. Used for
// foreground commands once their output has been returned inline.
func (m *Manager) Forget(id string) {
	m.mu.Lock()
	t, ok := m.tasks[id]
	if ok {
		delete(m.tasks, id)
		for i, n := range m.order {
			if n == id {
				m.order = append(m.order[:i], m.order[i+1:]...)
				break
			}
		}
	}
	m.mu.Unlock()
	if ok {
		_ = os.Remove(t.OutputPath)
	}
}

// RunForeground starts a task and waits up to timeout. finished is true if the
// process exited within the timeout (output is then returned); otherwise the task
// keeps running and the caller decides whether to background or kill it. A
// cancelled ctx kills the task and returns ctx.Err().
func (m *Manager) RunForeground(ctx context.Context, spec SpawnSpec, timeout time.Duration) (t *Task, finished bool, output string, err error) {
	t, err = m.Spawn(spec)
	if err != nil {
		return nil, false, "", err
	}
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-t.done:
		out, _ := m.Output(t.ID, 0)
		return t, true, out, nil
	case <-timer.C:
		return t, false, "", nil
	case <-ctx.Done():
		m.Kill(t.ID)
		<-t.done
		return t, false, "", ctx.Err()
	}
}

// Kill terminates a running task (process tree) and marks it killed.
func (m *Manager) Kill(id string) bool {
	m.mu.Lock()
	t, ok := m.tasks[id]
	m.mu.Unlock()
	if !ok {
		return false
	}
	m.terminate(t, TaskKilled, false)
	return true
}

// KillAll terminates every running task and returns how many were stopped.
func (m *Manager) KillAll() int {
	m.mu.Lock()
	ts := make([]*Task, 0, len(m.tasks))
	for _, t := range m.tasks {
		ts = append(ts, t)
	}
	m.mu.Unlock()
	n := 0
	for _, t := range ts {
		m.mu.Lock()
		running := t.Status == TaskRunning
		m.mu.Unlock()
		if running {
			m.terminate(t, TaskKilled, false)
			n++
		}
	}
	return n
}

// Get returns a snapshot of one task.
func (m *Manager) Get(id string) (TaskInfo, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	t, ok := m.tasks[id]
	if !ok {
		return TaskInfo{}, false
	}
	return t.info(), true
}

// List returns snapshots of all tasks in creation order.
func (m *Manager) List() []TaskInfo {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]TaskInfo, 0, len(m.order))
	for _, id := range m.order {
		if t, ok := m.tasks[id]; ok {
			out = append(out, t.info())
		}
	}
	return out
}

// Wait blocks until the task exits or the timeout/ctx fires. ok is true only if
// the task reached a terminal state.
func (m *Manager) Wait(ctx context.Context, id string, timeout time.Duration) (TaskInfo, bool) {
	m.mu.Lock()
	t, ok := m.tasks[id]
	m.mu.Unlock()
	if !ok {
		return TaskInfo{}, false
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-t.done:
		info, _ := m.Get(id)
		return info, true
	case <-timer.C:
		info, _ := m.Get(id)
		return info, false
	case <-ctx.Done():
		info, _ := m.Get(id)
		return info, false
	}
}

// Output returns the tail of a task's output file, up to maxBytes (default 30000).
func (m *Manager) Output(id string, maxBytes int) (string, error) {
	m.mu.Lock()
	t, ok := m.tasks[id]
	m.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("unknown task %q", id)
	}
	return tailFile(t.OutputPath, maxBytes)
}

// DrainNotifications returns and clears the pending completion notifications.
func (m *Manager) DrainNotifications() []Notification {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.pending) == 0 {
		return nil
	}
	out := m.pending
	m.pending = nil
	return out
}

// Cleanup kills all tasks, cancels the root context, and removes the session's
// task directory. Safe to call on a nil Manager.
func (m *Manager) Cleanup() {
	if m == nil {
		return
	}
	m.KillAll()
	m.closeAllSessions()
	if m.rootCancel != nil {
		m.rootCancel()
	}
	_ = os.RemoveAll(filepath.Dir(m.dir)) // <base>/<sessionID>
}

// tailFile reads the last maxBytes of a file, prefixing a notice when earlier
// output is omitted.
func tailFile(path string, maxBytes int) (string, error) {
	if maxBytes <= 0 {
		maxBytes = defaultMaxOutput
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return "", err
	}
	size := fi.Size()
	if size <= int64(maxBytes) {
		b, err := io.ReadAll(f)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	off := size - int64(maxBytes)
	if _, err := f.Seek(off, io.SeekStart); err != nil {
		return "", err
	}
	b, err := io.ReadAll(f)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("[%d KB of earlier output omitted — read the output file directly for full output]\n", off/1024) + string(b), nil
}

// shellCmd returns the platform shell and the argument flags that precede the
// command string. On Windows it uses PowerShell with -NoProfile -NonInteractive
// (the command is passed to -Command) so a user profile can't inject banners or
// prompts and an interactive prompt can't hang an unattended run. Elsewhere it
// uses `bash -c`.
func shellCmd() (string, []string) {
	if runtime.GOOS == "windows" {
		return "powershell", []string{"-NoProfile", "-NonInteractive", "-Command"}
	}
	return "bash", []string{"-c"}
}

// short trims a command to a compact single-line label for summaries.
func short(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 60 {
		return s[:57] + "..."
	}
	return s
}

// BackgroundTasksDisabled reports whether background execution is turned off via
// the AGENT_CORE_DISABLE_BACKGROUND_TASKS environment variable.
func BackgroundTasksDisabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENT_CORE_DISABLE_BACKGROUND_TASKS"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}
