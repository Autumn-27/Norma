package tool

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/Autumn-27/norma/permission"
)

// catastrophicBash matches a small set of irreversibly destructive commands
// that are hard-denied regardless of mode (NFR-10). This is a safety floor, not
// a full sandbox — hosts should still gate Bash via CanUseTool.
var catastrophicBash = []*regexp.Regexp{
	regexp.MustCompile(`\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*/(\s|$)`), // rm -rf /
	regexp.MustCompile(`:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;`),                               // fork bomb
	regexp.MustCompile(`\bmkfs\.\w+\s+/dev/`),                                                      // format a device
	regexp.MustCompile(`\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd)`),                                       // overwrite a disk
}

// disallowedAutoBackground lists commands that must never be silently moved to
// the background on timeout.
var disallowedAutoBackground = []string{"sleep"}

// NewBash builds the Bash tool: run a shell command with a timeout, optionally in
// the background.
func NewBash() CoreTool {
	props := map[string]any{
		"command":    map[string]any{"type": "string", "description": "The shell command to execute."},
		"timeout_ms": map[string]any{"type": "integer", "description": "Timeout in ms (default 120000, max 600000)."},
	}
	desc := "Executes a shell command in the working directory and returns combined stdout and stderr. Use for running tests, builds, package managers, and git. Prefer the dedicated file tools for file work."
	// Only advertise the cap where it is actually enforced: the guard needs
	// /proc (linux) and the task manager (runBashSync has no watchdog), and the
	// cap can be switched off outright. Promising a bound that never fires is
	// worse than saying nothing.
	if limit := memCapBytes(); memGuardSupported && limit > 0 && !BackgroundTasksDisabled() {
		desc += fmt.Sprintf(" A per-command memory cap (%s) kills a command whose process group exceeds it; the error names the cap.", humanBytes(limit))
	}
	prompt := "The command runs through the shell, so you are not limited to one command: chain steps with && or ;, use pipes, and pass multi-line scripts or heredocs. For anything beyond a simple one-liner — loops, parsing, string/JSON handling, HTTP logic, computation — prefer writing a short script (Python, Node, shell, etc.) and executing it (e.g. via a heredoc or by writing a file then running it) rather than forcing everything into a single command. A timeout (default 120s, max 600s) applies. Avoid commands needing interactive input."
	if !BackgroundTasksDisabled() {
		props["run_in_background"] = map[string]any{
			"type":        "boolean",
			"description": "Run in the background and return immediately; read its output later with TaskOutput or Read, and you are notified when it completes. A foreground command that exceeds its timeout is also moved to the background automatically.",
		}
		desc += " Set run_in_background to return immediately and read the output later."
	}
	return Build(Spec{
		Name:        "Bash",
		Description: desc,
		Prompt:      prompt,
		Schema: map[string]any{
			"type":       "object",
			"properties": props,
			"required":   []any{"command"},
		},
		Permissions: bashPermissions,
		Run:         runBash,
	})
}

func bashPermissions(_ context.Context, input json.RawMessage, _ permission.Context) permission.Decision {
	var in struct {
		Command string `json:"command"`
	}
	_ = json.Unmarshal(input, &in)
	if bad := catastrophicCommand(in.Command); bad != "" {
		return permission.Denied(bad)
	}
	return permission.AskUser("run shell command: " + in.Command)
}

// catastrophicCommand returns a denial message when cmd matches the safety
// floor, "" otherwise. Shared by every entry point that runs shell text, so the
// floor cannot be bypassed by using a different tool to reach a shell.
func catastrophicCommand(cmd string) string {
	for _, re := range catastrophicBash {
		if re.MatchString(cmd) {
			return "denied: command appears irreversibly destructive and is blocked by the safety floor"
		}
	}
	return ""
}

func runBash(ctx context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		Command         string `json:"command"`
		TimeoutMS       int    `json:"timeout_ms"`
		RunInBackground bool   `json:"run_in_background"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	if strings.TrimSpace(in.Command) == "" {
		return Errorf("Error: command is required"), nil
	}
	timeout := 120 * time.Second
	if in.TimeoutMS > 0 {
		timeout = time.Duration(in.TimeoutMS) * time.Millisecond
		if timeout > 600*time.Second {
			timeout = 600 * time.Second
		}
	}

	// Background-capable path: requires a task manager on the context.
	if tc != nil && tc.Tasks != nil && !BackgroundTasksDisabled() {
		return runBashManaged(ctx, tc, in.Command, timeout, in.RunInBackground)
	}
	// Fallback: synchronous execution (unchanged legacy behavior).
	return runBashSync(ctx, tc, in.Command, timeout)
}

// runBashManaged runs the command through the task manager, supporting explicit
// background launch and timeout auto-backgrounding.
func runBashManaged(ctx context.Context, tc *ToolContext, command string, timeout time.Duration, explicitBG bool) (Result, error) {
	spec := SpawnSpec{Command: command, Kind: KindBash, WorkingDir: tc.WorkingDir, Description: short(command), Env: tc.Env}

	if explicitBG {
		t, err := tc.Tasks.Spawn(spec)
		if err != nil {
			return Errorf("Error: failed to start background command: " + err.Error()), nil
		}
		tc.Tasks.Background(t.ID)
		return Text(fmt.Sprintf(
			"Running in background as %s. Output file: %s\nUse TaskOutput{task_id:%q} or Read to read its output; you are notified when it completes.",
			t.ID, t.OutputPath, t.ID)), nil
	}

	t, finished, output, err := tc.Tasks.RunForeground(ctx, spec, timeout)
	if err != nil {
		if t != nil {
			tc.Tasks.Forget(t.ID)
		}
		return Errorf("Error: " + err.Error()), nil
	}
	if finished {
		info, _ := tc.Tasks.Get(t.ID)
		tc.Tasks.Forget(t.ID)
		if info.Status == TaskFailed {
			if strings.TrimSpace(output) == "" {
				output = "(no output)"
			}
			// A watchdog kill carries no exit code of its own; the bare code
			// would read as a cryptic failure, so name the cap that fired and
			// what to change about the command.
			switch info.Reason {
			case terminateMemory:
				return Errorf(Capture(tc, output) + fmt.Sprintf(
					"\n\n[terminated: the command's process group exceeded the %s per-command memory cap and was killed — stream or page the data instead of accumulating it in memory]",
					humanBytes(memCapBytes()))), nil
			case terminateOversize:
				return Errorf(Capture(tc, output) + fmt.Sprintf(
					"\n\n[terminated: output exceeded the %s per-task output cap]",
					humanBytes(maxTaskOutputBytes))), nil
			}
			return Errorf(fmt.Sprintf("%s\n\n[exit code %d]", Capture(tc, output), info.ExitCode)), nil
		}
		if strings.TrimSpace(output) == "" {
			output = "(command produced no output)"
		}
		return Text(Capture(tc, output)), nil
	}

	// Timed out while still running.
	if autoBackgroundAllowed(command) {
		tc.Tasks.Background(t.ID)
		return Text(fmt.Sprintf(
			"Command still running after %s; moved to background as %s. Output file: %s\nUse TaskOutput{task_id:%q} to read its output; you are notified when it completes.",
			timeout, t.ID, t.OutputPath, t.ID)), nil
	}
	out, _ := tc.Tasks.Output(t.ID, 0)
	tc.Tasks.Kill(t.ID)
	tc.Tasks.Forget(t.ID)
	return Errorf(Capture(tc, out) + fmt.Sprintf("\n\n[command timed out after %s]", timeout)), nil
}

// runBashSync is the original blocking implementation, used when no task manager
// is available (background execution disabled).
func runBashSync(ctx context.Context, tc *ToolContext, command string, timeout time.Duration) (Result, error) {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	shell, flags := shellCmd()
	cmd := exec.CommandContext(cctx, shell, append(flags, command)...)
	if tc != nil {
		cmd.Dir = tc.WorkingDir
		cmd.Env = withEnv(tc.Env)
	} else {
		cmd.Env = withEnv(nil)
	}
	// Same isolation as the managed path: the child gets its own process group,
	// so a timeout can kill the whole tree instead of orphaning grandchildren.
	// CommandContext's default killer signals only the direct child, which is
	// exactly the grandchild leak this replaces.
	setDetached(cmd)
	cmd.Cancel = func() error {
		if cmd.Process != nil {
			treeKill(cmd.Process)
		}
		return nil
	}
	out := &syncCapBuffer{max: maxTaskOutputBytes, cmd: cmd}
	cmd.Stdout = out
	cmd.Stderr = out
	err := cmd.Run()

	text := Capture(tc, out.String())
	if out.Overflowed() {
		return Errorf(text + fmt.Sprintf(
			"\n\n[terminated: output exceeded the %s per-command cap]",
			humanBytes(maxTaskOutputBytes))), nil
	}
	if cctx.Err() == context.DeadlineExceeded {
		return Errorf(text + fmt.Sprintf("\n\n[command timed out after %s]", timeout)), nil
	}
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			if text == "" {
				text = "(no output)"
			}
			return Errorf(fmt.Sprintf("%s\n\n[exit code %d]", text, ee.ExitCode())), nil
		}
		return Errorf(fmt.Sprintf("%s\n\n[failed to run: %v]", text, err)), nil
	}
	if strings.TrimSpace(text) == "" {
		text = "(command produced no output)"
	}
	return Text(text), nil
}

// syncCapBuffer bounds a foreground command's output held in memory.
//
// The managed path caps the output file on disk; this path has no file, so
// without a cap a command like `yes` grows a bytes.Buffer until the host is
// OOM-killed — taking the unsaved transcript with it. On overflow the process
// group is killed immediately and Overflowed reports the termination, matching
// the managed path's oversize behaviour.
type syncCapBuffer struct {
	mu   sync.Mutex
	buf  bytes.Buffer
	max  int
	over bool
	cmd  *exec.Cmd
}

func (b *syncCapBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	if !b.over {
		if room := b.max - b.buf.Len(); room > 0 {
			b.buf.Write(p[:min(len(p), room)])
		}
		if b.buf.Len() >= b.max {
			b.over = true
			if b.cmd != nil {
				if proc := b.cmd.Process; proc != nil {
					go treeKill(proc) // outside the lock: the child is still writing
				}
			}
		}
	}
	b.mu.Unlock()
	// Report full acceptance: a short write would surface as an I/O error in the
	// child and change what the model sees, when the point is to terminate it.
	return len(p), nil
}

// String returns what was captured before the cap.
func (b *syncCapBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// Overflowed reports whether the cap was reached and the command killed.
func (b *syncCapBuffer) Overflowed() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.over
}

// autoBackgroundAllowed reports whether a timed-out command may be moved to the
// background instead of being killed.
func autoBackgroundAllowed(command string) bool {
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return false
	}
	base := fields[0]
	if i := strings.LastIndexAny(base, "/\\"); i >= 0 {
		base = base[i+1:]
	}
	for _, d := range disallowedAutoBackground {
		if base == d {
			return false
		}
	}
	return true
}
