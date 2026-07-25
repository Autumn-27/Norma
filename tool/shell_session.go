package tool

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hinshun/vt10x"
)

// Interactive PTY shell sessions (docs/交互式shell设计.md). A session is a persistent
// pseudo-terminal running an interactive program (msfconsole / ssh / a REPL / …) that
// the agent drives by sending input (text / named keys / raw bytes) and reading output
// incrementally (raw stream) or as a rendered screen snapshot (vt10x). Sessions live on
// the per-Session task Manager (tc.Tasks) and are cleaned up with it.

// Tunables.
const (
	sessRingCap     = 256 << 10 // per-session raw scrollback kept in memory
	sessDefQuietMs  = 8000      // silence that counts as "command done"
	sessDefGraceMs  = 15000     // startup grace before quiet can fire
	sessDefIdleMs   = 30 * 60 * 1000
	sessRereadMs    = 5 * 60 * 1000 // keep an exited session this long for re-reads
	sessDefRows     = 24
	sessDefCols     = 80
	sessDefReadLine = 200
	sessDefReadByte = 8192
	sessWaitCapMs   = 120000 // default hard wait ceiling for shell_send(wait)
)

// InteractiveShellDisabled reports whether interactive shell sessions are turned off
// via AGENT_CORE_DISABLE_INTERACTIVE_SHELL (mirrors BackgroundTasksDisabled).
func InteractiveShellDisabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENT_CORE_DISABLE_INTERACTIVE_SHELL"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// ptySession is one live interactive PTY.
type ptySession struct {
	id      string
	command string
	rows    int
	cols    int
	mode    string // interactive | hands-free | dispatch
	prompt  *regexp.Regexp

	cmd  *exec.Cmd
	ptmx *os.File

	mu       sync.Mutex
	buf      []byte // raw scrollback (ring, capped)
	total    int64  // total bytes ever produced (absolute cursor space)
	emu      vt10x.Terminal
	state    string // starting | running | exited | closed
	exitCode *int
	lastOut  time.Time
	exitedAt time.Time

	readCur  int64 // tool-side read cursor (shell_read without since_cursor continues here)
	notifyCh chan struct{} // pinged on new output
	done     chan struct{} // closed when the process exits
}

func (s *ptySession) readCursor() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readCur
}
func (s *ptySession) setReadCursor(c int64) {
	s.mu.Lock()
	s.readCur = c
	s.mu.Unlock()
}

// --- ring buffer + output ---

func (s *ptySession) append(p []byte) {
	s.mu.Lock()
	s.buf = append(s.buf, p...)
	s.total += int64(len(p))
	if len(s.buf) > sessRingCap {
		s.buf = s.buf[len(s.buf)-sessRingCap:]
	}
	if s.emu != nil {
		_, _ = s.emu.Write(p)
	}
	s.lastOut = time.Now()
	s.mu.Unlock()
	select {
	case s.notifyCh <- struct{}{}:
	default:
	}
}

func (s *ptySession) tailStr(n int) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if n <= 0 || n > len(s.buf) {
		return string(s.buf)
	}
	return string(s.buf[len(s.buf)-n:])
}

// readSince returns the output produced after absolute cursor c, advancing the
// returned cursor by exactly what is handed back (front-capped for pagination).
func (s *ptySession) readSince(c int64, maxLines, maxBytes int, drain bool) (out string, cursor int64, omitted bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	start := s.total - int64(len(s.buf))
	if c >= s.total {
		return "", s.total, false
	}
	from := c
	if from < start {
		from, omitted = start, true
	}
	seg := s.buf[from-start:]
	if !drain {
		if maxBytes > 0 && len(seg) > maxBytes {
			seg = seg[:maxBytes]
		}
		if maxLines > 0 {
			if idx := nthByte(seg, '\n', maxLines); idx >= 0 {
				seg = seg[:idx+1]
			}
		}
	}
	return string(seg), from + int64(len(seg)), omitted
}

func nthByte(b []byte, ch byte, n int) int {
	c := 0
	for i, x := range b {
		if x == ch {
			if c++; c == n {
				return i
			}
		}
	}
	return -1
}

// screen renders the vt10x-emulated terminal grid (lazy-init: the emulator is only
// created — and the scrollback replayed into it — on the first screen read).
func (s *ptySession) screen() (grid string, rows, cols, curX, curY int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.emu == nil {
		s.emu = vt10x.New(vt10x.WithSize(s.cols, s.rows))
		_, _ = s.emu.Write(s.buf)
	}
	cols, rows = s.emu.Size()
	var b strings.Builder
	for y := 0; y < rows; y++ {
		var line strings.Builder
		for x := 0; x < cols; x++ {
			ch := s.emu.Cell(x, y).Char
			if ch == 0 {
				ch = ' '
			}
			line.WriteRune(ch)
		}
		b.WriteString(strings.TrimRight(line.String(), " "))
		b.WriteByte('\n')
	}
	cur := s.emu.Cursor()
	return b.String(), rows, cols, cur.X, cur.Y
}

func (s *ptySession) running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == "starting" || s.state == "running"
}

func (s *ptySession) exit() *int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.exitCode
}

// waitComplete blocks until a completion signal: prompt/sentinel match on the tail,
// output silence for quiet, the process exits, or the deadline/ctx fires. Returns the
// reason. grace delays the first quiet check so a slow-starting program isn't judged done.
func (s *ptySession) waitComplete(ctx context.Context, quiet, grace, timeout time.Duration, prompt *regexp.Regexp, sentinel string) string {
	dl := time.NewTimer(timeout)
	defer dl.Stop()
	first := quiet
	if grace > first {
		first = grace
	}
	qt := time.NewTimer(first)
	defer qt.Stop()
	for {
		select {
		case <-s.done:
			return "exited"
		case <-ctx.Done():
			return "timeout"
		case <-dl.C:
			return "timeout"
		case <-s.notifyCh:
			tail := s.tailStr(4096)
			if prompt != nil && prompt.MatchString(tail) {
				return "prompt"
			}
			if sentinel != "" && strings.Contains(tail, sentinel) {
				return "sentinel"
			}
			if !qt.Stop() {
				select {
				case <-qt.C:
				default:
				}
			}
			qt.Reset(quiet)
		case <-qt.C:
			return "quiet"
		}
	}
}

// --- Manager session methods ---

func (m *Manager) reapSessions() {
	now := time.Now()
	m.sessMu.Lock()
	var kill []*ptySession
	for id, s := range m.sessions {
		s.mu.Lock()
		idle := now.Sub(s.lastOut)
		exited := s.state == "exited"
		gone := exited && !s.exitedAt.IsZero() && now.Sub(s.exitedAt) > sessRereadMs*time.Millisecond
		stale := !exited && idle > sessDefIdleMs*time.Millisecond
		s.mu.Unlock()
		if gone || stale {
			kill = append(kill, s)
			delete(m.sessions, id)
		}
	}
	m.sessMu.Unlock()
	for _, s := range kill {
		s.kill()
	}
}

func (s *ptySession) kill() {
	s.mu.Lock()
	s.state = "closed"
	ptmx, cmd := s.ptmx, s.cmd
	s.mu.Unlock()
	if ptmx != nil {
		_ = ptmx.Close()
	}
	if cmd != nil && cmd.Process != nil {
		treeKill(cmd.Process)
	}
}

func (m *Manager) closeAllSessions() {
	m.sessMu.Lock()
	ss := make([]*ptySession, 0, len(m.sessions))
	for _, s := range m.sessions {
		ss = append(ss, s)
	}
	m.sessions = map[string]*ptySession{}
	m.sessMu.Unlock()
	for _, s := range ss {
		s.kill()
	}
}

func (m *Manager) session(id string) (*ptySession, bool) {
	m.sessMu.Lock()
	defer m.sessMu.Unlock()
	s, ok := m.sessions[id]
	return s, ok
}

// openSession starts an interactive program in a fresh PTY and begins its read loop.
func (m *Manager) openSession(command, workDir string, env []string, rows, cols int, mode string, prompt *regexp.Regexp) (*ptySession, error) {
	if !ptySupported() {
		return nil, fmt.Errorf("交互式 shell(PTY) 在当前平台不支持")
	}
	m.reapSessions()
	if rows <= 0 {
		rows = sessDefRows
	}
	if cols <= 0 {
		cols = sessDefCols
	}
	shell, flags := shellCmd()
	cmd := exec.CommandContext(m.rootCtx, shell, append(flags, command)...)
	cmd.Dir = workDir
	if e := withEnv(env); e != nil {
		cmd.Env = e
	}
	ptmx, err := startPTY(cmd, rows, cols)
	if err != nil {
		return nil, err
	}
	m.sessMu.Lock()
	m.sessSeq++
	id := fmt.Sprintf("sh_%d", m.sessSeq)
	m.sessMu.Unlock()

	s := &ptySession{
		id: id, command: command, rows: rows, cols: cols, mode: mode, prompt: prompt,
		cmd: cmd, ptmx: ptmx, state: "running", lastOut: time.Now(),
		notifyCh: make(chan struct{}, 1), done: make(chan struct{}),
	}
	m.sessMu.Lock()
	m.sessions[id] = s
	m.sessMu.Unlock()

	go m.readLoop(s)
	return s, nil
}

func (m *Manager) readLoop(s *ptySession) {
	b := make([]byte, 32*1024)
	for {
		n, err := s.ptmx.Read(b)
		if n > 0 {
			s.append(b[:n])
		}
		if err != nil {
			break
		}
	}
	_ = s.cmd.Wait()
	s.mu.Lock()
	if s.state != "closed" {
		s.state = "exited"
	}
	if s.cmd.ProcessState != nil {
		ec := s.cmd.ProcessState.ExitCode()
		s.exitCode = &ec
	}
	s.exitedAt = time.Now()
	s.mu.Unlock()
	close(s.done)
}

// dispatchWaiter (dispatch mode) waits for the first completion signal and enqueues a
// task-notification with the tail, so the model is woken between turns to review it.
func (m *Manager) dispatchWaiter(s *ptySession, quiet, grace, timeout time.Duration) {
	by := s.waitComplete(m.rootCtx, quiet, grace, timeout, s.prompt, "")
	from := s.total - 4096
	if from < 0 {
		from = 0
	}
	tail, _, _ := s.readSince(from, 0, 4096, true)
	status := TaskCompleted
	if !s.running() {
		status = TaskCompleted
	}
	m.enqueueSessNote(s, KindBash, status, fmt.Sprintf("交互会话 %s（%s）%s。用 shell_read(session_id=%q) 继续，或 shell_close 结束。\n%s",
		s.id, short(s.command), by, s.id, tail))
}

// enqueueSessNote queues a task-notification for a session so the model is woken
// between turns (dispatch completion / monitor pattern hit / session exit).
func (m *Manager) enqueueSessNote(s *ptySession, kind TaskKind, status TaskStatus, summary string) {
	m.mu.Lock()
	m.pending = append(m.pending, Notification{TaskID: s.id, Kind: kind, Status: status, Summary: summary})
	m.mu.Unlock()
}

// monitorWaiter (monitor mode) keeps the session alive and watches its output for any
// of the given patterns; each pattern fires a one-shot notification when first matched.
// A session exit fires a final notification. The session is NOT killed on a match.
func (m *Manager) monitorWaiter(s *ptySession, pats []*regexp.Regexp) {
	fired := make([]bool, len(pats))
	for {
		select {
		case <-m.rootCtx.Done():
			return
		case <-s.done:
			from := s.total - 4096
			if from < 0 {
				from = 0
			}
			tail, _, _ := s.readSince(from, 0, 4096, true)
			m.enqueueSessNote(s, KindMonitor, TaskCompleted,
				fmt.Sprintf("监听会话 %s（%s）已结束(exit %s)。\n%s", s.id, short(s.command), fmtExit(s.exit()), tail))
			return
		case <-s.notifyCh:
			tail := s.tailStr(8192)
			for i, p := range pats {
				if !fired[i] && p.MatchString(tail) {
					fired[i] = true
					m.enqueueSessNote(s, KindMonitor, TaskRunning,
						fmt.Sprintf("监听会话 %s 命中 /%s/：\n%s", s.id, p.String(), tail))
				}
			}
		}
	}
}

// --- input building ---

var namedKeys = map[string]string{
	"enter": "\r", "return": "\r", "tab": "\t", "esc": "\x1b", "escape": "\x1b",
	"backspace": "\x7f", "space": " ", "up": "\x1b[A", "down": "\x1b[B",
	"right": "\x1b[C", "left": "\x1b[D", "home": "\x1b[H", "end": "\x1b[F",
	"pageup": "\x1b[5~", "pagedown": "\x1b[6~", "delete": "\x1b[3~", "insert": "\x1b[2~",
}

func keyToBytes(k string) (string, bool) {
	k = strings.ToLower(strings.TrimSpace(k))
	if v, ok := namedKeys[k]; ok {
		return v, true
	}
	if strings.HasPrefix(k, "ctrl+") && len(k) == 6 {
		c := k[5]
		switch {
		case c >= 'a' && c <= 'z':
			return string([]byte{c & 0x1f}), true
		case c == '[':
			return "\x1b", true
		case c == '\\':
			return "\x1c", true
		}
	}
	return "", false
}

func hexToBytes(items []string) []byte {
	var out []byte
	for _, h := range items {
		h = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(h)), "0x")
		if v, err := strconv.ParseUint(h, 16, 8); err == nil {
			out = append(out, byte(v))
		}
	}
	return out
}
