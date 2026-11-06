package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/Autumn-27/norma/permission"
)

const defaultMaxOutput = 30000

func maxOut(tc *ToolContext) int {
	if tc != nil && tc.MaxOutputChars > 0 {
		return tc.MaxOutputChars
	}
	return defaultMaxOutput
}

var spillSeq atomic.Int64

// Capture bounds a tool's textual output for the model. When the output exceeds
// the limit (ToolContext.MaxOutputChars, default 30000) and the host configured
// an OutputDir, the FULL output is written to a file there and a tight head + a
// pointer to that file is returned — so the model can read slices on demand
// (grep/sed) instead of losing the overflow. Without an OutputDir it falls back
// to a head+tail truncation (the overflow is discarded). Tools that produce
// potentially large output should return Capture(tc, output) rather than the raw
// string.
//
// The pointer is always an ABSOLUTE path: a relative one only resolves from the
// working directory it was computed against, and the reader is often somewhere
// else — a later command after a `cd`, a resumed session with a different
// WorkingDir, or an out-of-band tool (grep/sed) the model runs itself. Heads cut
// by Capture and tail-fallbacks in truncate() are aligned to UTF-8 rune
// boundaries, so a bounded result never carries a half-encoded character.
func Capture(tc *ToolContext, s string) string {
	max := maxOut(tc)
	if len(s) <= max {
		return s
	}
	if tc != nil && tc.OutputDir != "" {
		if path, err := spillOutput(tc.OutputDir, s); err == nil {
			lines := strings.Count(s, "\n") + 1
			// Two notice forms, longest first; a tight cap gets the short one.
			// The notice counts against max when it can, so the head shrinks to
			// make room. When even the short form does not fit, the head keeps its
			// full cap and the notice is appended past it: the pointer is the only
			// way back to the content, so it is the one thing worth spending bytes
			// outside the budget on. The bound below therefore applies to the
			// OUTPUT; a spill result is bounded to max + the notice.
			forms := []string{
				"\n\n... <persisted-output>[Output too large: full " +
					fmt.Sprintf("%d bytes / %d lines. Full output saved to  %s </persisted-output>", len(s), lines, path),
				"\n\n... <persisted-output>[Full output saved to  " + path + " </persisted-output>",
			}
			for _, notice := range forms {
				budget := max - len(notice)
				if budget < 0 {
					budget = max
				}
				return cutHead(s, budget) + notice
			}
		}
	}
	return truncate(s, max)
}

// alreadyCaptured reports whether s already carries a marker Capture leaves
// behind — a spill pointer or a head+tail truncation notice. It is how a global
// post-tool net recognises output a tool already ran through Capture itself.
//
// A false negative (legitimate text that happens to contain a marker) only means
// the guard fails to re-cap already-large output; it never corrupts. The markers
// are distinctive enough that this is vanishingly rare.
func alreadyCaptured(s string) bool {
	return strings.Contains(s, "<persisted-output>") || strings.Contains(s, "characters truncated]")
}

// CaptureOnce is Capture with an idempotency guard: output a tool already ran
// through Capture (so it carries a spill pointer or truncation notice) is
// returned unchanged. This lets a single post-tool net cap every tool's output
// without re-truncating or re-spilling what built-in / MCP / custom tools
// already handled themselves. Output within the limit is a no-op either way,
// since Capture returns it unchanged.
func CaptureOnce(tc *ToolContext, s string) string {
	if alreadyCaptured(s) {
		return s
	}
	return Capture(tc, s)
}

func spillOutput(dir, s string) (string, error) {
	// Resolve the directory first: the returned pointer is what the model will
	// read, so it must not depend on anyone's current working directory.
	//
	// A failure here is reported rather than papered over with the relative
	// path. filepath.Abs only fails when dir is relative and os.Getwd fails —
	// i.e. the process's cwd has been deleted, which is precisely when a
	// relative pointer is most useless. Capture's caller falls back to
	// truncate(), which hands the model bounded output instead of a path it
	// cannot resolve.
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("output-%d-%d.txt", time.Now().Unix(), spillSeq.Add(1))
	path := filepath.Join(abs, name)
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// resolvePath joins a relative path against the working directory.
func resolvePath(tc *ToolContext, p string) string {
	if filepath.IsAbs(p) || tc == nil || tc.WorkingDir == "" {
		return p
	}
	return filepath.Join(tc.WorkingDir, p)
}

// withinWorkspace reports whether p lands inside workdir. An empty workdir
// confines nothing (a host that did not name a workspace gets no boundary).
//
// Lexical resolution only: this stops a path ARGUMENT from climbing out with
// ".." or an absolute path. It does not follow symlinks, so it is not a defence
// against someone who can already plant a link inside the workspace.
func withinWorkspace(workdir, p string) bool {
	if workdir == "" {
		return true
	}
	root, err := filepath.Abs(workdir)
	if err != nil {
		return false
	}
	abs := p
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(root, abs)
	}
	abs = filepath.Clean(abs)
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return false
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	return true
}

// inputPath reads the path argument a file tool was given ("file_path" for the
// file tools, "path" for LS/Glob/Grep).
func inputPath(input json.RawMessage) string {
	var in struct {
		FilePath string `json:"file_path"`
		Path     string `json:"path"`
	}
	_ = json.Unmarshal(input, &in)
	if in.FilePath != "" {
		return in.FilePath
	}
	return in.Path
}

// readPathPermissions auto-allows a read that stays inside the workspace and
// asks for one that leaves it.
//
// Why this exists: read-only tools are auto-allowed by the pipeline, and Read
// takes an absolute path verbatim. Without this check, `Read ~/.ssh/id_rsa` is
// approved by nobody and its contents go into the model's context — that is, to
// the model provider.
func readPathPermissions(_ context.Context, input json.RawMessage, pc permission.Context) permission.Decision {
	if p := inputPath(input); p != "" && !withinWorkspace(pc.WorkingDir, p) {
		return permission.AskUser("read outside the workspace: " + p)
	}
	return permission.Allowed()
}

// editPathPermissions always asks. It returns Ask, not Allow, so acceptEdits'
// auto-allow does not fire on it (the pipeline never auto-allows a tool whose
// own check wants a human), and it names the escape when there is one.
func editPathPermissions(_ context.Context, input json.RawMessage, pc permission.Context) permission.Decision {
	p := inputPath(input)
	if p != "" && !withinWorkspace(pc.WorkingDir, p) {
		return permission.AskUser("edit outside the workspace: " + p)
	}
	return permission.AskUser("edit " + p)
}

// truncate trims s to at most max bytes, keeping head and tail. Both cuts are
// aligned to rune boundaries (see cutHead/cutTail): slicing bytes blindly can split
// a multi-byte character, and the mojibake then travels into the model's context
// (and into transcripts) as invalid UTF-8.
//
// The notice counts against max, so the result never exceeds what the caller
// asked for. A caller who says "30000" and gets 30_060 has no bound at all.
func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	notice := fmt.Sprintf("\n\n... [%d characters truncated] ...\n\n", len(s)-max)
	budget := max - len(notice)
	if budget <= 0 {
		return cutHead(s, max)
	}
	half := budget / 2
	return cutHead(s, half) + notice + cutTail(s, half)
}

// byteSeq is string or []byte: both index to a byte, so the rune-boundary
// primitives below serve either without converting (and allocating) between them.
type byteSeq interface{ string | []byte }

// runeSafeLen returns the largest n <= max with b[:n] ending on a rune boundary.
func runeSafeLen[T byteSeq](b T, max int) int {
	if max <= 0 {
		return 0
	}
	if len(b) <= max {
		return len(b)
	}
	for max > 0 && !utf8.RuneStart(b[max]) {
		max--
	}
	return max
}

// runeSafeStart returns the smallest n with b[n:] starting on a rune boundary,
// i.e. the number of leading UTF-8 continuation bytes to drop. Use it when the
// length is already right and only the boundary is in question — after slicing a
// ring buffer, or seeking a file to an arbitrary byte offset.
func runeSafeStart[T byteSeq](b T) int {
	i := 0
	for i < len(b) && !utf8.RuneStart(b[i]) {
		i++
	}
	return i
}

// cutHead returns the longest prefix of s that is at most max bytes AND ends on a
// rune boundary. max<len(s) is assumed (callers check the length first).
func cutHead(s string, max int) string { return s[:runeSafeLen(s, max)] }

// cutTail returns the longest suffix of s that is at most max bytes AND starts on
// a rune boundary.
func cutTail(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(s) <= max {
		return s
	}
	s = s[len(s)-max:]
	return s[runeSafeStart(s):]
}
