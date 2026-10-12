package tool

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"
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
			return cutHead(s, max) + fmt.Sprintf(
				"\n\n... <persisted-output>[Output too large: full %d bytes / %d lines.Full output saved to  %s </persisted-output>",
				len(s), lines, path)
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
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
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

// truncate trims s to max bytes, keeping head and tail. Both cuts are aligned to
// rune boundaries (see cutHead/cutTail): slicing bytes blindly can split a
// multi-byte character, and the mojibake then travels into the model's context
// (and into transcripts) as invalid UTF-8.
func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	half := max / 2
	return cutHead(s, half) + fmt.Sprintf("\n\n... [%d characters truncated] ...\n\n", len(s)-max) + cutTail(s, half)
}

// cutHead returns the longest prefix of s that is at most max bytes AND ends on a
// rune boundary. max<len(s) is assumed (callers check the length first).
func cutHead(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(s) <= max {
		return s
	}
	for max > 0 && !utf8.RuneStart(s[max]) {
		max--
	}
	return s[:max]
}

// cutTail returns the longest suffix of s that is at most max bytes AND starts on
// a rune boundary.
func cutTail(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(s) <= max {
		return s
	}
	start := len(s) - max
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[start:]
}
