package tool

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

// The spill pointer is what the model reads later, often from a different working
// directory (a later command after a cd, a resumed session, or a grep the model
// runs itself) — so it must be absolute, and the file must exist exactly there.
func TestCaptureSpillPointerIsAbsolute(t *testing.T) {
	// A RELATIVE OutputDir under a WorkingDir is the case that used to produce a
	// relative pointer: it must come back absolute all the same.
	work := t.TempDir()
	// t.Chdir restores the old directory automatically and fails fast if this
	// test is ever made parallel — process-global cwd and t.Parallel() would
	// otherwise turn into order-dependent failures elsewhere in the package.
	t.Chdir(work)

	tc := &ToolContext{WorkingDir: work, OutputDir: "cmd-output", MaxOutputChars: 200}
	out := Capture(tc, strings.Repeat("z", 5000))
	if !strings.Contains(out, "<persisted-output>") {
		t.Fatalf("oversized output should spill: %q", out)
	}
	ref := extractPointer(t, out)
	if !filepath.IsAbs(ref) {
		t.Fatalf("spill pointer must be absolute, got %q", ref)
	}
	body, err := os.ReadFile(ref)
	if err != nil {
		t.Fatalf("spill file not readable at the pointer: %v", err)
	}
	if len(body) != 5000 {
		t.Fatalf("spill file holds %d bytes, want the full 5000", len(body))
	}
}

// A bounded result must never carry a half-encoded character: both the spilled
// head and the head+tail fallback used without an OutputDir cut on rune
// boundaries, even when the cut lands mid-character.
func TestCaptureCutsOnRuneBoundaries(t *testing.T) {
	// 3-byte runes with a length that forces the cut into the middle of one.
	long := strings.Repeat("测", 400)
	if len(long)%3 != 0 {
		t.Fatal("fixture not aligned")
	}
	for _, tc := range []*ToolContext{
		{MaxOutputChars: 101},                         // truncate fallback, odd byte limit
		{OutputDir: t.TempDir(), MaxOutputChars: 101}, // spill head
	} {
		out := Capture(tc, long)
		if !utf8.ValidString(out) {
			t.Fatalf("captured output is not valid UTF-8 (pointer=%v)", tc.OutputDir != "")
		}
		if !strings.HasPrefix(out, "测") {
			t.Fatalf("head must start on a rune boundary: %q", out[:30])
		}
	}
	// The fallback keeps a tail too, and that tail must also start on a boundary.
	out := Capture(&ToolContext{MaxOutputChars: 101}, long+"尾部")
	if !utf8.ValidString(out) || !strings.HasSuffix(out, "尾部") {
		t.Fatalf("truncated tail broken: valid=%v out=%q", utf8.ValidString(out), out)
	}
	if !strings.Contains(out, "characters truncated]") {
		t.Fatalf("truncation marker lost: %q", out)
	}
}

// extractPointer pulls the path out of a "<persisted-output>" notice.
func extractPointer(t *testing.T, out string) string {
	t.Helper()
	const marker = "Full output saved to  "
	i := strings.Index(out, marker)
	if i < 0 {
		t.Fatalf("spill notice missing: %q", out)
	}
	rest := out[i+len(marker):]
	j := strings.Index(rest, " </persisted-output>")
	if j < 0 {
		t.Fatalf("spill notice malformed: %q", out)
	}
	return strings.TrimSpace(rest[:j])
}
