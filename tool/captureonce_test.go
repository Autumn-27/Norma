package tool

import (
	"os"
	"strings"
	"testing"
)

// First pass spills oversized output; a second pass over that result must not
// spill again or change it — the marker makes CaptureOnce idempotent, which is
// what lets a global post-tool net coexist with tools that Capture themselves.
func TestCaptureOnceSpillIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	tc := &ToolContext{WorkingDir: dir, OutputDir: dir, MaxOutputChars: 100}
	long := strings.Repeat("x", 5000)

	first := CaptureOnce(tc, long)
	if !strings.Contains(first, "<persisted-output>") {
		t.Fatalf("first CaptureOnce should spill oversized output: %q", first)
	}
	if files, _ := os.ReadDir(dir); len(files) != 1 {
		t.Fatalf("expected exactly 1 spill file, got %d", len(files))
	}

	second := CaptureOnce(tc, first)
	if second != first {
		t.Fatalf("CaptureOnce not idempotent on spilled output:\n first=%q\n second=%q", first, second)
	}
	if files, _ := os.ReadDir(dir); len(files) != 1 {
		t.Fatalf("second CaptureOnce spilled again: %d files", len(files))
	}
}

// Without an OutputDir, Capture falls back to head+tail truncation (a different
// marker). CaptureOnce must recognise that too and not re-truncate.
func TestCaptureOnceTruncationIsIdempotent(t *testing.T) {
	tc := &ToolContext{MaxOutputChars: 100}
	long := strings.Repeat("y", 5000)

	first := CaptureOnce(tc, long)
	if !strings.Contains(first, "characters truncated]") {
		t.Fatalf("expected truncation marker: %q", first)
	}
	if second := CaptureOnce(tc, first); second != first {
		t.Fatalf("CaptureOnce re-truncated already-truncated output")
	}
}

func TestCaptureOncePassesShortOutput(t *testing.T) {
	tc := &ToolContext{MaxOutputChars: 100}
	if got := CaptureOnce(tc, "ok"); got != "ok" {
		t.Fatalf("short output changed: %q", got)
	}
}
