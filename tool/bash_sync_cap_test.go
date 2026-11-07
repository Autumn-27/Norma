package tool

import (
	"bytes"
	"testing"
)

// The sync Bash path keeps output in memory, so it needs a hard cap: a command
// like `yes` otherwise grows the buffer until the host is OOM-killed.
func TestSyncCapBufferStopsAtTheCap(t *testing.T) {
	b := &syncCapBuffer{max: 1024}
	chunk := bytes.Repeat([]byte("x"), 512)
	for range 10 {
		if n, err := b.Write(chunk); n != len(chunk) || err != nil {
			t.Fatalf("Write = %d, %v; want a full write so the child never sees an I/O error", n, err)
		}
	}
	if got := len(b.String()); got != 1024 {
		t.Fatalf("captured %d bytes, want exactly the cap of 1024", got)
	}
	if !b.Overflowed() {
		t.Fatal("Overflowed = false after exceeding the cap")
	}
}

// Under the cap nothing is dropped and nothing is reported.
func TestSyncCapBufferBelowTheCap(t *testing.T) {
	b := &syncCapBuffer{max: 1024}
	b.Write([]byte("hello"))
	if b.String() != "hello" {
		t.Fatalf("String = %q", b.String())
	}
	if b.Overflowed() {
		t.Fatal("Overflowed = true below the cap")
	}
}
