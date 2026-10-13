package tool

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestRuneSafeHelpers(t *testing.T) {
	// "日" is 3 bytes: E6 97 A5.
	s := "a日b"
	for max, want := range map[int]string{0: "", 1: "a", 2: "a", 3: "a", 4: "a日", 5: "a日b", 9: "a日b"} {
		if got := s[:runeSafeLen(s, max)]; got != want {
			t.Errorf("runeSafeLen(%q, %d) cut to %q, want %q", s, max, got, want)
		}
	}
	if got := runeSafeStart(s); got != 0 {
		t.Errorf("runeSafeStart on an aligned string = %d, want 0", got)
	}
	// Slice into the middle of 日: two continuation bytes must be skipped.
	mid := s[2:]
	if got := runeSafeStart(mid); got != 2 {
		t.Errorf("runeSafeStart(%q) = %d, want 2", mid, got)
	}
	if got := mid[runeSafeStart(mid):]; got != "b" {
		t.Errorf("aligned tail = %q, want %q", got, "b")
	}
	// Byte-slice instantiation must behave identically.
	if got := runeSafeLen([]byte(s), 4); got != 4 {
		t.Errorf("runeSafeLen([]byte) = %d, want 4", got)
	}
}

// tailFile seeks to size-maxBytes, an arbitrary byte offset that lands mid-rune
// for any multi-byte output. This tail goes straight to the model through
// Manager.Output/TaskOutput, below the harness capture limit, so nothing else
// would fix it up.
func TestTailFileStartsOnRuneBoundary(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.log")
	// 3-byte runes against a limit that is not a multiple of 3 guarantees the
	// naive offset splits one.
	body := strings.Repeat("日", 400)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, maxBytes := range []int{100, 101, 102, 299} {
		out, err := tailFile(path, maxBytes)
		if err != nil {
			t.Fatalf("maxBytes=%d: %v", maxBytes, err)
		}
		if !utf8.ValidString(out) {
			t.Errorf("maxBytes=%d: tail is not valid UTF-8", maxBytes)
		}
		if strings.Contains(out, "�") {
			t.Errorf("maxBytes=%d: tail carries a replacement character", maxBytes)
		}
		if !strings.Contains(out, "earlier output omitted") {
			t.Errorf("maxBytes=%d: truncation notice missing: %q", maxBytes, out)
		}
	}

	// A file under the limit is returned whole and untouched.
	small := filepath.Join(dir, "small.log")
	if err := os.WriteFile(small, []byte("日本語"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := tailFile(small, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if out != "日本語" {
		t.Errorf("small file = %q, want %q", out, "日本語")
	}
}

// short() feeds notification summaries, so a CJK command must not be cut into a
// replacement character there either.
func TestShortCutsOnRuneBoundary(t *testing.T) {
	// One ASCII byte shifts the 57-byte cut off a multiple of 3, so it lands
	// inside a rune. (Pure CJK would make 57 a boundary by coincidence and the
	// naive s[:57] would pass.)
	out := short("x" + strings.Repeat("日", 40))
	if !utf8.ValidString(out) {
		t.Fatalf("short() produced invalid UTF-8: %q", out)
	}
	if strings.Contains(out, "�") {
		t.Errorf("short() produced a replacement character: %q", out)
	}
	if !strings.HasSuffix(out, "...") {
		t.Errorf("short() dropped the ellipsis: %q", out)
	}
	if len(out) > 60 {
		t.Errorf("short() returned %d bytes, want <= 60", len(out))
	}
}

// readSince paginates with an absolute cursor. Cutting a page on a rune boundary
// must not lose bytes: the cursor advances by exactly what was handed back, so
// the next page resumes at the boundary and the two concatenate to the original.
func TestReadSinceCutsOnRuneBoundaryWithoutLosingBytes(t *testing.T) {
	body := strings.Repeat("日", 100) // 300 bytes
	s := &ptySession{buf: []byte(body), total: int64(len(body))}

	var got strings.Builder
	cursor := int64(0)
	for range 100 { // bounded: guards against a cursor that fails to advance
		// 100 is not a multiple of 3, so every page boundary splits a rune.
		out, next, _ := s.readSince(cursor, 0, 100, false)
		if out == "" {
			break
		}
		if !utf8.ValidString(out) {
			t.Fatalf("page at cursor %d is not valid UTF-8", cursor)
		}
		got.WriteString(out)
		if next <= cursor {
			t.Fatalf("cursor did not advance: %d -> %d", cursor, next)
		}
		cursor = next
	}
	if got.String() != body {
		t.Errorf("paginated read lost or duplicated bytes: got %d bytes, want %d", got.Len(), len(body))
	}
}

// When the ring buffer has dropped old output, the requested cursor is clamped
// to the buffer base — a raw byte offset that can land mid-rune. The returned
// cursor must account for the continuation bytes skipped, or the caller's next
// read is misaligned.
func TestReadSinceAlignsAfterDroppedOutput(t *testing.T) {
	body := strings.Repeat("日", 100)
	// total > len(buf): the first 2 bytes of the stream were dropped, so the
	// buffer now begins inside the first rune.
	s := &ptySession{buf: []byte(body)[2:], total: int64(len(body))}

	out, cursor, omitted := s.readSince(0, 0, 0, true)
	if !omitted {
		t.Error("omitted must be reported when the cursor predates the buffer")
	}
	if !utf8.ValidString(out) {
		t.Fatalf("output is not valid UTF-8: starts %q", out[:min(6, len(out))])
	}
	if want := strings.Repeat("日", 99); out != want {
		t.Errorf("got %d bytes, want the 99 whole runes after the partial one (%d bytes)", len(out), len(want))
	}
	if cursor != s.total {
		t.Errorf("cursor = %d, want %d — it must equal the first byte not returned", cursor, s.total)
	}
}
