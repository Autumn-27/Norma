package tool

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestAtomicReplaceFailurePreservesOriginal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "target")
	if err := os.WriteFile(path, []byte("original"), 0o640); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("injected partial write failure")
	err := atomicReplace(path, func(f *os.File) error {
		if _, err := f.WriteString("partial"); err != nil {
			return err
		}
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("got %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "original" {
		t.Fatalf("original damaged: %q %v", got, err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 {
		t.Fatalf("temporary file leaked: %v %v", entries, err)
	}
}

func TestMultiEditReplacesInodeAndPreservesSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix inode and symlink semantics")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, []byte("old"), 0o751); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink("target", link); err != nil {
		t.Fatal(err)
	}
	// A reader holding the old inode must never see a truncated/partial file.
	reader, err := os.Open(target)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	before, _ := reader.Stat()
	r := run(t, NewMultiEdit(), dir, map[string]any{"file_path": "link", "edits": []any{map[string]any{"old_string": "old", "new_string": "replacement"}}})
	if r.IsError {
		t.Fatal(r.Flatten())
	}
	after, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if os.SameFile(before, after) {
		t.Fatal("file was overwritten in place")
	}
	if after.Mode().Perm() != before.Mode().Perm() {
		t.Fatal("permissions changed")
	}
	buf := make([]byte, 3)
	if _, err := reader.Read(buf); err != nil || string(buf) != "old" {
		t.Fatalf("old reader changed: %q %v", buf, err)
	}
	got, err := os.ReadFile(link)
	if err != nil || string(got) != "replacement" {
		t.Fatalf("new content: %q %v", got, err)
	}
	if dest, err := os.Readlink(link); err != nil || dest != "target" {
		t.Fatalf("symlink replaced: %q %v", dest, err)
	}
}
