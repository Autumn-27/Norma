package tool

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/permission"
)

func run(t *testing.T, tl CoreTool, dir string, in any) Result {
	t.Helper()
	raw, _ := json.Marshal(in)
	if err := ValidateInput(tl.InputSchema(), raw); err != nil {
		t.Fatalf("%s schema: %v", tl.Name(), err)
	}
	res, err := tl.Call(context.Background(), raw, &ToolContext{WorkingDir: dir})
	if err != nil {
		t.Fatalf("%s: %v", tl.Name(), err)
	}
	return res
}

func TestFileTools(t *testing.T) {
	dir := t.TempDir()
	run(t, NewWrite(), dir, map[string]any{"file_path": "a.txt", "content": "hello\nworld\n"})
	if b, _ := os.ReadFile(filepath.Join(dir, "a.txt")); string(b) != "hello\nworld\n" {
		t.Fatalf("write: %q", b)
	}
	res := run(t, NewRead(), dir, map[string]any{"file_path": "a.txt"})
	if !strings.Contains(res.Flatten(), "1\thello") {
		t.Fatalf("read: %q", res.Flatten())
	}
	run(t, NewEdit(), dir, map[string]any{"file_path": "a.txt", "old_string": "world", "new_string": "gophers"})
	if b, _ := os.ReadFile(filepath.Join(dir, "a.txt")); string(b) != "hello\ngophers\n" {
		t.Fatalf("edit: %q", b)
	}
	run(t, NewWrite(), dir, map[string]any{"file_path": "b.txt", "content": "x x"})
	if r := run(t, NewEdit(), dir, map[string]any{"file_path": "b.txt", "old_string": "x", "new_string": "y"}); !r.IsError {
		t.Fatalf("expected non-unique edit error")
	}
}

func TestSearchTools(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "one.go"), []byte("package main\nfunc Foo(){}\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "two.txt"), []byte("Foo lives\n"), 0o644)
	res := run(t, NewGrep(), dir, map[string]any{"pattern": "Foo", "glob": "*.go", "output_mode": "files_with_matches"})
	if !strings.Contains(res.Flatten(), "one.go") || strings.Contains(res.Flatten(), "two.txt") {
		t.Fatalf("grep glob: %q", res.Flatten())
	}
	res = run(t, NewGlob(), dir, map[string]any{"pattern": "**/*.go"})
	if !strings.Contains(res.Flatten(), "one.go") {
		t.Fatalf("glob: %q", res.Flatten())
	}
}

func TestGrepContextLines(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "f.txt"), []byte("a\nb\nMATCH\nd\ne\n"), 0o644)
	// -C 1 should include the line before and after the match, with the match
	// line using ':' and context lines using '-'.
	res := run(t, NewGrep(), dir, map[string]any{"pattern": "MATCH", "-C": 1})
	out := res.Flatten()
	for _, want := range []string{"f.txt-2-b", "f.txt:3:MATCH", "f.txt-4-d"} {
		if !strings.Contains(out, want) {
			t.Fatalf("context output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "f.txt-1-a") {
		t.Fatalf("-C 1 should not reach line 1:\n%s", out)
	}
}

func TestBashAndSafetyFloor(t *testing.T) {
	dir := t.TempDir()
	res := run(t, NewBash(), dir, map[string]any{"command": "echo hi"})
	if !strings.Contains(res.Flatten(), "hi") {
		t.Fatalf("bash: %q", res.Flatten())
	}
	// Safety floor: rm -rf / must be denied by CheckPermissions.
	bash := NewBash()
	raw, _ := json.Marshal(map[string]any{"command": "rm -rf /"})
	if d := bash.CheckPermissions(context.Background(), raw, permission.Context{}); d.Behavior != permission.Deny {
		t.Fatalf("expected deny for rm -rf /, got %v", d.Behavior)
	}
}

func TestSchemaValidation(t *testing.T) {
	// Missing required field.
	raw, _ := json.Marshal(map[string]any{"offset": 1})
	if err := ValidateInput(NewRead().InputSchema(), raw); err == nil {
		t.Fatal("expected missing file_path error")
	}
	// Wrong type.
	raw, _ = json.Marshal(map[string]any{"file_path": 123})
	if err := ValidateInput(NewRead().InputSchema(), raw); err == nil {
		t.Fatal("expected type error")
	}
}
