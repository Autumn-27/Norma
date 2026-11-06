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

func TestWithinWorkspace(t *testing.T) {
	root := t.TempDir()
	inside := []string{"a.go", "./sub/a.go", root + "/a.go", "sub/../a.go"}
	for _, p := range inside {
		if !withinWorkspace(root, p) {
			t.Errorf("%q should be inside %q", p, root)
		}
	}
	outside := []string{"../a.go", "/etc/passwd", root + "/../x", "../../x"}
	for _, p := range outside {
		if withinWorkspace(root, p) {
			t.Errorf("%q should be outside %q", p, root)
		}
	}
	// No workspace named: nothing to confine against.
	if !withinWorkspace("", "/etc/passwd") {
		t.Error("an empty working dir must confine nothing")
	}
}

// decide runs the real tool's permission check through the real pipeline, which
// is the only combination that matters: a tool that asks is useless if the
// pipeline auto-allows read-only tools past it.
// asked reports whether the pipeline consulted the approval callback, i.e.
// whether the tool's own verdict escaped the auto-allow shortcuts.
func decide(t *testing.T, tool CoreTool, input map[string]any, workdir string) (permission.Behavior, bool) {
	t.Helper()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	pc := permission.Context{Mode: permission.ModeDefault, WorkingDir: workdir}
	asked := false
	b := permission.Evaluate(context.Background(), permission.Request{
		ToolName:     tool.Name(),
		Input:        raw,
		IsReadOnly:   tool.IsReadOnly(raw),
		ToolDecision: tool.CheckPermissions(context.Background(), raw, pc),
		Ctx:          pc,
		Ask: func(context.Context, string, json.RawMessage, permission.Context) permission.Decision {
			asked = true
			return permission.Allowed()
		},
	}).Behavior
	return b, asked
}

// A read inside the workspace is auto-allowed; the same read with an absolute
// path outside it must stop and ask. This is the check that keeps `Read
// ~/.ssh/id_rsa` from reaching the model without a human seeing it.
func TestReadConfinesToWorkspace(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.go"), []byte("package a"), 0o644); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	read := NewRead()

	// Sanity: the inside read really is the auto-allow path.
	pc := permission.Context{Mode: permission.ModeDefault, WorkingDir: root}
	raw, _ := json.Marshal(map[string]any{"file_path": "a.go"})
	if d := read.CheckPermissions(context.Background(), raw, pc); d.Behavior != permission.Allow {
		t.Fatalf("a read inside the workspace should self-allow, got %v", d.Behavior)
	}
	if b, asked := decide(t, read, map[string]any{"file_path": "a.go"}, root); b != permission.Allow || asked {
		t.Fatalf("a read inside the workspace should be auto-allowed without asking, got %v asked=%v", b, asked)
	}

	for _, p := range []string{outside, "../outside.go", "/etc/passwd"} {
		if b, asked := decide(t, read, map[string]any{"file_path": p}, root); !asked {
			t.Errorf("read of %q was auto-allowed (%v) without consulting the host", p, b)
		}
	}
}

// Edits always ask; the escape case is called out separately in the message so
// a host can render it differently. Both mutating tools plus MultiEdit must be
// covered — a missing one is how the plan-mode flow stalled.
func TestEditToolsAskAndNameEscapes(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "x.go")

	for _, tt := range []struct {
		tool  CoreTool
		input map[string]any
	}{
		{NewWrite(), map[string]any{"file_path": outside, "content": "x"}},
		{NewEdit(), map[string]any{"file_path": outside, "old_string": "a", "new_string": "b"}},
		{NewMultiEdit(), map[string]any{"file_path": outside, "edits": []any{}}},
	} {
		t.Run(tt.tool.Name(), func(t *testing.T) {
			if _, asked := decide(t, tt.tool, tt.input, root); !asked {
				t.Fatalf("%s outside the workspace never reached the approval callback", tt.tool.Name())
			}
			raw, _ := json.Marshal(tt.input)
			pc := permission.Context{Mode: permission.ModeDefault, WorkingDir: root}
			d := tt.tool.CheckPermissions(context.Background(), raw, pc)
			if d.Behavior != permission.Ask {
				t.Fatalf("%s should ask, got %v", tt.tool.Name(), d.Behavior)
			}
			if want := "outside the workspace"; !containsStr(d.Message, want) {
				t.Errorf("%s message %q should mention %q", tt.tool.Name(), d.Message, want)
			}
		})
	}
}

func containsStr(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

// A shell the model can run is a shell that can run `env`. Credentials must not
// reach it unless the host puts them there on purpose.
func TestBashEnvDropsSecrets(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
	t.Setenv("GITHUB_TOKEN", "gh-token")
	t.Setenv("PGPASSWORD", "pw")
	t.Setenv("NORMA_PLAIN_VAR", "keep-me")

	out := withEnv(nil)
	joined := strings.Join(out, "\n")
	for _, leaked := range []string{"sk-ant-secret", "aws-secret", "gh-token", "pw"} {
		if strings.Contains(joined, leaked) {
			t.Errorf("child env still carries %q", leaked)
		}
	}
	if !strings.Contains(joined, "NORMA_PLAIN_VAR=keep-me") {
		t.Error("a non-secret variable was dropped")
	}

	// The host can still pass one deliberately.
	forced := withEnv([]string{"ANTHROPIC_API_KEY=host-chosen"})
	if !strings.Contains(strings.Join(forced, "\n"), "ANTHROPIC_API_KEY=host-chosen") {
		t.Error("a host-supplied variable must survive the filter")
	}
}

// truncate must respect its own limit. A cap of 30000 that returns 30060 has no
// bound at all, and every caller of Capture inherits that.
func TestTruncateRespectsItsLimit(t *testing.T) {
	long := strings.Repeat("a", 10_000)
	for _, max := range []int{10, 101, 500, 30_000} {
		out := truncate(long, max)
		if len(out) > max {
			t.Errorf("truncate(%d) returned %d bytes", max, len(out))
		}
	}
	// And the no-spill path through Capture is bounded the same way.
	out := Capture(&ToolContext{MaxOutputChars: 101}, long)
	if len(out) > 101 {
		t.Errorf("Capture returned %d bytes for a 101-byte cap", len(out))
	}
}
