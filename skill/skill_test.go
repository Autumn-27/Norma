package skill

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/tool"
)

func TestSkillToolReturnsInstructions(t *testing.T) {
	r := NewRegistry(Skill{
		Name:         "release",
		Description:  "cut a release",
		WhenToUse:    "when the user wants to ship a version",
		Instructions: "1. bump version\n2. tag\n3. push",
	})
	st := r.Tool()
	// Description is the primary discovery field (agentskills.io); WhenToUse is a
	// fallback only when Description is empty.
	if !strings.Contains(st.Prompt(), "release: cut a release") {
		t.Fatalf("tool prompt missing skill listing: %q", st.Prompt())
	}
	res, _ := st.Call(context.Background(), []byte(`{"name":"release","args":"v2.0"}`), &tool.ToolContext{})
	out := res.Flatten()
	if !strings.Contains(out, "1. bump version") || !strings.Contains(out, "v2.0") {
		t.Fatalf("skill result: %q", out)
	}
	res, _ = st.Call(context.Background(), []byte(`{"name":"nope"}`), &tool.ToolContext{})
	if !res.IsError {
		t.Fatal("unknown skill should error")
	}
}

func TestLoadDir(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "audit.md"), []byte("---\nname: audit\ndescription: security audit\nwhenToUse: when reviewing for vulns\n---\n\nReview the code for OWASP top 10."), 0o644)
	// A subdirectory skill.
	sub := filepath.Join(dir, "deploy")
	os.MkdirAll(sub, 0o755)
	os.WriteFile(filepath.Join(sub, "SKILL.md"), []byte("---\nname: deploy\n---\nRun make deploy."), 0o644)

	r, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	a, ok := r.Get("audit")
	if !ok || !strings.Contains(a.Instructions, "OWASP") || a.WhenToUse == "" {
		t.Fatalf("audit skill: %+v ok=%v", a, ok)
	}
	d, ok := r.Get("deploy")
	if !ok || !strings.Contains(d.Instructions, "make deploy") {
		t.Fatalf("deploy skill: %+v ok=%v", d, ok)
	}
}
