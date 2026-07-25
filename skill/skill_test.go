package skill

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
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
	// A1: instructions travel as an independent user message (Result.Extra), not
	// in the tool_result — the tool_result is a short acknowledgement.
	if !strings.Contains(res.Flatten(), "loaded") {
		t.Fatalf("expected ack in tool_result, got: %q", res.Flatten())
	}
	if len(res.Extra) != 1 {
		t.Fatalf("expected 1 extra message, got %d", len(res.Extra))
	}
	body := res.Extra[0].Text()
	if !strings.Contains(body, "1. bump version") || !strings.Contains(body, "v2.0") {
		t.Fatalf("skill instructions message: %q", body)
	}
	if name, ok := llm.SkillInvocationName(res.Extra[0]); !ok || name != "release" {
		t.Fatalf("extra message not a skill invocation for release: name=%q ok=%v", name, ok)
	}
	res, _ = st.Call(context.Background(), []byte(`{"name":"nope"}`), &tool.ToolContext{})
	if !res.IsError {
		t.Fatal("unknown skill should error")
	}
}

func TestParseRealYAML(t *testing.T) {
	// Things the old hand-rolled parser could not handle: a quoted colon in the
	// value, and a multi-line block scalar.
	src := "---\n" +
		"name: deploy\n" +
		"description: \"ship it: fast and safe\"\n" +
		"whenToUse: |\n" +
		"  line one\n" +
		"  line two\n" +
		"mcps:\n" +
		"  - browser\n" +
		"  - nmap\n" +
		"---\n" +
		"Body here."
	sk := parse(src)
	if sk.Description != "ship it: fast and safe" {
		t.Fatalf("quoted-colon description mishandled: %q", sk.Description)
	}
	if !strings.Contains(sk.WhenToUse, "line one") || !strings.Contains(sk.WhenToUse, "line two") {
		t.Fatalf("multi-line block scalar mishandled: %q", sk.WhenToUse)
	}
	if strings.Join(sk.MCPs, ",") != "browser,nmap" {
		t.Fatalf("block-list mcps mishandled: %v", sk.MCPs)
	}
	if sk.Instructions != "Body here." {
		t.Fatalf("body mishandled: %q", sk.Instructions)
	}
}

func TestParseMalformedYAMLDegrades(t *testing.T) {
	// Broken YAML in the header must not drop the skill: fields empty, body loads.
	sk := parse("---\n: : : not valid\n\t- bad indent\n---\nStill loads.")
	if sk.Instructions != "Still loads." {
		t.Fatalf("body should still load on bad frontmatter, got %q", sk.Instructions)
	}
}

func TestListingBudgetDegradesToNamesOnly(t *testing.T) {
	r := NewRegistry()
	longDesc := strings.Repeat("d", 400)
	for i := 0; i < 100; i++ {
		r.Add(Skill{Name: "skill" + string(rune('A'+i%26)) + string(rune('0'+i/26)), Description: longDesc})
	}
	list := r.Tool().Prompt()
	if len(list) > maxSkillListingChars+2000 {
		t.Fatalf("listing not bounded: %d chars", len(list))
	}
	// Every skill must still be listed by name even after the budget is hit.
	for _, s := range r.List() {
		if !strings.Contains(list, "- "+s.Name) {
			t.Fatalf("skill %q dropped from listing", s.Name)
		}
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
