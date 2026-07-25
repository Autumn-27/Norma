package skill

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestParseMCPsFrontmatter(t *testing.T) {
	cases := map[string][]string{
		"---\nname: a\nmcps: browser\n---\nbody":            {"browser"},
		"---\nname: b\nmcps: browser, nmap\n---\nbody":       {"browser", "nmap"},
		"---\nname: c\nmcps: [browser, \"nmap\"]\n---\nbody": {"browser", "nmap"},
	}
	for src, want := range cases {
		sk := parse(src)
		if strings.Join(sk.MCPs, ",") != strings.Join(want, ",") {
			t.Fatalf("MCPs=%v want %v (src=%q)", sk.MCPs, want, src)
		}
	}
	// no mcps → nil
	if sk := parse("---\nname: d\n---\nbody"); sk.MCPs != nil {
		t.Fatalf("expected nil MCPs, got %v", sk.MCPs)
	}
}

func TestOnInvokeAppendsAndFires(t *testing.T) {
	reg := NewRegistry(Skill{Name: "browsing", Description: "d", Instructions: "do it", MCPs: []string{"browser"}})
	var firedFor string
	reg.OnInvoke = func(s Skill) string {
		firedFor = s.Name
		return "UNLOCKED:" + strings.Join(s.MCPs, ",")
	}
	tl := reg.Tool()
	r, err := tl.Call(context.Background(), json.RawMessage(`{"name":"browsing"}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	txt := r.Content[0].Text
	if firedFor != "browsing" {
		t.Fatalf("OnInvoke not fired for browsing, got %q", firedFor)
	}
	if !strings.Contains(txt, "UNLOCKED:browser") {
		t.Fatalf("OnInvoke output not appended:\n%s", txt)
	}
}
