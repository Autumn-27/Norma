package tool

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Autumn-27/norma/permission"
)

// These tools can run arbitrary commands, so they must be gated exactly like
// Bash. They used to return Allowed() from their own check, which short-circuits
// the pipeline before the host's approval callback is consulted: `shell_open`
// with `rm -rf /` was allowed in every mode that does not deny non-read-only
// tools outright.
func TestShellToolsApplyTheBashFloor(t *testing.T) {
	tools := map[string]CoreTool{}
	for _, tl := range ShellSessionTools() {
		tools[tl.Name()] = tl
	}
	cases := []struct {
		tool   string
		input  string
		denied bool
	}{
		{"shell_open", `{"command":"rm -rf /"}`, true},
		{"shell_open", `{"command":"python3"}`, false},
		{"shell_send", `{"session_id":"s1","text":"rm -rf /","submit":true}`, true},
		{"shell_send", `{"session_id":"s1","text":"print(1)","submit":true}`, false},
		{"shell_read", `{"session_id":"s1"}`, false},
	}
	for _, c := range cases {
		tl, ok := tools[c.tool]
		if !ok {
			t.Fatalf("%s missing from ShellSessionTools()", c.tool)
		}
		in := json.RawMessage(c.input)
		pc := permission.Context{Mode: permission.ModeDefault, WorkingDir: t.TempDir()}

		asked := false
		ask := func(context.Context, string, json.RawMessage, permission.Context) permission.Decision {
			asked = true
			return permission.Denied("host said no")
		}
		got := permission.Evaluate(context.Background(), permission.Request{
			ToolName:     tl.Name(),
			Input:        in,
			IsReadOnly:   tl.IsReadOnly(in),
			ToolDecision: tl.CheckPermissions(context.Background(), in, pc),
			Ctx:          pc,
			Ask:          ask,
		})
		switch {
		case c.denied && got.Behavior != permission.Deny:
			t.Fatalf("%s %s: verdict=%s, want deny (the safety floor must apply)", c.tool, c.input, got.Behavior)
		case c.denied:
			// denied before the host is bothered
			if asked {
				t.Fatalf("%s %s: a hard deny must not reach the approval callback", c.tool, c.input)
			}
		}
	}

	// A plain command must reach the host's callback rather than being allowed
	// by the tool's own verdict.
	in := json.RawMessage(`{"command":"python3"}`)
	pc := permission.Context{Mode: permission.ModeDefault, WorkingDir: t.TempDir()}
	asked := false
	permission.Evaluate(context.Background(), permission.Request{
		ToolName:     "shell_open",
		Input:        in,
		ToolDecision: tools["shell_open"].CheckPermissions(context.Background(), in, pc),
		Ctx:          pc,
		Ask: func(context.Context, string, json.RawMessage, permission.Context) permission.Decision {
			asked = true
			return permission.Allowed()
		},
	})
	if !asked {
		t.Fatal("shell_open on a plain command was allowed without asking the host")
	}
}
