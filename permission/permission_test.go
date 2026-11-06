package permission

import (
	"context"
	"encoding/json"
	"testing"
)

func eval(req Request) Behavior {
	return Evaluate(context.Background(), req).Behavior
}

func TestDenyPriorityOverBypass(t *testing.T) {
	req := Request{ToolName: "Bash", Ctx: Context{Mode: ModeBypass, Disallowed: []string{"Bash"}}}
	if b := eval(req); b != Deny {
		t.Fatalf("deny must beat bypass, got %v", b)
	}
}

func TestReadOnlyAutoAllow(t *testing.T) {
	if b := eval(Request{ToolName: "Read", IsReadOnly: true, Ctx: Context{Mode: ModeDefault}}); b != Allow {
		t.Fatalf("read-only should allow, got %v", b)
	}
}

func TestPlanModeDeniesWrites(t *testing.T) {
	if b := eval(Request{ToolName: "Write", IsReadOnly: false, Ctx: Context{Mode: ModePlan}}); b != Deny {
		t.Fatalf("plan mode should deny writes, got %v", b)
	}
}

func TestAcceptEditsAllowsEdit(t *testing.T) {
	if b := eval(Request{ToolName: "Edit", Ctx: Context{Mode: ModeAcceptEdits}}); b != Allow {
		t.Fatalf("acceptEdits should allow Edit, got %v", b)
	}
}

func TestAskReachesCallback(t *testing.T) {
	called := false
	req := Request{
		ToolName: "Bash", IsReadOnly: false,
		ToolDecision: AskUser("?"),
		Ctx:          Context{Mode: ModeDefault},
		Ask: func(_ context.Context, _ string, _ json.RawMessage, _ Context) Decision {
			called = true
			return Allowed()
		},
	}
	if b := eval(req); b != Allow || !called {
		t.Fatalf("ask callback not honored: b=%v called=%v", b, called)
	}
}

func TestAcceptEditsAllowsEveryFileTool(t *testing.T) {
	// All three mutating file tools must be auto-allowed, or plan mode's
	// approve-then-implement flips to AcceptEdits and then stalls on one of them.
	for _, name := range []string{"Write", "Edit", "MultiEdit"} {
		if b := eval(Request{ToolName: name, Ctx: Context{Mode: ModeAcceptEdits}}); b != Allow {
			t.Fatalf("acceptEdits should allow %s, got %v", name, b)
		}
	}
}

func TestAllowRuleMatchesParenForm(t *testing.T) {
	// "Bash(git:*)" is a command prefix, not a wildcard for the whole tool.
	allow := []string{"Bash(git:*)"}
	ctx := Context{Mode: ModeDefault, Allowed: allow}

	match := func(cmd string) Behavior {
		input, _ := json.Marshal(map[string]string{"command": cmd})
		return eval(Request{ToolName: "Bash", Input: input, Ctx: ctx})
	}
	if b := match("git status"); b != Allow {
		t.Fatalf("git status should match Bash(git:*), got %v", b)
	}
	if b := match("git"); b != Allow {
		t.Fatalf("bare git should match Bash(git:*), got %v", b)
	}
	// The whole point: a rule naming git must not grant the shell.
	// No callback configured, so "not allowed by rule" ends in Deny.
	if b := match("rm -rf /tmp/work"); b != Deny {
		t.Fatalf("rm must not match Bash(git:*), got %v", b)
	}
	// A word boundary, not a string prefix.
	if b := match("gitleaks detect"); b != Deny {
		t.Fatalf("gitleaks must not match Bash(git:*), got %v", b)
	}
}

// A rule form this code cannot evaluate must grant nothing.
func TestUnknownParenRuleGrantsNothing(t *testing.T) {
	ctx := Context{Mode: ModeDefault, Allowed: []string{"Read(/etc/*)"}}
	input, _ := json.Marshal(map[string]string{"file_path": "/etc/passwd"})
	if b := eval(Request{ToolName: "Read", Input: input, Ctx: ctx}); b == Allow {
		t.Fatal("a Read(...) rule must not be honoured: this code cannot evaluate it")
	}
}

// Deny rules use the same matcher, so a scoped deny denies only its scope.
func TestScopedDenyOnlyDeniesItsScope(t *testing.T) {
	ctx := Context{Mode: ModeBypass, Disallowed: []string{"Bash(rm:*)"}}
	with := func(cmd string) Behavior {
		input, _ := json.Marshal(map[string]string{"command": cmd})
		return eval(Request{ToolName: "Bash", Input: input, Ctx: ctx})
	}
	if b := with("rm -rf /tmp/work"); b != Deny {
		t.Fatalf("rm should be denied by Bash(rm:*), got %v", b)
	}
	if b := with("go test ./..."); b != Allow {
		t.Fatalf("go test must not be denied by Bash(rm:*), got %v", b)
	}
}

// A read-only tool may still escalate to a human; the pipeline must not
// auto-allow past its own check.
func TestReadOnlyToolCanEscalate(t *testing.T) {
	called := false
	req := Request{
		ToolName: "Read", IsReadOnly: true,
		ToolDecision: AskUser("outside the workspace"),
		Ctx:          Context{Mode: ModeDefault},
		Ask: func(context.Context, string, json.RawMessage, Context) Decision {
			called = true
			return Denied("no")
		},
	}
	if b := eval(req); b != Deny || !called {
		t.Fatalf("a read-only tool's Ask must reach the callback: b=%v called=%v", b, called)
	}
}
