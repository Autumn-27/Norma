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

func TestAllowRuleMatchesParenForm(t *testing.T) {
	if b := eval(Request{ToolName: "Bash", Ctx: Context{Mode: ModeDefault, Allowed: []string{"Bash(git:*)"}}}); b != Allow {
		t.Fatalf("paren allow rule should match tool name, got %v", b)
	}
}
