package harness

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/hook"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

func TestDeferredTargetExecutionPipeline(t *testing.T) {
	for _, test := range []struct {
		name                                            string
		mode                                            permission.Mode
		disallow, hardDeny, approve, blockHook, invalid bool
		wantRun, wantAsk, wantPre, wantPost             bool
	}{
		{name: "approval denied", wantAsk: true},
		{name: "approval granted", approve: true, wantRun: true, wantAsk: true, wantPre: true, wantPost: true},
		{name: "deny rule wins bypass", mode: permission.ModeBypass, disallow: true},
		{name: "target hard deny wins bypass", mode: permission.ModeBypass, hardDeny: true},
		{name: "dontAsk rejects target", mode: permission.ModeDontAsk},
		{name: "target hook blocks", approve: true, blockHook: true, wantAsk: true, wantPre: true},
		{name: "target schema rejects", invalid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ran, asked, pre, post := false, false, false, false
			target := tool.Build(tool.Spec{
				Name: "target", Schema: map[string]any{"type": "object", "required": []string{"msg"}, "properties": map[string]any{"msg": map[string]any{"type": "string"}}},
				Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
					if test.hardDeny {
						return permission.Denied("target denies")
					}
					return permission.AskUser("target approval")
				},
				Run: func(_ context.Context, raw json.RawMessage, tc *tool.ToolContext) (tool.Result, error) {
					ran = true
					if string(raw) != `{"msg":"rewritten"}` || tc.ToolUseID != "call1" {
						t.Errorf("target lost rewritten input or call id: %s, %s", raw, tc.ToolUseID)
					}
					return tool.Text("target output"), nil
				},
			})
			reg := tool.NewRegistry(target)
			reg.Add(tool.NewExecuteExtraTool(reg, tool.NewUnlockSet("target")))
			hooks := hook.NewRegistry().On(hook.PreToolUse, func(_ context.Context, e hook.Event) hook.Result {
				if e.ToolName != "target" {
					return hook.Result{}
				}
				pre = true
				if test.blockHook {
					return hook.Result{Decision: "block", Message: "target hook blocks"}
				}
				return hook.Result{UpdatedInput: json.RawMessage(`{"msg":"rewritten"}`)}
			}).On(hook.PostToolUse, func(_ context.Context, e hook.Event) hook.Result {
				if e.ToolName == "target" {
					post = true
					if e.IsError || !strings.Contains(string(e.Result), "target output") {
						t.Errorf("bad target hook result: %+v", e)
					}
				}
				return hook.Result{}
			})
			raw := `{"tool_name":"target","params":{"msg":"hi"}}`
			if test.invalid {
				raw = `{"tool_name":"target","params":{}}`
			}
			model := &scriptedModel{turns: [][]llm.StreamEvent{toolTurn("call1", tool.ExecuteExtraToolName, raw), textTurn("done")}}
			in := QueryInput{Tools: reg, PermissionMode: test.mode, Hooks: hooks, WorkingDir: t.TempDir(), Messages: []llm.Message{llm.UserText("go")}}
			if test.disallow {
				in.Disallowed = []string{"target"}
			}
			in.CanUseTool = func(_ context.Context, name string, raw json.RawMessage, _ permission.Context) permission.Decision {
				asked = true
				if name != "target" || string(raw) != `{"msg":"hi"}` {
					t.Errorf("approval must name target: %s %s", name, raw)
				}
				if test.approve {
					return permission.Allowed()
				}
				return permission.Denied("host denied")
			}
			term := drain(t, in, QueryDeps{CallModel: model.call})
			res := term.Messages[2].Content[0]
			if res.ToolUseID != "call1" || res.IsError == test.wantRun {
				t.Errorf("wrong wrapper result: %+v", res)
			}
			if ran != test.wantRun || asked != test.wantAsk || pre != test.wantPre || post != test.wantPost {
				t.Fatalf("ran/asked/pre/post = %v/%v/%v/%v", ran, asked, pre, post)
			}
		})
	}
}

func TestDeferredRawInputAndExtraMessages(t *testing.T) {
	// A target that repairs its own input must retain that contract when deferred.
	target := tool.Build(tool.Spec{
		Name: "repair", RawInput: true,
		Schema: map[string]any{"type": "object", "required": []string{"missing"}},
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, input json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			if string(input) != "{}" {
				t.Errorf("input changed: %s", input)
			}
			return tool.Result{Content: []llm.ContentBlock{llm.TextBlock("repaired")}, Extra: []llm.Message{llm.UserText("standing guidance")}}, nil
		},
	})
	reg := tool.NewRegistry(target)
	reg.Add(tool.NewExecuteExtraTool(reg, tool.NewUnlockSet("repair")))
	model := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("call1", tool.ExecuteExtraToolName, `{"tool_name":"repair"}`), textTurn("done"),
	}}
	term := drain(t, QueryInput{Tools: reg, Messages: []llm.Message{llm.UserText("go")}}, QueryDeps{CallModel: model.call})
	if res := term.Messages[2].Content[0]; res.IsError || res.Content[0].Text != "repaired" {
		t.Fatalf("raw target failed: %+v", res)
	}
	if len(term.Messages) != 5 || term.Messages[3].Content[0].Text != "standing guidance" {
		t.Fatalf("extra message lost: %+v", term.Messages)
	}
}
