package agentcore

import (
	"context"
	"encoding/json"
	"iter"
	"testing"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

func deferredEcho() tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:        "echo",
		Description: "Echoes msg.",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"msg": map[string]any{"type": "string"}},
			"required":   []any{"msg"},
		},
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var a struct {
				Msg string `json:"msg"`
			}
			_ = json.Unmarshal(in, &a)
			return tool.Text("echo:" + a.Msg), nil
		},
	})
}

// TestDeferredToolsWithheldButCallable proves the deferred mechanism end-to-end:
// the deferred tool's schema is NOT sent to the model, the two access tools ARE,
// and the model can still invoke the deferred tool via ExecuteExtraTool.
func TestDeferredToolsWithheldButCallable(t *testing.T) {
	turns := [][]llm.StreamEvent{
		toolTurn("e1", tool.ExecuteExtraToolName, `{"tool_name":"echo","params":{"msg":"hi"}}`),
		textTurn("done"),
	}
	var sentSchemas []llm.ToolSchema
	var n int
	callModel := func(_ context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		if n == 0 {
			sentSchemas = req.Tools // capture what the model was offered on turn 1
		}
		evs := turns[n]
		n++
		return func(yield func(llm.StreamEvent, error) bool) {
			for _, e := range evs {
				if !yield(e, nil) {
					return
				}
			}
		}
	}

	sess := NewSession(Options{
		SystemPrompt:   []string{"sys"},
		Tools:          []tool.CoreTool{deferredEcho()},
		DeferredTools:  []string{"echo"},
		PermissionMode: permission.ModeBypass,
		Deps:           harness.QueryDeps{CallModel: callModel},
	})

	var results []llm.ContentBlock
	for ev, err := range sess.Prompt(context.Background(), "echo hi") {
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if ev.Kind == harness.KindToolResult {
			results = append(results, *ev.ToolResult)
		}
	}

	// Schema list: echo withheld; the two access tools present.
	has := func(name string) bool {
		for _, s := range sentSchemas {
			if s.Name == name {
				return true
			}
		}
		return false
	}
	if has("echo") {
		t.Fatal("deferred tool 'echo' schema should be withheld from the model")
	}
	if !has(tool.SearchExtraToolsName) || !has(tool.ExecuteExtraToolName) {
		t.Fatalf("access tools missing from schema list: %+v", sentSchemas)
	}

	// The deferred tool still ran via ExecuteExtraTool.
	if len(results) == 0 || results[0].IsError || results[0].Content[0].Text != "echo:hi" {
		t.Fatalf("ExecuteExtraTool should have run echo, got %+v", results)
	}
}

// TestDeferredLockedToolRejected proves unlock-set gating: a deferred tool not in
// the unlock set is refused by ExecuteExtraTool.
func TestDeferredLockedToolRejected(t *testing.T) {
	turns := [][]llm.StreamEvent{
		toolTurn("e1", tool.ExecuteExtraToolName, `{"tool_name":"echo","params":{"msg":"hi"}}`),
		textTurn("done"),
	}
	var n int
	callModel := func(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		evs := turns[n]
		n++
		return func(yield func(llm.StreamEvent, error) bool) {
			for _, e := range evs {
				if !yield(e, nil) {
					return
				}
			}
		}
	}

	sess := NewSession(Options{
		SystemPrompt:   []string{"sys"},
		Tools:          []tool.CoreTool{deferredEcho()},
		DeferredTools:  []string{"echo"},
		UnlockSet:      tool.NewUnlockSet(), // echo deferred but LOCKED
		PermissionMode: permission.ModeBypass,
		Deps:           harness.QueryDeps{CallModel: callModel},
	})

	var results []llm.ContentBlock
	for ev, err := range sess.Prompt(context.Background(), "echo hi") {
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if ev.Kind == harness.KindToolResult {
			results = append(results, *ev.ToolResult)
		}
	}
	if len(results) == 0 || !results[0].IsError {
		t.Fatalf("locked deferred tool should be rejected, got %+v", results)
	}
}
