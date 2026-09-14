package harness

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// A tool's InputSchema does two unrelated jobs: it tells the model what to send,
// and it decides what gets through. Most tools want both. A tool that repairs
// malformed arguments itself wants only the first — the gate would reject the
// very calls it exists to repair, because a truncated or fence-wrapped call is
// not valid JSON.
//
// These tests cover both sides of the opt-out.

// recordingSpec builds a tool that remembers the bytes it was handed.
func recordingSpec(name string, raw bool, got *json.RawMessage) tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:   name,
		Schema: map[string]any{"type": "object", "required": []any{"content"}},
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Decision{Behavior: permission.Allow}
		},
		RawInput: raw,
		Run: func(_ context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			*got = in
			return tool.Result{Content: []llm.ContentBlock{llm.TextBlock("ran")}}, nil
		},
	})
}

func execWith(t *testing.T, ct tool.CoreTool, input string) llm.ContentBlock {
	t.Helper()
	l := &loop{ctx: context.Background(), in: QueryInput{
		Tools:          tool.NewRegistry(ct),
		PermissionMode: permission.ModeBypass,
	}}
	res, _ := l.execOne(llm.ContentBlock{
		Type: llm.BlockToolUse, ID: "toolu_1", Name: ct.Name(), Input: json.RawMessage(input),
	}, nil)
	return res
}

// The default: invalid JSON is answered with a validation error and Run is never
// reached. This is the behaviour every other tool relies on.
func TestSchemaGateStillRejectsForOrdinaryTools(t *testing.T) {
	var got json.RawMessage
	res := execWith(t, recordingSpec("Ordinary", false, &got), `{"content":[{"a`)

	if !res.IsError {
		t.Fatal("malformed input produced a success result for a tool that did not opt out")
	}
	if !strings.Contains(res.Content[0].Text, "invalid tool input") {
		t.Fatalf("result = %q, want the schema validation error", res.Content[0].Text)
	}
	if got != nil {
		t.Fatalf("Run was reached with %q despite failing validation", got)
	}
}

// The opt-out: the same bytes reach Run untouched, so the tool can repair them.
func TestRawInputToolReceivesMalformedArgumentsVerbatim(t *testing.T) {
	cases := map[string]string{
		"truncated mid-object": `{"content":[{"startId":"m00001","end`,
		"fenced in markdown":   "```json\n{\"content\":[]}\n```",
		"trailing comma":       `{"content":[{"a":1,},],}`,
		"not an object":        `"just a string"`,
		"missing required":     `{"topic":"t"}`,
		"empty":                ``,
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			var got json.RawMessage
			res := execWith(t, recordingSpec("Lenient", true, &got), input)

			if res.IsError {
				t.Fatalf("the harness rejected input the tool opted in to handle: %q", res.Content[0].Text)
			}
			if string(got) != input {
				t.Fatalf("Run received %q, want the bytes the model sent, %q", got, input)
			}
		})
	}
}

// Opting out of validation must not opt out of anything else. Permission is the
// one that matters: a tool that skips the schema check still answers to the same
// permission decision.
func TestRawInputToolStillHonoursPermissions(t *testing.T) {
	var got json.RawMessage
	denied := tool.Build(tool.Spec{
		Name:     "Denied",
		Schema:   map[string]any{"type": "object"},
		RawInput: true,
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Decision{Behavior: permission.Deny, Message: "nope"}
		},
		Run: func(_ context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			got = in
			return tool.Result{}, nil
		},
	})
	res := execWith(t, denied, `{"garbage`)

	if !res.IsError {
		t.Fatal("a denied raw-input tool produced a success result")
	}
	if got != nil {
		t.Fatal("a denied raw-input tool was executed anyway")
	}
}

// A tool that does not implement the optional interface at all — the shape every
// host-supplied CoreTool has — must keep being validated.
type bareTool struct{ tool.CoreTool }

func TestToolsWithoutTheOptionalInterfaceAreValidated(t *testing.T) {
	var got json.RawMessage
	res := execWith(t, bareTool{recordingSpec("Bare", true, &got)}, `{"content":[{"a`)

	if !res.IsError {
		t.Fatal("a CoreTool that does not implement AcceptsRawInput skipped validation; " +
			"the assertion is matching something it should not")
	}
}
