package subagent

import (
	"context"
	"iter"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// fakeProvider always answers with a fixed line of text and ends the turn.
type fakeProvider struct{ reply string }

func (f fakeProvider) Stream(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		yield(llm.StreamEvent{Type: llm.SETextDelta, Text: f.reply}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "end_turn"}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func (f fakeProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	acc := llm.NewAccumulator()
	for ev, err := range f.Stream(ctx, req) {
		if err != nil {
			return llm.Message{}, "", llm.Usage{}, err
		}
		acc.Add(ev)
	}
	return acc.Message(), acc.StopReason, acc.Usage, nil
}

func TestSubagentDispatch(t *testing.T) {
	agentTool := NewTool(Config{
		Provider:    fakeProvider{reply: "subagent answer"},
		ParentTools: tool.NewRegistry(tool.DefaultTools()...),
		Agents: map[string]Definition{
			"researcher": {Description: "researches", Prompt: []string{"You research."}, PermissionMode: permission.ModeBypass},
		},
	})
	res, err := agentTool.Call(context.Background(), []byte(`{"agent_type":"researcher","prompt":"find x"}`), &tool.ToolContext{})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(res.Flatten(), "subagent answer") {
		t.Fatalf("subagent result: %q", res.Flatten())
	}
}

func TestUnknownAgentType(t *testing.T) {
	agentTool := NewTool(Config{Provider: fakeProvider{}, Agents: map[string]Definition{}})
	res, _ := agentTool.Call(context.Background(), []byte(`{"agent_type":"nope","prompt":"x"}`), &tool.ToolContext{})
	if !res.IsError {
		t.Fatal("expected error for unknown agent_type")
	}
}

func TestRecursionDepthGuard(t *testing.T) {
	agentTool := NewTool(Config{
		Provider: fakeProvider{reply: "x"},
		MaxDepth: 2,
		Agents:   map[string]Definition{"a": {Prompt: []string{"p"}, PermissionMode: permission.ModeBypass}},
	})
	// Simulate being already at max depth.
	ctx := context.WithValue(context.Background(), depthKey, 2)
	res, _ := agentTool.Call(ctx, []byte(`{"agent_type":"a","prompt":"x"}`), &tool.ToolContext{})
	if !res.IsError || !strings.Contains(res.Flatten(), "recursion depth") {
		t.Fatalf("expected depth guard, got %q", res.Flatten())
	}
}
