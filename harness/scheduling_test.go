package harness

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// concurrencyProbe records the peak number of simultaneous executions.
type concurrencyProbe struct {
	mu      sync.Mutex
	active  int
	maxSeen int
}

func (p *concurrencyProbe) enter() {
	p.mu.Lock()
	p.active++
	if p.active > p.maxSeen {
		p.maxSeen = p.active
	}
	p.mu.Unlock()
}
func (p *concurrencyProbe) leave() {
	p.mu.Lock()
	p.active--
	p.mu.Unlock()
}

func probeTool(name string, safe bool, p *concurrencyProbe) tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:       name,
		Schema:     map[string]any{"type": "object"},
		ReadOnly:   func(json.RawMessage) bool { return safe },
		Concurrent: func(json.RawMessage) bool { return safe },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, _ json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			p.enter()
			time.Sleep(25 * time.Millisecond) // widen the overlap window
			p.leave()
			return tool.Text("ok"), nil
		},
	})
}

func twoToolTurn(name string) []llm.StreamEvent {
	return []llm.StreamEvent{
		{Type: llm.SEToolUseStart, ToolID: "t1", ToolName: name},
		{Type: llm.SEToolInputJSON, Text: `{}`},
		{Type: llm.SEToolUseStart, ToolID: "t2", ToolName: name},
		{Type: llm.SEToolInputJSON, Text: `{}`},
		{Type: llm.SEMessageDelta, StopReason: "tool_use"},
		{Type: llm.SEMessageStop},
	}
}

func TestSafeToolsRunConcurrently(t *testing.T) {
	p := &concurrencyProbe{}
	m := &scriptedModel{turns: [][]llm.StreamEvent{twoToolTurn("safe"), textTurn("done")}}
	in := QueryInput{
		Messages: []llm.Message{llm.UserText("go")}, PermissionMode: permission.ModeBypass,
		Tools: tool.NewRegistry(probeTool("safe", true, p)),
	}
	drain(t, in, QueryDeps{CallModel: m.call})
	if p.maxSeen != 2 {
		t.Fatalf("concurrency-safe tools should overlap (maxSeen=%d, want 2)", p.maxSeen)
	}
}

func TestNonSafeToolsRunExclusively(t *testing.T) {
	p := &concurrencyProbe{}
	m := &scriptedModel{turns: [][]llm.StreamEvent{twoToolTurn("excl"), textTurn("done")}}
	in := QueryInput{
		Messages: []llm.Message{llm.UserText("go")}, PermissionMode: permission.ModeBypass,
		Tools: tool.NewRegistry(probeTool("excl", false, p)),
	}
	drain(t, in, QueryDeps{CallModel: m.call})
	if p.maxSeen != 1 {
		t.Fatalf("non-concurrency-safe tools must run exclusively (maxSeen=%d, want 1)", p.maxSeen)
	}
}
