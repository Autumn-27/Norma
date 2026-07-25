// Package subagent lets the agent delegate to isolated child agents (FR-09).
// A subagent reuses the same harness loop with its own prompt, tool subset, and
// permission mode, giving context isolation. It is exposed to the parent as an
// "Agent" CoreTool; recursion is bounded by a depth guard carried on the
// context.
package subagent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
	"github.com/Autumn-27/norma/transcript"
)

// Definition configures a named subagent (FR-09.1).
type Definition struct {
	Description    string          // shown to the parent model in the Agent tool prompt
	Prompt         []string        // the subagent's system prompt
	Tools          []string        // tool names it may use (empty = inherit all)
	PermissionMode permission.Mode // its permission mode (default: parent's)
	MaxTurns       int
}

// Config wires the Agent tool.
type Config struct {
	Provider    llm.Provider
	ParentTools *tool.Registry
	Agents      map[string]Definition
	MaxDepth    int // default 3
	CanUseTool  permission.CanUseTool
	Hooks       harness.HookRunner
	// Transcript, when non-nil, persists each subagent run to its own sidechain
	// file under the parent session (whose id is read from the context). Subagent
	// tokens are recorded there, separate from the parent's totals.
	Transcript *transcript.Store
}

type ctxKey int

const depthKey ctxKey = 0

func depth(ctx context.Context) int {
	v, _ := ctx.Value(depthKey).(int)
	return v
}

// NewTool builds the "Agent" CoreTool that dispatches to a named subagent.
func NewTool(cfg Config) tool.CoreTool {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = 3
	}
	var names string
	for n, d := range cfg.Agents {
		names += fmt.Sprintf("\n- %s: %s", n, d.Description)
	}
	return tool.Build(tool.Spec{
		Name:        "Agent",
		Description: "Delegates a self-contained sub-task to an isolated subagent that runs with its own prompt and tools, returning only its final answer. Use it to parallelize independent work or to keep large intermediate output out of the main context.",
		Prompt:      "Available agent_type values:" + names,
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"agent_type": map[string]any{"type": "string", "description": "Which subagent to run."},
				"prompt":     map[string]any{"type": "string", "description": "The task for the subagent."},
			},
			"required": []any{"agent_type", "prompt"},
		},
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed() // the subagent's own tool calls are gated internally
		},
		Run: func(ctx context.Context, input json.RawMessage, tc *tool.ToolContext) (tool.Result, error) {
			var in struct {
				AgentType string `json:"agent_type"`
				Prompt    string `json:"prompt"`
			}
			if err := json.Unmarshal(input, &in); err != nil {
				return tool.Result{}, err
			}
			if depth(ctx) >= cfg.MaxDepth {
				return tool.Errorf(fmt.Sprintf("Error: subagent recursion depth %d exceeded", cfg.MaxDepth)), nil
			}
			def, ok := cfg.Agents[in.AgentType]
			if !ok {
				return tool.Errorf(fmt.Sprintf("Error: unknown agent_type %q", in.AgentType)), nil
			}
			text, err := run(ctx, cfg, def, in.AgentType, in.Prompt, tc.WorkingDir)
			if err != nil {
				return tool.Errorf("Error: subagent failed: " + err.Error()), nil
			}
			return tool.Text(text), nil
		},
	})
}

// Run executes the named agent from cfg.Agents to completion and returns its
// final text. It is the programmatic entry point (the Agent tool uses the same
// machinery); callers like the coordinator use it to fan out work directly.
func Run(ctx context.Context, cfg Config, agentType, prompt, workDir string) (string, error) {
	def, ok := cfg.Agents[agentType]
	if !ok {
		return "", fmt.Errorf("subagent: unknown agent_type %q", agentType)
	}
	return run(ctx, cfg, def, agentType, prompt, workDir)
}

// run executes a subagent to completion and returns its final text.
func run(ctx context.Context, cfg Config, def Definition, agentType, prompt, workDir string) (string, error) {
	if cfg.Hooks != nil {
		if hs, ok := cfg.Hooks.(interface {
			FireSubagentStart(context.Context, string)
		}); ok {
			hs.FireSubagentStart(ctx, agentType)
		}
	}
	registry := filterTools(cfg.ParentTools, def.Tools)
	childCtx := context.WithValue(ctx, depthKey, depth(ctx)+1)

	seed := llm.UserText(prompt)
	in := harness.QueryInput{
		Provider:       cfg.Provider,
		System:         def.Prompt,
		Messages:       []llm.Message{seed},
		Tools:          registry,
		MaxTurns:       def.MaxTurns,
		WorkingDir:     workDir,
		AgentID:        agentType,
		PermissionMode: def.PermissionMode,
		CanUseTool:     cfg.CanUseTool,
		Hooks:          cfg.Hooks,
	}
	// Persist this run to its own sidechain under the parent session.
	if cfg.Transcript != nil {
		parentSID := transcript.SessionIDFrom(ctx)
		if parentSID == "" {
			parentSID = transcript.NewSessionID()
		}
		w := cfg.Transcript.NewWriter(parentSID, transcript.NewAgentID())
		w.RecordMessage(seed, llm.Usage{})
		in.Recorder = w
	}
	var final string
	var rerr error
	for ev, err := range harness.Query(childCtx, in, harness.QueryDeps{}) {
		if err != nil {
			return "", err
		}
		if ev.Kind == harness.KindResult && ev.Terminal != nil {
			final = ev.Terminal.Text
			rerr = ev.Terminal.Err
		}
	}
	return final, rerr
}

// filterTools returns a registry with only the named tools (all if names empty).
func filterTools(parent *tool.Registry, names []string) *tool.Registry {
	if parent == nil {
		return tool.NewRegistry()
	}
	if len(names) == 0 {
		return parent
	}
	keep := map[string]bool{}
	for _, n := range names {
		keep[n] = true
	}
	var sel []tool.CoreTool
	for _, t := range parent.List() {
		if keep[t.Name()] {
			sel = append(sel, t)
		}
	}
	return tool.NewRegistry(sel...)
}
