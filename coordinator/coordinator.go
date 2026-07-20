// Package coordinator implements coordinator mode: a top-level agent that
// decomposes work and delegates each unit to isolated "worker" subagents
// instead of doing the work itself. It offers two paths:
// an LLM-driven one (expose AgentTool + SystemSegment so the model spawns
// workers via the Agent tool) and a programmatic one (RunParallel fans tasks
// out to workers concurrently and returns their results).
package coordinator

import (
	"context"
	"strings"
	"sync"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/subagent"
	"github.com/Autumn-27/norma/tool"
)

// DefaultWorkerPrompt is the default system prompt for worker subagents.
const DefaultWorkerPrompt = `You are a worker agent spawned by a coordinator. Complete the task described in the prompt thoroughly and report back a concise summary of what you did and found.

Guidelines:
- Complete the task fully — don't leave it half-done, but don't gold-plate either.
- Use tools proactively: read files, search code, run commands, edit files.
- Be thorough in research: check multiple locations and naming conventions.
- For implementation: make targeted changes and run tests to verify.
- Report back with actionable findings — the coordinator will synthesize your results.
- If you hit errors, investigate and try to fix them before reporting failure.`

// Config configures coordinator mode.
type Config struct {
	Provider       llm.Provider
	WorkerTools    *tool.Registry  // tools available to workers (e.g. the core toolset)
	WorkerPrompt   []string        // worker system prompt (default DefaultWorkerPrompt)
	PermissionMode permission.Mode // workers' permission mode (default acceptEdits)
	WorkingDir     string
	MaxTurns       int // per-worker turn cap
	MaxDepth       int // subagent recursion cap (default 3)
	MaxParallel    int // bounded concurrency for RunParallel (default 4)
	CanUseTool     permission.CanUseTool
	Hooks          harness.HookRunner
}

func (c *Config) workerPrompt() []string {
	if len(c.WorkerPrompt) > 0 {
		return c.WorkerPrompt
	}
	return []string{DefaultWorkerPrompt}
}

func (c *Config) mode() permission.Mode {
	if c.PermissionMode != "" {
		return c.PermissionMode
	}
	return permission.ModeAcceptEdits
}

// subagentConfig wires a subagent.Config with a single "worker" definition.
func (c *Config) subagentConfig() subagent.Config {
	return subagent.Config{
		Provider:    c.Provider,
		ParentTools: c.WorkerTools,
		MaxDepth:    c.MaxDepth,
		CanUseTool:  c.CanUseTool,
		Hooks:       c.Hooks,
		Agents: map[string]subagent.Definition{
			"worker": {
				Description:    "Executes a research, implementation, or verification task autonomously and reports back.",
				Prompt:         c.workerPrompt(),
				PermissionMode: c.mode(),
				MaxTurns:       c.MaxTurns,
			},
		},
	}
}

// AgentTool returns the Agent tool the coordinator LLM uses to spawn workers.
// Add it (and nothing else mutating) to the coordinator session's tools.
func (c *Config) AgentTool() tool.CoreTool {
	return subagent.NewTool(c.subagentConfig())
}

// SystemSegment returns coordinator guidance to append to the system prompt.
func (c *Config) SystemSegment() string {
	var names []string
	if c.WorkerTools != nil {
		for _, t := range c.WorkerTools.List() {
			names = append(names, t.Name())
		}
	}
	return "# Coordinator mode\n" +
		"You orchestrate work by delegating to worker subagents — you do NOT implement directly. " +
		"Break the request into independent units and spawn a worker for each via the Agent tool " +
		"(agent_type \"worker\"), then synthesize their reports into a final answer. " +
		"Prefer parallel, independent units. Workers have these tools: " + strings.Join(names, ", ") + "."
}

// Result is the outcome of one programmatic worker run.
type Result struct {
	Task   string
	Output string
	Err    error
}

// RunParallel fans tasks out to workers concurrently (bounded by MaxParallel)
// and returns results in task order.
func (c *Config) RunParallel(ctx context.Context, tasks []string) []Result {
	cfg := c.subagentConfig()
	results := make([]Result, len(tasks))
	limit := c.MaxParallel
	if limit <= 0 {
		limit = 4
	}
	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
	for i, task := range tasks {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, task string) {
			defer wg.Done()
			defer func() { <-sem }()
			out, err := subagent.Run(ctx, cfg, "worker", task, c.WorkingDir)
			results[i] = Result{Task: task, Output: out, Err: err}
		}(i, task)
	}
	wg.Wait()
	return results
}
