// Example security-audit retargets the same core to a read-only security
// reviewer that delegates focused checks to a subagent. It demonstrates: a
// non-coding system prompt, plan (read-only) permission mode, and the Agent
// (subagent) tool wired alongside the core read tools.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/Autumn-27/norma/agentcore"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/subagent"
	"github.com/Autumn-27/norma/tool"
)

func main() {
	model := envOr("AGENT_MODEL", "gpt-4o")
	prov, err := llm.NewProvider(llm.Config{Format: llm.FormatOpenAI, Model: model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	wd, _ := os.Getwd()

	// Read-only core tools for the auditor.
	readTools := []tool.CoreTool{tool.NewRead(), tool.NewGlob(), tool.NewGrep()}
	parent := tool.NewRegistry(readTools...)

	// A subagent specialized in tracing a single finding to its source.
	agentTool := subagent.NewTool(subagent.Config{
		Provider:    prov,
		ParentTools: parent,
		Agents: map[string]subagent.Definition{
			"tracer": {
				Description:    "traces a vulnerability candidate to its exact source location",
				Prompt:         []string{"You trace a single security finding to the precise file and line. Use Read/Grep/Glob only. Report file:line and a one-line justification."},
				Tools:          []string{"Read", "Grep", "Glob"},
				PermissionMode: permission.ModePlan,
				MaxTurns:       12,
			},
		},
	})

	sess := agentcore.NewSession(agentcore.Options{
		Provider: prov,
		SystemPrompt: []string{
			"You are a security auditor. Review the codebase for injection, hardcoded secrets, unsafe deserialization, and authz gaps. You may only read — never modify. Delegate pin-pointing a finding to the `tracer` subagent. Produce a concise findings list with file:line references.",
		},
		Tools:          append(readTools, agentTool),
		PermissionMode: permission.ModePlan, // read-only
		WorkingDir:     wd,
		MaxTurns:       40,
	})

	for ev, err := range sess.Prompt(context.Background(), "Audit this repository for security issues.") {
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		if ev.Kind == harness.KindText {
			fmt.Print(ev.Text)
		}
	}
	fmt.Println()
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
