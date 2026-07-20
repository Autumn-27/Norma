// Example planner-memory shows the Phase-4 features: plan mode (the agent
// explores read-only, presents a plan for approval, then implements) and
// persistent memory (relevant memories are auto-injected, and the agent can
// record durable facts). Both layer onto the same Session with no change to the
// core loop.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Autumn-27/norma/agentcore"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/memory"
	"github.com/Autumn-27/norma/permission"
)

func main() {
	model := envOr("AGENT_MODEL", "claude-opus-4-8")
	prov, err := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, Model: model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	wd, _ := os.Getwd()

	// Memory persisted under the working directory.
	store := memory.NewStore(filepath.Join(wd, ".agent-memory"))

	sess := agentcore.NewSession(agentcore.Options{
		Provider:     prov,
		SystemPrompt: []string{"You are a coding agent. Plan before you build; record durable facts (preferences, project constraints) to memory."},
		WorkingDir:   wd,
		MaxTurns:     50,
		// Plan mode: start read-only; ask the terminal to approve the plan.
		Plan: &agentcore.PlanOptions{
			StartInPlanMode: true,
			TargetMode:      permission.ModeAcceptEdits,
			Approver:        approvePlan,
		},
		// Memory: inject relevant memories per prompt and expose the index.
		Memory: &agentcore.MemoryOptions{
			Store:                store,
			AutoInject:           true,
			IncludeIndexInPrompt: true,
		},
	})

	fmt.Printf("planner-memory (%s) — plan mode + memory. Ctrl-D to quit.\n", model)
	in := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n› ")
		if !in.Scan() {
			return
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		for ev, err := range sess.Prompt(context.Background(), line) {
			if err != nil {
				fmt.Fprintf(os.Stderr, "[error: %v]\n", err)
				break
			}
			switch ev.Kind {
			case harness.KindText:
				fmt.Print(ev.Text)
			case harness.KindToolUse:
				fmt.Printf("\n· %s\n", ev.ToolUse.Name)
			case harness.KindResult:
				fmt.Println()
			}
		}
	}
}

// approvePlan prints the plan and asks for terminal approval.
func approvePlan(_ context.Context, plan string) (bool, string) {
	fmt.Printf("\n--- Proposed plan ---\n%s\n---------------------\nApprove this plan? [y/N] ", plan)
	r := bufio.NewReader(os.Stdin)
	line, _ := r.ReadString('\n')
	if s := strings.ToLower(strings.TrimSpace(line)); s == "y" || s == "yes" {
		return true, ""
	}
	fmt.Print("Feedback (optional): ")
	fb, _ := r.ReadString('\n')
	return false, strings.TrimSpace(fb)
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
