// Example custom-llm shows (1) pointing the SDK at any model via the OpenAI or
// Anthropic wire format, (2) a fully custom system prompt for a non-coding
// scenario, and (3) a custom tool. The loop, scheduler, and permission boundary
// are unchanged — only the provider, prompt, and tools differ.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/Autumn-27/norma/agentcore"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

func newProvider() (llm.Provider, string) {
	// Switch any of these in — same SDK, different model/format.
	//
	// Anthropic:
	//   llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, Model: "claude-opus-4-8"})
	// OpenAI:
	//   llm.NewProvider(llm.Config{Format: llm.FormatOpenAI, Model: "gpt-4o"})
	// DeepSeek (OpenAI-compatible gateway):
	//   llm.NewProvider(llm.Config{Format: llm.FormatOpenAI, BaseURL: "https://api.deepseek.com/v1", Model: "deepseek-chat"})
	model := envOr("AGENT_MODEL", "gpt-4o")
	p, err := llm.NewProvider(llm.Config{Format: llm.FormatOpenAI, Model: model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return p, model
}

// addTool is a custom tool: a non-coding capability plugged into the same loop.
func addTool() tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:        "add",
		Description: "Adds two numbers and returns their sum.",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"a": map[string]any{"type": "number"}, "b": map[string]any{"type": "number"}},
			"required":   []any{"a", "b"},
		},
		ReadOnly:   func(json.RawMessage) bool { return true },
		Concurrent: func(json.RawMessage) bool { return true },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var a struct{ A, B float64 }
			if err := json.Unmarshal(in, &a); err != nil {
				return tool.Result{}, err
			}
			return tool.Text(fmt.Sprintf("%g", a.A+a.B)), nil
		},
	})
}

func main() {
	prov, model := newProvider()
	sess := agentcore.NewSession(agentcore.Options{
		Provider: prov,
		// Fully custom, non-coding system prompt — the SDK adds nothing.
		SystemPrompt:   []string{"You are a friendly math tutor. Use the `add` tool for any addition instead of computing it yourself, then explain the result in one sentence."},
		Tools:          []tool.CoreTool{addTool()},
		PermissionMode: permission.ModeBypass,
	})

	fmt.Printf("(model: %s)\n", model)
	for ev, err := range sess.Prompt(context.Background(), "What is 2024 plus 1?") {
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
