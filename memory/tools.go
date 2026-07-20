package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// Tools returns the RecallMemory and RecordMemory tools bound to a store.
func Tools(s *Store) []tool.CoreTool {
	return []tool.CoreTool{RecallTool(s), RecordTool(s)}
}

// RecallTool lets the agent read memories relevant to a query.
func RecallTool(s *Store) tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:        "RecallMemory",
		Description: "Searches your long-term memory for facts relevant to a query and returns their full content. Use it when prior context (user preferences, project state, past decisions) would help.",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string", "description": "What to recall."}},
			"required":   []any{"query"},
		},
		ReadOnly:   func(json.RawMessage) bool { return true },
		Concurrent: func(json.RawMessage) bool { return true },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, input json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var in struct {
				Query string `json:"query"`
			}
			if err := json.Unmarshal(input, &in); err != nil {
				return tool.Result{}, err
			}
			mems := s.Relevant(in.Query, 5)
			if len(mems) == 0 {
				return tool.Text("No relevant memories found."), nil
			}
			return tool.Text(RenderForInjection(mems, time.Now())), nil
		},
	})
}

// RecordTool lets the agent persist a durable fact to memory.
func RecordTool(s *Store) tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:        "RecordMemory",
		Description: "Saves a durable fact to long-term memory so it persists across sessions. Record stable facts (user preferences, project constraints, decisions) — not transient conversation detail.",
		Prompt:      "type is one of: user (about the user), feedback (how you should work), project (ongoing work/constraints), reference (external resources). Give a short name and a one-line description.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":        map[string]any{"type": "string", "description": "Short title for the memory."},
				"description": map[string]any{"type": "string", "description": "One-line summary."},
				"type":        map[string]any{"type": "string", "description": "user | feedback | project | reference"},
				"content":     map[string]any{"type": "string", "description": "The fact to remember (markdown)."},
			},
			"required": []any{"name", "content"},
		},
		ReadOnly:   func(json.RawMessage) bool { return false },
		Concurrent: func(json.RawMessage) bool { return false },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, input json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var in struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Type        string `json:"type"`
				Content     string `json:"content"`
			}
			if err := json.Unmarshal(input, &in); err != nil {
				return tool.Result{}, err
			}
			if strings.TrimSpace(in.Name) == "" || strings.TrimSpace(in.Content) == "" {
				return tool.Errorf("Error: name and content are required"), nil
			}
			m := Memory{Name: in.Name, Description: in.Description, Type: Type(in.Type), Content: in.Content}
			if err := s.Save(m); err != nil {
				return tool.Errorf("Error saving memory: " + err.Error()), nil
			}
			return tool.Text(fmt.Sprintf("Saved memory %q.", in.Name)), nil
		},
	})
}
