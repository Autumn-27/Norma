package tool

import "github.com/Autumn-27/norma/llm"

// Registry is an ordered, name-indexed collection of tools (FR-04.3).
type Registry struct {
	order  []string
	byName map[string]CoreTool
}

// NewRegistry builds a registry from tools, preserving order.
func NewRegistry(tools ...CoreTool) *Registry {
	r := &Registry{byName: map[string]CoreTool{}}
	for _, t := range tools {
		r.Add(t)
	}
	return r
}

// Add registers a tool, replacing any existing one with the same name.
func (r *Registry) Add(t CoreTool) {
	if _, ok := r.byName[t.Name()]; !ok {
		r.order = append(r.order, t.Name())
	}
	r.byName[t.Name()] = t
}

// Get returns the named tool.
func (r *Registry) Get(name string) (CoreTool, bool) {
	t, ok := r.byName[name]
	return t, ok
}

// List returns all tools in registration order.
func (r *Registry) List() []CoreTool {
	out := make([]CoreTool, 0, len(r.order))
	for _, n := range r.order {
		out = append(out, r.byName[n])
	}
	return out
}

// Schemas renders the registry into provider-facing tool definitions, merging
// each tool's extended Prompt into its description.
func (r *Registry) Schemas() []llm.ToolSchema {
	tools := r.List()
	out := make([]llm.ToolSchema, 0, len(tools))
	for _, t := range tools {
		desc := t.Description()
		if p := t.Prompt(); p != "" {
			desc += "\n\n" + p
		}
		out = append(out, llm.ToolSchema{Name: t.Name(), Description: desc, InputSchema: t.InputSchema()})
	}
	return out
}

// DefaultTools returns the core toolset every agent gets: the file/shell tools
// (Read, Write, Edit, MultiEdit, LS, Glob, Grep, Bash) plus Sleep — a small
// context-cancellable wait that pairs with background execution. Network and
// stateful tools (WebFetch, TodoWrite) remain opt-in via their own constructors.
func DefaultTools() []CoreTool {
	return []CoreTool{
		NewRead(), NewWrite(), NewEdit(), NewMultiEdit(), NewLS(), NewGlob(), NewGrep(), NewBash(),
		NewSleep(),
	}
}
