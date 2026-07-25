// Package skill implements the skill system: named, reusable instruction packages
// the agent can invoke on demand. The format follows the agentskills.io open
// standard — a SKILL.md with YAML frontmatter (name, description, and optional
// license/compatibility fields) followed by Markdown instructions.
package skill

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// Skill is one invokable instruction package.
// Fields align with the agentskills.io specification:
//   - Name and Description are required
//   - License, Compatibility are optional standard fields
//   - WhenToUse is kept for backwards compatibility (pre-spec skills used it;
//     new skills should fold this into Description instead)
type Skill struct {
	Name          string
	Description   string
	License       string
	Compatibility string
	WhenToUse     string // legacy; new skills use Description for when-to-use
	Instructions  string
	// MCPs names MCP servers this skill unlocks when invoked (frontmatter `mcps:`).
	// Hosts use this to reveal + unlock those servers' deferred tools on load — see
	// Registry.OnInvoke.
	MCPs []string
	// Dir is the directory that contains this skill's SKILL.md file.
	// Empty for skills loaded from sources other than the filesystem.
	Dir string
}

// Registry is a name-indexed set of skills.
type Registry struct {
	order  []string
	byName map[string]Skill
	// OnInvoke, if set, is called when a skill is invoked (before its instructions
	// are returned). Its return string is appended to the tool output — hosts use it
	// to reveal + unlock the skill's MCPs (e.g. append a deferred-tools name block
	// and add those tools to the session unlock set). Side effects (unlocking) run
	// here too.
	OnInvoke func(Skill) string
}

// NewRegistry builds a registry from skills.
func NewRegistry(skills ...Skill) *Registry {
	r := &Registry{byName: map[string]Skill{}}
	for _, s := range skills {
		r.Add(s)
	}
	return r
}

// Add registers (or replaces) a skill.
func (r *Registry) Add(s Skill) {
	if _, ok := r.byName[s.Name]; !ok {
		r.order = append(r.order, s.Name)
	}
	r.byName[s.Name] = s
}

// Get returns a skill by name.
func (r *Registry) Get(name string) (Skill, bool) {
	s, ok := r.byName[name]
	return s, ok
}

// List returns skills in registration order.
func (r *Registry) List() []Skill {
	out := make([]Skill, 0, len(r.order))
	for _, n := range r.order {
		out = append(out, r.byName[n])
	}
	return out
}

// Tool returns the Skill tool bound to this registry. Invoking it with a skill
// name returns that skill's instructions for the agent to follow.
func (r *Registry) Tool() tool.CoreTool {
	var avail strings.Builder
	for _, s := range r.List() {
		// Description is the primary discovery field per the agentskills.io spec;
		// it should already describe "what + when to use". Fall back to WhenToUse
		// only for legacy skills that haven't been updated yet.
		desc := s.Description
		if desc == "" {
			desc = s.WhenToUse
		}
		fmt.Fprintf(&avail, "\n- %s: %s", s.Name, desc)
	}
	return tool.Build(tool.Spec{
		Name:        "Skill",
		Description: "Invokes a named skill — a reusable procedure — and returns its step-by-step instructions for you to follow. Use a skill when the task matches one of the available skills below.",
		Prompt:      "Available skills:" + avail.String(),
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string", "description": "The skill to invoke."},
				"args": map[string]any{"type": "string", "description": "Optional additional context for the skill."},
			},
			"required": []any{"name"},
		},
		ReadOnly:   func(json.RawMessage) bool { return true },
		Concurrent: func(json.RawMessage) bool { return false },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, input json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var in struct {
				Name string `json:"name"`
				Args string `json:"args"`
			}
			if err := json.Unmarshal(input, &in); err != nil {
				return tool.Result{}, err
			}
			s, ok := r.Get(in.Name)
			if !ok {
				return tool.Errorf(fmt.Sprintf("Error: unknown skill %q. Available: %s", in.Name, strings.Join(r.names(), ", "))), nil
			}
			instructions := s.Instructions
			if s.Dir != "" {
				instructions = strings.ReplaceAll(instructions, "${SKILL_DIR}", s.Dir)
			}
			var out string
			if s.Dir != "" {
				out = fmt.Sprintf("Skill: %s\nBase directory: %s\n\n%s", s.Name, s.Dir, instructions)
			} else {
				out = fmt.Sprintf("Skill: %s\n\n%s", s.Name, instructions)
			}
			if strings.TrimSpace(in.Args) != "" {
				out += "\n\n## Additional context from the caller\n\n" + in.Args
			}
			// Host hook: reveal + unlock this skill's MCPs (and any other on-load
			// side effects). Its return text is appended to the instructions.
			if r.OnInvoke != nil {
				if extra := r.OnInvoke(s); extra != "" {
					out += "\n\n" + extra
				}
			}
			return tool.Text(out), nil
		},
	})
}

func (r *Registry) names() []string {
	out := append([]string(nil), r.order...)
	sort.Strings(out)
	return out
}

// LoadDir loads skills from a directory. Each skill is a subdirectory containing
// SKILL.md (or a bare .md file) with YAML frontmatter following the agentskills.io
// specification (name, description, and optional license/compatibility fields).
func LoadDir(dir string) (*Registry, error) {
	r := NewRegistry()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		var path string
		switch {
		case e.IsDir():
			p := filepath.Join(dir, e.Name(), "SKILL.md")
			if _, statErr := os.Stat(p); statErr == nil {
				path = p
			}
		case strings.HasSuffix(e.Name(), ".md"):
			path = filepath.Join(dir, e.Name())
		}
		if path == "" {
			continue
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			continue
		}
		s := parse(string(data))
		if s.Name == "" {
			s.Name = strings.TrimSuffix(e.Name(), ".md")
		}
		if e.IsDir() {
			s.Dir = filepath.Join(dir, e.Name())
		} else {
			s.Dir = dir
		}
		r.Add(s)
	}
	return r, nil
}

// parse extracts YAML frontmatter and body from a SKILL.md file.
// Recognised frontmatter keys follow the agentskills.io specification:
// name, description, license, compatibility.
// whenToUse / when_to_use are also parsed for backwards compatibility.
func parse(s string) Skill {
	var sk Skill
	body := s
	if strings.HasPrefix(s, "---") {
		rest := strings.TrimLeft(strings.TrimPrefix(s, "---"), "\r\n")
		if i := strings.Index(rest, "\n---"); i >= 0 {
			head := rest[:i]
			body = strings.TrimLeft(rest[i+len("\n---"):], "-\r\n")
			for _, line := range strings.Split(head, "\n") {
				k, v, ok := strings.Cut(line, ":")
				if !ok {
					continue
				}
				k, v = strings.TrimSpace(k), strings.TrimSpace(v)
				switch k {
				case "name":
					sk.Name = v
				case "description":
					sk.Description = v
				case "license":
					sk.License = v
				case "compatibility":
					sk.Compatibility = v
				case "whenToUse", "when_to_use":
					sk.WhenToUse = v
				case "mcps", "mcp":
					sk.MCPs = parseList(v)
				}
			}
		}
	}
	sk.Instructions = strings.TrimSpace(body)
	return sk
}

// parseList parses a frontmatter list value in either inline form:
// "a, b" or "[a, b]" (quotes optional) → ["a","b"].
func parseList(v string) []string {
	v = strings.TrimSpace(v)
	v = strings.TrimPrefix(v, "[")
	v = strings.TrimSuffix(v, "]")
	var out []string
	for _, p := range strings.Split(v, ",") {
		if p = strings.TrimSpace(strings.Trim(p, `"' `)); p != "" {
			out = append(out, p)
		}
	}
	return out
}
