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

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
	"gopkg.in/yaml.v3"
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
// Listing budget: the available-skills block is regenerated every turn inside
// the Skill tool's prompt, so a large registry would inflate context on every
// request. Cap the total and clip over-long descriptions; once the budget is hit
// the remaining skills degrade to name-only (never dropped — the model must
// still be able to invoke them).
const (
	maxSkillListingChars = 8000
	maxSkillDescChars    = 500
)

// name returns that skill's instructions for the agent to follow.
func (r *Registry) Tool() tool.CoreTool {
	avail := r.listing()
	return tool.Build(tool.Spec{
		Name:        "Skill",
		Description: "Invokes a named skill — a reusable procedure — and returns its step-by-step instructions for you to follow. Use a skill when the task matches one of the available skills below.",
		Prompt:      "Available skills:" + avail,
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
			// The skill's body — instructions plus any caller-supplied context. This
			// travels in an independent user message (Result.Extra) rather than the
			// tool_result, so it reads as standing guidance and so compaction can
			// re-inject it verbatim after a summary boundary (see llm.SkillInvocationMessage
			// + compaction re-injection).
			body := instructions
			if strings.TrimSpace(in.Args) != "" {
				body += "\n\n## Additional context from the caller\n\n" + in.Args
			}
			invocation := llm.SkillInvocationMessage(s.Name, s.Dir, body)

			// The tool_result itself is a short acknowledgement pointing at the
			// injected guidance message.
			ack := fmt.Sprintf("Skill %q loaded. Its instructions were added to the conversation as a separate guidance message — follow them.", s.Name)
			// Host hook: reveal + unlock this skill's MCPs (and any other on-load side
			// effects). Its return text is transient disclosure (deferred-tool names are
			// re-rendered in the system prompt each turn), so it rides on the tool_result.
			if r.OnInvoke != nil {
				if extra := r.OnInvoke(s); extra != "" {
					ack += "\n\n" + extra
				}
			}
			return tool.Result{
				Content: []llm.ContentBlock{llm.TextBlock(ack)},
				Extra:   []llm.Message{invocation},
			}, nil
		},
	})
}

func (r *Registry) names() []string {
	out := append([]string(nil), r.order...)
	sort.Strings(out)
	return out
}

// listing renders the available-skills block ("- name: description" per skill),
// clipping long descriptions and degrading to name-only once maxSkillListingChars
// is reached. Skills are never dropped — the model must be able to invoke each.
func (r *Registry) listing() string {
	var b strings.Builder
	used := 0
	namesOnly := false
	for _, s := range r.List() {
		// Description is the primary discovery field per the agentskills.io spec;
		// it should already describe "what + when to use". Fall back to WhenToUse
		// only for legacy skills that haven't been updated yet.
		desc := s.Description
		if desc == "" {
			desc = s.WhenToUse
		}
		if len(desc) > maxSkillDescChars {
			desc = desc[:maxSkillDescChars] + "…"
		}
		line := "\n- " + s.Name
		if !namesOnly {
			full := line + ": " + desc
			if used+len(full) > maxSkillListingChars {
				namesOnly = true // this and all following skills go name-only
			} else {
				line = full
			}
		}
		b.WriteString(line)
		used += len(line)
	}
	return b.String()
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

// frontmatter is the YAML head of a SKILL.md. Recognised keys follow the
// agentskills.io specification (name, description, license, compatibility);
// whenToUse / when_to_use and mcps / mcp are accepted for compatibility.
type frontmatter struct {
	Name          string       `yaml:"name"`
	Description   string       `yaml:"description"`
	License       string       `yaml:"license"`
	Compatibility string       `yaml:"compatibility"`
	WhenToUse     string       `yaml:"whenToUse"`
	WhenToUseUS   string       `yaml:"when_to_use"`
	MCPs          stringOrList `yaml:"mcps"`
	MCP           stringOrList `yaml:"mcp"`
}

// parse extracts YAML frontmatter and body from a SKILL.md file using a real
// YAML parser (multi-line values, quoted colons, block/flow lists all work).
// Malformed frontmatter degrades gracefully: the fields stay empty and the body
// still loads, so a bad header never drops the whole skill.
func parse(s string) Skill {
	var sk Skill
	body := s
	if strings.HasPrefix(s, "---") {
		rest := strings.TrimLeft(strings.TrimPrefix(s, "---"), "\r\n")
		if i := strings.Index(rest, "\n---"); i >= 0 {
			head := rest[:i]
			body = strings.TrimLeft(rest[i+len("\n---"):], "-\r\n")
			var fm frontmatter
			if err := yaml.Unmarshal([]byte(head), &fm); err == nil {
				sk.Name = fm.Name
				sk.Description = fm.Description
				sk.License = fm.License
				sk.Compatibility = fm.Compatibility
				sk.WhenToUse = fm.WhenToUse
				if sk.WhenToUse == "" {
					sk.WhenToUse = fm.WhenToUseUS
				}
				sk.MCPs = fm.MCPs
				if sk.MCPs == nil {
					sk.MCPs = fm.MCP
				}
			}
		}
	}
	sk.Instructions = strings.TrimSpace(body)
	return sk
}

// stringOrList accepts a frontmatter value written either as a scalar
// ("a" or "a, b") or as a YAML list ([a, b] / block form) and normalises it to a
// trimmed, non-empty []string.
type stringOrList []string

func (s *stringOrList) UnmarshalYAML(value *yaml.Node) error {
	var str string
	if err := value.Decode(&str); err == nil {
		*s = parseList(str)
		return nil
	}
	var list []string
	if err := value.Decode(&list); err == nil {
		out := make([]string, 0, len(list))
		for _, p := range list {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
		if len(out) > 0 {
			*s = out
		}
		return nil
	}
	return nil // unparseable → leave nil rather than fail the whole skill
}

// parseList splits a scalar list value ("a, b" or "[a, b]", quotes optional).
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
