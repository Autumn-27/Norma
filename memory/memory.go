// Package memory gives the agent persistent, file-backed long-term memory.
// Each memory is a markdown file with YAML frontmatter
// (name / description / type) in a memory directory; a MEMORY.md index lists
// them. The agent recalls memories relevant to a query and records new ones via
// tools, and relevant memories can be auto-injected into the conversation.
package memory

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Type classifies a memory.
type Type string

const (
	TypeUser      Type = "user"      // facts about the user (role, preferences)
	TypeFeedback  Type = "feedback"  // guidance on how the agent should work
	TypeProject   Type = "project"   // ongoing work, goals, constraints
	TypeReference Type = "reference" // pointers to external resources
)

// Memory is a single stored fact.
type Memory struct {
	Name        string
	Description string
	Type        Type
	Content     string
	Path        string
	ModTime     time.Time
}

// Store is a memory directory.
type Store struct {
	Dir string
}

// NewStore roots a store at dir (created on first write).
func NewStore(dir string) *Store { return &Store{Dir: dir} }

const indexFile = "MEMORY.md"

func (s *Store) ensure() error { return os.MkdirAll(s.Dir, 0o755) }

var slugRe = regexp.MustCompile(`[^a-z0-9]+`)

func slug(name string) string {
	s := slugRe.ReplaceAllString(strings.ToLower(name), "-")
	return strings.Trim(s, "-")
}

// Scan reads every memory file (excluding the index), newest first.
func (s *Store) Scan() ([]Memory, error) {
	entries, err := os.ReadDir(s.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Memory
	for _, e := range entries {
		if e.IsDir() || e.Name() == indexFile || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		m, err := s.readFile(filepath.Join(s.Dir, e.Name()))
		if err == nil {
			out = append(out, m)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModTime.After(out[j].ModTime) })
	return out, nil
}

// Read returns the memory with the given name (or its slug).
func (s *Store) Read(name string) (Memory, error) {
	p := filepath.Join(s.Dir, slug(name)+".md")
	return s.readFile(p)
}

func (s *Store) readFile(path string) (Memory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Memory{}, err
	}
	m := parseFrontmatter(string(data))
	m.Path = path
	if m.Name == "" {
		m.Name = strings.TrimSuffix(filepath.Base(path), ".md")
	}
	if info, e := os.Stat(path); e == nil {
		m.ModTime = info.ModTime()
	}
	return m, nil
}

// Save writes a memory file and rebuilds the MEMORY.md index.
func (s *Store) Save(m Memory) error {
	if err := s.ensure(); err != nil {
		return err
	}
	if m.Name == "" {
		return fmt.Errorf("memory: name is required")
	}
	path := filepath.Join(s.Dir, slug(m.Name)+".md")
	if err := os.WriteFile(path, []byte(renderFrontmatter(m)), 0o644); err != nil {
		return err
	}
	return s.rebuildIndex()
}

// Index returns the MEMORY.md content (an index of memories), building it from a
// scan if the file is absent.
func (s *Store) Index() string {
	if data, err := os.ReadFile(filepath.Join(s.Dir, indexFile)); err == nil {
		return string(data)
	}
	return s.buildIndex()
}

func (s *Store) rebuildIndex() error {
	return os.WriteFile(filepath.Join(s.Dir, indexFile), []byte(s.buildIndex()), 0o644)
}

func (s *Store) buildIndex() string {
	mems, _ := s.Scan()
	var b strings.Builder
	b.WriteString("# Memory\n\n")
	if len(mems) == 0 {
		b.WriteString("(no memories yet)\n")
		return b.String()
	}
	for _, m := range mems {
		fmt.Fprintf(&b, "- [%s](%s.md) — %s\n", m.Name, slug(m.Name), m.Description)
	}
	return b.String()
}

// parseFrontmatter extracts name/description/type and the body from a markdown
// file with optional YAML frontmatter.
func parseFrontmatter(s string) Memory {
	var m Memory
	if strings.HasPrefix(s, "---") {
		rest := strings.TrimPrefix(s, "---")
		rest = strings.TrimLeft(rest, "\r\n")
		if i := strings.Index(rest, "\n---"); i >= 0 {
			head := rest[:i]
			body := rest[i+len("\n---"):]
			body = strings.TrimLeft(body, "-")
			m.Content = strings.TrimLeft(body, "\r\n")
			for _, line := range strings.Split(head, "\n") {
				k, v, ok := strings.Cut(line, ":")
				if !ok {
					continue
				}
				k = strings.TrimSpace(k)
				v = strings.TrimSpace(v)
				switch k {
				case "name":
					m.Name = v
				case "description":
					m.Description = v
				case "type":
					m.Type = Type(v)
				}
			}
			return m
		}
	}
	m.Content = s
	return m
}

func renderFrontmatter(m Memory) string {
	var b strings.Builder
	b.WriteString("---\n")
	fmt.Fprintf(&b, "name: %s\n", m.Name)
	fmt.Fprintf(&b, "description: %s\n", m.Description)
	if m.Type != "" {
		fmt.Fprintf(&b, "type: %s\n", m.Type)
	}
	b.WriteString("---\n\n")
	b.WriteString(strings.TrimRight(m.Content, "\n"))
	b.WriteString("\n")
	return b.String()
}
