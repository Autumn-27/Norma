package memory

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

var wordRe = regexp.MustCompile(`[a-z0-9_]+`)

var stop = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "of": true,
	"to": true, "in": true, "is": true, "it": true, "for": true, "on": true,
	"do": true, "i": true, "you": true, "my": true, "me": true, "what": true,
	"how": true, "can": true, "with": true, "this": true, "that": true,
}

func tokenize(s string) []string {
	words := wordRe.FindAllString(strings.ToLower(s), -1)
	var out []string
	for _, w := range words {
		if len(w) >= 2 && !stop[w] {
			out = append(out, w)
		}
	}
	return out
}

func score(m Memory, tokens []string) int {
	name := strings.ToLower(m.Name + " " + string(m.Type))
	desc := strings.ToLower(m.Description)
	body := strings.ToLower(m.Content)
	s := 0
	for _, t := range tokens {
		if strings.Contains(name, t) {
			s += 3
		}
		if strings.Contains(desc, t) {
			s += 2
		}
		if strings.Contains(body, t) {
			s++
		}
	}
	return s
}

// Relevant returns up to max memories scored against the query (keyword match on
// name/description/content), most relevant first, recency breaking ties.
func (s *Store) Relevant(query string, max int) []Memory {
	mems, _ := s.Scan()
	tokens := tokenize(query)
	if len(tokens) == 0 {
		return nil
	}
	type sc struct {
		m Memory
		s int
	}
	var ranked []sc
	for _, m := range mems {
		if v := score(m, tokens); v > 0 {
			ranked = append(ranked, sc{m, v})
		}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].s != ranked[j].s {
			return ranked[i].s > ranked[j].s
		}
		return ranked[i].m.ModTime.After(ranked[j].m.ModTime)
	})
	out := make([]Memory, 0, max)
	for i, r := range ranked {
		if max > 0 && i >= max {
			break
		}
		out = append(out, r.m)
	}
	return out
}

// Age renders a human age like "today" / "yesterday" / "5 days ago".
func Age(m Memory, now time.Time) string {
	if m.ModTime.IsZero() {
		return "unknown age"
	}
	days := int(now.Sub(m.ModTime).Hours() / 24)
	switch {
	case days <= 0:
		return "today"
	case days == 1:
		return "yesterday"
	default:
		return fmt.Sprintf("%d days ago", days)
	}
}

// stalenessNote returns a caveat for memories older than a day, or "" if fresh.
func stalenessNote(m Memory, now time.Time) string {
	if m.ModTime.IsZero() || now.Sub(m.ModTime).Hours() < 48 {
		return ""
	}
	return "(This memory is a point-in-time note, not live state — verify claims about code against the current source before relying on them.)"
}

// SystemSegment renders the memory index for inclusion in the system prompt so
// the agent knows what it remembers and can RecallMemory for details.
func (s *Store) SystemSegment() string {
	return "# Memory\nYou have persistent long-term memory. Index of saved memories — call RecallMemory to read full content, and RecordMemory to save a durable fact:\n\n" + s.Index()
}

// RenderForInjection formats memories as context to inject before a user turn.
func RenderForInjection(mems []Memory, now time.Time) string {
	if len(mems) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Relevant memories (saved earlier):\n")
	for _, m := range mems {
		fmt.Fprintf(&b, "\n--- %s (saved %s) ---\n", m.Name, Age(m, now))
		if note := stalenessNote(m, now); note != "" {
			b.WriteString(note + "\n")
		}
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("\n")
	}
	return b.String()
}
