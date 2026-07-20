// Package compaction keeps a conversation within a model's context window using
// a graduated, non-destructive strategy:
//
//   - L1 Snip            — drop the oldest working-set turns (last resort)
//   - L2 MicroCompact    — clear stale COMPACTABLE tool-result bodies
//   - L3 Context Collapse— fold old read/search spans and prose
//   - L4 AutoCompact     — summarize the head behind a compact boundary
//
// Compaction is non-destructive: the full history is retained in the message
// array and a boundary marker is inserted; only messages after the last
// boundary are sent to the model (llm.MessagesForAPI). Thresholds are derived
// from the model's context window (minus a summary reservation and a buffer),
// with a predictive pre-request check. Token counts use a real TokenCounter
// when available, falling back to a 4/3-scaled length estimate.
package compaction

import (
	"context"
	"strings"

	"github.com/Autumn-27/norma/llm"
)

// Summarizer condenses messages into a digest. Injected so the host controls
// which model summarizes (and so tests can mock).
type Summarizer func(ctx context.Context, msgs []llm.Message) (string, error)

// Config tunes the compactor.
type Config struct {
	// ContextWindow is the model's total context window in tokens (default 200000).
	ContextWindow int
	// MaxOutputTokens reserves room for the next turn / summary (default 8000).
	MaxOutputTokens int
	// KeepRecent is the number of recent messages always kept in the API view.
	KeepRecent int
	// ToolResultBudget is the per-tool-result token size above which MicroCompact
	// clears a stale result (default 1000).
	ToolResultBudget int
	// CompactTools are the tool names whose old exchanges Collapse folds.
	CompactTools []string
	// CountTokens, when set, gives exact token counts (e.g. llm.NewAnthropicTokenCounter).
	CountTokens llm.TokenCounter
}

func (c *Config) defaults() {
	if c.ContextWindow == 0 {
		c.ContextWindow = 200000
	}
	if c.MaxOutputTokens == 0 {
		c.MaxOutputTokens = 8000
	}
	if c.KeepRecent == 0 {
		c.KeepRecent = 8
	}
	if c.ToolResultBudget == 0 {
		c.ToolResultBudget = 1000
	}
	if c.CompactTools == nil {
		c.CompactTools = []string{"Read", "Grep", "Glob", "RecallMemory", "WebFetch", "WebSearch", "LS"}
	}
}

// compactableTools — tools whose results MicroCompact may clear.
var compactableTools = map[string]bool{
	"Read": true, "Bash": true, "Grep": true, "Glob": true, "LS": true,
	"WebFetch": true, "WebSearch": true, "Edit": true, "Write": true, "MultiEdit": true,
}

const (
	microClearedMessage   = "[Old tool result content cleared]"
	collapsedMessage      = "[read/search result collapsed to save context]"
	toolResultGrowthGuess = 15000 // estimated per-turn tool-result token growth
	maxAutoCompactFails   = 3
)

// Compactor applies the strategy. Construct with New.
type Compactor struct {
	cfg         Config
	summarize   Summarizer
	collapseSet map[string]bool
	failures    int
	tripped     bool
}

// New builds a Compactor. summarize may be nil to disable AutoCompact.
func New(cfg Config, summarize Summarizer) *Compactor {
	cfg.defaults()
	cs := make(map[string]bool, len(cfg.CompactTools))
	for _, n := range cfg.CompactTools {
		cs[n] = true
	}
	return &Compactor{cfg: cfg, summarize: summarize, collapseSet: cs}
}

// --- threshold math ---

func (c *Compactor) reservedForSummary() int {
	r := c.cfg.MaxOutputTokens
	if r > 20000 || r == 0 {
		r = 20000
	}
	return r
}
func (c *Compactor) effectiveWindow() int { return c.cfg.ContextWindow - c.reservedForSummary() }
func (c *Compactor) buffer() int {
	switch {
	case c.cfg.ContextWindow >= 800000:
		return 50000
	case c.cfg.ContextWindow >= 400000:
		return 30000
	default:
		return 13000
	}
}
func (c *Compactor) autoThreshold() int     { return c.effectiveWindow() - c.buffer() }
func (c *Compactor) microThreshold() int    { return c.effectiveWindow() * 6 / 10 }
func (c *Compactor) collapseThreshold() int { return c.effectiveWindow() * 7 / 10 }
func (c *Compactor) maxTurnGrowth() int     { return c.cfg.MaxOutputTokens + toolResultGrowthGuess }
func (c *Compactor) predictiveOver(tok int) bool {
	return tok+c.maxTurnGrowth() > c.effectiveWindow()
}

// count returns the token count of the API view (post-boundary), using the real
// counter when available and otherwise a 4/3-scaled length estimate.
func (c *Compactor) count(ctx context.Context, msgs []llm.Message) int {
	if c.cfg.CountTokens != nil {
		if n, err := c.cfg.CountTokens(ctx, msgs); err == nil {
			return n
		}
	}
	return EstimateTokens(llm.MessagesForAPI(msgs)) * 4 / 3
}

// EstimateTokens approximates the token footprint of messages (~4 chars/token).
func EstimateTokens(msgs []llm.Message) int {
	chars := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			chars += len(b.Text) + len(b.Input)
			for _, cb := range b.Content {
				chars += len(cb.Text)
			}
		}
	}
	return chars / 4
}

// Pre conditions the history before a model request, applying the lightest
// strategy that brings the API view under budget and escalating as needed. It
// is non-destructive: AutoCompact inserts a boundary and retains prior history.
func (c *Compactor) Pre(ctx context.Context, msgs []llm.Message) []llm.Message {
	if c.cfg.ContextWindow <= 0 {
		return msgs
	}
	tok := c.count(ctx, msgs)
	if tok <= c.microThreshold() {
		return msgs
	}
	if tok > c.microThreshold() {
		msgs = c.microCompact(msgs)
		tok = c.count(ctx, msgs)
	}
	if tok > c.collapseThreshold() {
		msgs = c.collapse(msgs)
		tok = c.count(ctx, msgs)
	}
	// AutoCompact: at the threshold, or predictively if the next turn would overflow.
	if tok >= c.autoThreshold() || c.predictiveOver(tok) {
		if out, ok := c.autoCompact(ctx, msgs, tok, "auto"); ok {
			msgs = out
			tok = c.count(ctx, msgs)
		}
	}
	// Snip is the last resort if summarization couldn't run (e.g. breaker tripped).
	if tok >= c.autoThreshold() {
		msgs = c.snip(msgs)
	}
	return msgs
}

// Reactive force-compacts after a prompt-too-long error: collapse (drain) →
// AutoCompact → snip, cheapest first.
func (c *Compactor) Reactive(ctx context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	before := c.count(ctx, msgs)
	msgs = c.microCompact(msgs)
	msgs = c.collapse(msgs)
	if c.count(ctx, msgs) >= before {
		if out, ok := c.autoCompact(ctx, msgs, before, "reactive"); ok {
			msgs = out
		}
	}
	msgs = c.snip(msgs)
	return msgs, c.count(ctx, msgs) < before
}

// IsOverflow is the method form (satisfies the harness Compactor interface).
func (c *Compactor) IsOverflow(err error) bool { return IsOverflow(err) }

// IsOverflow reports whether err is a context-length / prompt-too-long failure.
func IsOverflow(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "too long") || strings.Contains(s, "context length") ||
		strings.Contains(s, "context_length") || strings.Contains(s, "maximum context") ||
		strings.Contains(s, "status 413")
}

// --- working-set geometry ---

// workingBounds returns [start, tailStart): start is just after the last
// boundary marker (0 if none); tailStart is where the preserved recent tail
// begins. Messages in [start, tailStart) are eligible for clearing/summary.
func (c *Compactor) workingBounds(msgs []llm.Message) (start, tailStart int) {
	start = llm.LastBoundaryIndex(msgs) + 1
	tailStart = lastNNonMarkerStart(msgs, c.cfg.KeepRecent)
	if tailStart < start {
		tailStart = start
	}
	return
}

func lastNNonMarkerStart(msgs []llm.Message, keep int) int {
	count, i := 0, len(msgs)
	for i > 0 && count < keep {
		i--
		if !llm.IsBoundaryMarker(msgs[i]) {
			count++
		}
	}
	return i
}

func toolNameMap(msgs []llm.Message) map[string]string {
	m := map[string]string{}
	for _, msg := range msgs {
		for _, b := range msg.Content {
			if b.Type == llm.BlockToolUse {
				m[b.ID] = b.Name
			}
		}
	}
	return m
}

// --- L2 MicroCompact: clear COMPACTABLE tool results in the working head ---

func (c *Compactor) microCompact(msgs []llm.Message) []llm.Message {
	start, tailStart := c.workingBounds(msgs)
	if tailStart <= start {
		return msgs
	}
	names := toolNameMap(msgs)
	out := append([]llm.Message(nil), msgs...)
	for i := start; i < tailStart; i++ {
		nc := append([]llm.ContentBlock(nil), out[i].Content...)
		for j, b := range nc {
			if b.Type == llm.BlockToolResult && compactableTools[names[b.ToolUseID]] {
				if estTokens(flatten(b.Content)) > c.cfg.ToolResultBudget {
					nc[j].Content = []llm.ContentBlock{llm.TextBlock(microClearedMessage)}
				}
			}
		}
		out[i].Content = nc
	}
	return out
}

// --- L3 Context Collapse: fold read/search results + truncate prose ---

func (c *Compactor) collapse(msgs []llm.Message) []llm.Message {
	start, tailStart := c.workingBounds(msgs)
	if tailStart <= start {
		return msgs
	}
	names := toolNameMap(msgs)
	out := append([]llm.Message(nil), msgs...)
	for i := start; i < tailStart; i++ {
		nc := append([]llm.ContentBlock(nil), out[i].Content...)
		for j, b := range nc {
			switch b.Type {
			case llm.BlockText, llm.BlockThinking:
				if r := []rune(b.Text); len(r) > 200 {
					nc[j].Text = string(r[:200]) + " …[collapsed]"
				}
			case llm.BlockToolResult:
				if c.collapseSet[names[b.ToolUseID]] {
					nc[j].Content = []llm.ContentBlock{llm.TextBlock(collapsedMessage)}
				}
			}
		}
		out[i].Content = nc
	}
	return out
}

// --- L4 AutoCompact: non-destructive boundary + summary ---

func (c *Compactor) autoCompact(ctx context.Context, msgs []llm.Message, preTok int, trigger string) ([]llm.Message, bool) {
	if c.summarize == nil || c.tripped {
		return msgs, false
	}
	start, tailStart := c.workingBounds(msgs)
	if tailStart <= start {
		return msgs, false
	}
	toSummarize := contentOnly(msgs[start:tailStart])
	if len(toSummarize) == 0 {
		return msgs, false
	}
	summary, err := c.summarize(ctx, toSummarize)
	if err != nil {
		c.failures++
		if c.failures >= maxAutoCompactFails {
			c.tripped = true
		}
		return msgs, false
	}
	c.failures = 0

	boundary := llm.BoundaryMessage(llm.BoundaryMeta{
		Trigger:            trigger,
		PreTokens:          preTok,
		MessagesSummarized: len(toSummarize),
	})
	summaryMsg := llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{
		llm.TextBlock("[conversation summarized to save context]\n\n" + summary),
	}}
	// Retain everything before tailStart (non-destructive); insert the new
	// boundary + summary just before the preserved recent tail.
	out := make([]llm.Message, 0, len(msgs)+2)
	out = append(out, msgs[:tailStart]...)
	out = append(out, boundary, summaryMsg)
	out = append(out, msgs[tailStart:]...)
	return out, true
}

// --- L1 Snip: drop oldest working-set messages (last resort) ---

func (c *Compactor) snip(msgs []llm.Message) []llm.Message {
	target := c.autoThreshold()
	for EstimateTokens(llm.MessagesForAPI(msgs))*4/3 > target {
		start, tailStart := c.workingBounds(msgs)
		if tailStart <= start {
			return msgs
		}
		// remove the oldest working-set message
		msgs = append(msgs[:start:start], msgs[start+1:]...)
	}
	return msgs
}

func contentOnly(msgs []llm.Message) []llm.Message {
	out := make([]llm.Message, 0, len(msgs))
	for _, m := range msgs {
		if !llm.IsBoundaryMarker(m) {
			out = append(out, m)
		}
	}
	return out
}

func flatten(blocks []llm.ContentBlock) string {
	var s strings.Builder
	for _, b := range blocks {
		s.WriteString(b.Text)
	}
	return s.String()
}

func estTokens(s string) int { return len(s) / 4 }
