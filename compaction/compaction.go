// Package compaction keeps a conversation within a model's context window using
// three mechanisms aligned 1:1 with claude-code (see docs/COMPACTION-SPEC-3.md):
//
//   - MicroCompact — clear stale COMPACTABLE tool-result bodies (non-destructive:
//     the result can be re-read). Keeps the most recent N compactable results and
//     clears the rest.
//   - AutoCompact  — summarize the working head behind a compact boundary at the
//     token threshold (or predictively). Non-destructive: the full history is
//     retained in the message array; only messages after the last boundary are
//     sent to the model (llm.MessagesForAPI).
//   - Reactive     — on a prompt-too-long (413) error, force one AutoCompact and
//     retry. It reuses AutoCompact; it is not a separate algorithm.
//
// Snip and Context-Collapse are intentionally absent: snip's removal decision can
// only come from a model tool / user command (no automatic trigger), and
// context-collapse is disabled by default in claude-code. Thresholds are derived
// from the model's context window (minus a summary reservation and a buffer),
// with a predictive pre-request check. Token counts prefer the previous model
// response's input_tokens and otherwise fall back to a 4/3-scaled per-block
// length estimate.
package compaction

import (
	"context"
	"sort"
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
	// MaxOutputTokens is the model's max output capability; the summary reservation
	// is min(MaxOutputTokens, 20000) (default 8000).
	MaxOutputTokens int
	// KeepRecent is the number of recent messages AutoCompact always preserves as
	// the un-summarized tail (default 8). MicroCompact uses its own microKeepRecent.
	KeepRecent int
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
}

// compactableTools — tools whose results MicroCompact may clear. Mirrors
// claude-code COMPACTABLE_TOOLS (file read, shell, grep/glob, web, edit/write);
// deliberately excludes LS/MultiEdit/NotebookEdit.
var compactableTools = map[string]bool{
	"Read": true, "Bash": true, "Grep": true, "Glob": true,
	"WebFetch": true, "WebSearch": true, "Edit": true, "Write": true,
}

const (
	microClearedMessage   = "[Old tool result content cleared]" // == claude-code TIME_BASED_MC_CLEARED_MESSAGE
	microKeepRecent       = 5                                   // == claude-code KEEP_RECENT (compactable results kept)
	toolResultGrowthGuess = 15000                               // == claude-code TOOL_RESULT_GROWTH_ESTIMATE
	maxAutoCompactFails   = 3                                   // == claude-code MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES
)

// Compactor applies the strategy. Construct with New.
type Compactor struct {
	cfg       Config
	summarize Summarizer
	failures  int
	tripped   bool
}

// New builds a Compactor. summarize may be nil to disable AutoCompact.
func New(cfg Config, summarize Summarizer) *Compactor {
	cfg.defaults()
	return &Compactor{cfg: cfg, summarize: summarize}
}

// --- threshold math ---

// reservedForSummary is min(model max output, 20000). MaxOutputTokens stands in
// for the model's output capability; 0 (unset) reserves the full 20000.
func (c *Compactor) reservedForSummary() int {
	r := c.cfg.MaxOutputTokens
	if r > 20000 || r == 0 {
		r = 20000
	}
	return r
}
func (c *Compactor) effectiveWindow() int { return c.cfg.ContextWindow - c.reservedForSummary() }

// buffer tiers off the EFFECTIVE window (== claude-code getAutocompactBufferTokens).
func (c *Compactor) buffer() int {
	switch ew := c.effectiveWindow(); {
	case ew >= 800000:
		return 50000
	case ew >= 400000:
		return 30000
	default:
		return 13000
	}
}
func (c *Compactor) autoThreshold() int  { return c.effectiveWindow() - c.buffer() }
func (c *Compactor) microThreshold() int { return c.effectiveWindow() * 6 / 10 }

// maxTurnGrowth == claude-code estimateMaxTurnGrowth: min(maxOutput,20000)+15000.
func (c *Compactor) maxTurnGrowth() int { return c.reservedForSummary() + toolResultGrowthGuess }
func (c *Compactor) predictiveOver(tok int) bool {
	return tok+c.maxTurnGrowth() > c.effectiveWindow()
}

// count returns the token count of the API view (post-boundary). When
// lastInputTokens > 0 (the total token size the previous model response
// reported), it is preferred over local estimation — mirroring claude-code's
// tokenCountWithEstimation. After a working-set mutation, callers pass 0 to force
// a fresh per-block estimate.
func (c *Compactor) count(msgs []llm.Message, lastInputTokens int) int {
	if lastInputTokens > 0 {
		return lastInputTokens
	}
	return EstimateTokens(llm.MessagesForAPI(msgs)) * 4 / 3
}

// EstimateTokens approximates the token footprint of messages (~4 chars/token),
// counting each block by type (text, thinking text, tool_use name+input, and
// nested tool_result content). Mirrors claude-code roughTokenCountEstimation
// (without the 4/3 scale, which count applies).
func EstimateTokens(msgs []llm.Message) int {
	chars := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			chars += blockChars(b)
		}
	}
	return chars / 4
}

// blockChars is the per-block character footprint. A Norma ContentBlock has no
// image/document type; if one is added, its fixed 2000-token cost (claude-code
// IMAGE_MAX_TOKEN_SIZE) would be added here.
func blockChars(b llm.ContentBlock) int {
	switch b.Type {
	case llm.BlockText:
		return len(b.Text)
	case llm.BlockThinking:
		return len(b.Thinking)
	case llm.BlockToolUse:
		return len(b.Name) + len(b.Input)
	case llm.BlockToolResult:
		n := 0
		for _, cb := range b.Content {
			n += blockChars(cb)
		}
		return n
	default:
		return len(b.Text)
	}
}

// Pre conditions the history before a model request: MicroCompact to clear stale
// tool results, then AutoCompact at the threshold (or predictively). Both are
// non-destructive. lastInputTokens is the previous response's total token size
// (0 if none yet), used as the initial gate count.
func (c *Compactor) Pre(ctx context.Context, msgs []llm.Message, lastInputTokens int) []llm.Message {
	if c.cfg.ContextWindow <= 0 {
		return msgs
	}
	tok := c.count(msgs, lastInputTokens)
	if tok <= c.microThreshold() {
		return msgs
	}
	// MicroCompact: clear old COMPACTABLE tool results (non-destructive). Only
	// re-estimate when it actually changed the array; otherwise the gate count
	// (which prefers the accurate last-response size) still holds.
	if out, changed := c.microCompact(msgs); changed {
		msgs = out
		tok = c.count(msgs, 0)
	}
	// AutoCompact: at the threshold, or predictively if the next turn would overflow.
	if tok >= c.autoThreshold() || c.predictiveOver(tok) {
		if out, ok := c.autoCompact(ctx, msgs, tok, "auto"); ok {
			msgs = out
		}
	}
	return msgs
}

// Reactive force-compacts after a prompt-too-long error by running one
// AutoCompact (mirrors claude-code tryReactiveCompact, which only summarizes —
// no micro/collapse/snip cascade). Returns whether the API view shrank.
func (c *Compactor) Reactive(ctx context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	before := c.count(msgs, 0)
	if out, ok := c.autoCompact(ctx, msgs, before, "reactive"); ok {
		return out, c.count(out, 0) < before
	}
	return msgs, false
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
// begins. Messages in [start, tailStart) are eligible for summary.
func (c *Compactor) workingBounds(msgs []llm.Message) (start, tailStart int) {
	start = llm.LastBoundaryIndex(msgs) + 1
	tailStart = max(lastNNonMarkerStart(msgs, c.cfg.KeepRecent), start)
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

// --- MicroCompact: keep the most recent N compactable results, clear the rest ---

// microCompact clears the bodies of all COMPACTABLE tool results except the most
// recent microKeepRecent, returning the (possibly new) message slice and whether
// anything changed. It is non-destructive: a cleared result can be re-read.
func (c *Compactor) microCompact(msgs []llm.Message) ([]llm.Message, bool) {
	// Collect compactable tool_use ids in encounter order.
	var ids []string
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Type == llm.BlockToolUse && compactableTools[b.Name] {
				ids = append(ids, b.ID)
			}
		}
	}
	if len(ids) <= microKeepRecent {
		return msgs, false // nothing old enough to clear
	}
	clearSet := make(map[string]bool, len(ids)-microKeepRecent)
	for _, id := range ids[:len(ids)-microKeepRecent] {
		clearSet[id] = true
	}
	changed := false
	out := append([]llm.Message(nil), msgs...)
	for i := range out {
		var nc []llm.ContentBlock
		for j, b := range out[i].Content {
			if b.Type == llm.BlockToolResult && clearSet[b.ToolUseID] && !isCleared(b.Content) {
				if nc == nil {
					nc = append([]llm.ContentBlock(nil), out[i].Content...)
				}
				nc[j].Content = []llm.ContentBlock{llm.TextBlock(microClearedMessage)}
				changed = true
			}
		}
		if nc != nil {
			out[i].Content = nc
		}
	}
	if !changed {
		return msgs, false
	}
	return out, true
}

func isCleared(blocks []llm.ContentBlock) bool {
	return len(blocks) == 1 && blocks[0].Type == llm.BlockText && blocks[0].Text == microClearedMessage
}

// --- AutoCompact: non-destructive boundary + summary ---

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

	// Skill instructions in the summarized head are lossy in the prose summary, so
	// re-inject the latest invocation of each skill verbatim just after the summary
	// (post-boundary → sent to the model). Retained history keeps the originals; we
	// only re-surface skills that fell behind the new boundary.
	reinjected := reinjectSkills(msgs, tailStart)
	boundary := llm.BoundaryMessage(llm.BoundaryMeta{
		Trigger:            trigger,
		PreTokens:          preTok,
		MessagesSummarized: len(toSummarize),
		ActiveSkills:       skillNames(reinjected),
	})
	summaryMsg := llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{
		llm.TextBlock(compactContinuationPreamble + summary),
	}}
	// Retain everything before tailStart (non-destructive); insert the new
	// boundary + summary (+ re-injected skills) just before the preserved recent tail.
	out := make([]llm.Message, 0, len(msgs)+2+len(reinjected))
	out = append(out, msgs[:tailStart]...)
	out = append(out, boundary, summaryMsg)
	out = append(out, reinjected...)
	out = append(out, msgs[tailStart:]...)
	return out, true
}

// compactContinuationPreamble prefixes the inserted summary message, mirroring
// claude-code getCompactUserSummaryMessage. formatCompactSummary already prefixes
// the digest with "Summary:\n".
const compactContinuationPreamble = "This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:\n\n"

// Skill re-injection budget: per-skill and total token caps for verbatim skill
// instructions re-surfaced after a compaction boundary (mirrors claude-code's
// 5K/25K policy). Most-recently-invoked skills win when the total is exceeded.
const (
	maxSkillReinjectTokens      = 5000
	maxSkillReinjectTotalTokens = 25000
)

// skillNames returns the invoked-skill names of the given re-injected messages.
func skillNames(msgs []llm.Message) []string {
	var out []string
	for _, m := range msgs {
		if name, ok := llm.SkillInvocationName(m); ok {
			out = append(out, name)
		}
	}
	return out
}

// reinjectSkills finds the latest invocation of each skill in the history behind
// the new boundary (msgs[:tailStart]) and returns copies to place after the
// summary, newest-first under budget, restored to chronological order. A skill
// whose latest invocation is already in the preserved tail (index >= tailStart)
// is skipped — it is still in the model's view.
func reinjectSkills(msgs []llm.Message, tailStart int) []llm.Message {
	type entry struct {
		msg llm.Message
		idx int
	}
	latest := map[string]*entry{}
	var order []string
	for i, m := range msgs {
		name, ok := llm.SkillInvocationName(m)
		if !ok {
			continue
		}
		if e, seen := latest[name]; seen {
			e.msg, e.idx = m, i
		} else {
			latest[name] = &entry{msg: m, idx: i}
			order = append(order, name)
		}
	}
	if len(latest) == 0 {
		return nil
	}
	// Candidates whose latest invocation fell behind the boundary, newest-first.
	var cands []*entry
	for _, name := range order {
		if e := latest[name]; e.idx < tailStart {
			cands = append(cands, e)
		}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].idx > cands[b].idx })

	kept := make([]*entry, 0, len(cands))
	total := 0
	for _, e := range cands {
		m := llm.TruncateSkillMessage(e.msg, maxSkillReinjectTokens*4)
		t := EstimateTokens([]llm.Message{m})
		if total+t > maxSkillReinjectTotalTokens {
			continue // drop this one; a smaller, older skill may still fit
		}
		total += t
		kept = append(kept, &entry{msg: m, idx: e.idx})
	}
	if len(kept) == 0 {
		return nil
	}
	sort.Slice(kept, func(a, b int) bool { return kept[a].idx < kept[b].idx })
	out := make([]llm.Message, 0, len(kept))
	for _, e := range kept {
		out = append(out, e.msg)
	}
	return out
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
