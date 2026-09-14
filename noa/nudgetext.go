package noa

import (
	"fmt"
	"sort"
	"strings"
)

// NudgeVoice is the register a nudge is written in.
type NudgeVoice string

const (
	VoiceGentle    NudgeVoice = "gentle"
	VoiceEmergency NudgeVoice = "emergency"
)

// NudgeSections are the surface-class fragments a host may replace freely.
// Unlike the four compression rules these carry no quality risk: they set tone
// and framing, not what a summary must preserve.
type NudgeSections struct {
	EfficiencyNote  SectionOverride
	EmergencyHeader SectionOverride
	T2Guidance      SectionOverride
	T3Guidance      SectionOverride
}

// The nudge carries every one-off instruction: when to compress, what not to
// compress, how to call the tool, and how to write the summary. None of it is
// resident in the system prompt, so nothing here is a repeat.
const (
	sectionWhenToCompress = `WHEN TO COMPRESS

- A sub-agent or delegated task returned a large result and you have extracted the
  key facts from it.
- Verbose command output (build/test logs, git diff, package installs, directory
  listings) where you have already used the information you need.
- Exploration that led nowhere — keep the one-line lesson, compress the rest.
- Repeated reads of the same file, or repeated status checks, once the decision is
  recorded.
- Resolved discussion threads where the decision is captured in the summary or in
  code.
- Intermediate steps of a completed multi-step task, once the final result is
  recorded.
- A task phase has ended — bug found, root cause identified, research sprint
  wrapped, feature shipped.`

	sectionWhenNotToCompress = `WHEN NOT TO COMPRESS

- Content the current step is actively reading or reasoning about.
- Important user messages — preserve their exact intent, constraints, and
  acceptance criteria. If a message in the range must stay verbatim, shrink the
  range to exclude it rather than compressing it.
- Protected tool outputs — these are hard-excluded and will be dropped from your
  range automatically; you do not need to work around them.
- The most recent messages and the last user message — these sit in a protected
  zone and cannot be compressed. A range entirely inside that zone is rejected.`

	sectionCompressTool = `THE COMPRESS TOOL

  Single range:
    Compress({ content: [{ startId: "m00150", endId: "m00220", summary: "..." }] })

  Batch — multiple unrelated ranges, each with its own topic and summary
  (preferred: one call does more work and the batch is validated as a unit):
    Compress({ content: [
      { topic: "Auth exploration", startId: "m00150", endId: "m00220", summary: "..." },
      { topic: "Deploy debugging", startId: "m00300", endId: "m00350", summary: "..." }
    ]})

  Boundaries accept a message ref (mNNNNN) or a block id (bN). Ranges within one
  call must not overlap.

  topic is a short 3-5 word label. It appears in the summary header and in the
  archive file's metadata. Give each range its own topic when batching.

  summaryMaxChars raises the per-summary limit above the 20000-char default. Use it
  when the content genuinely needs more detail — do not truncate critical
  information just to fit.`

	sectionMultiTier = `MULTI-TIER COMPRESSION

Blocks have tiers 1..3. Compressing a range of BLOCKS (rather than raw messages)
produces a block one tier higher, capped at tier 3.

ONE TIER AT A TIME. A compression absorbs only the blocks at the LOWEST tier
present in the range. Higher-tier blocks inside the range stay active and are NOT
absorbed:

  Compress(b1 .. b4) over [b1:T1, b2:T1, b3:T2, b4:T1]
    -> creates b5:T2, absorbing b1, b2 and b4.  b3 remains active.

This is deliberate. Re-summarizing an already-distilled tier-2 summary alongside
raw tier-1 summaries puts the same content through three lossy passes and detail
disappears fast. If you want to consolidate the tier-2 layer, issue a separate call
covering only tier-2 blocks.

Raw uncompressed messages sitting between the boundary blocks ARE absorbed. Apply
HOW TO COMPRESS to those raw messages and the tier rules to the existing summaries,
so the whole span is covered and nothing is lost.

Tier 3 is terminal. Compressing tier-3 blocks into another tier-3 block reclaims
nothing and is rejected unless the range also contains new uncompressed messages.`

	// The closing clause of the efficiency note is deliberate. Upstream's gentle
	// nudge has no imperative at all, which makes the most frequently fired form
	// the least directive. But a bare imperative would push the model into
	// compressing something the current step still needs — worse than not
	// compressing. Offering an explicit way to decline resolves both.
	defaultEfficiencyNote = `This is an efficiency nudge to compress early and keep context lean — not an
overflow warning. A separate, stronger alert will appear if the context is
actually full.

Pick the ranges below that the current step no longer needs and compress them now,
in one batched call. If none qualify, continue working — this is advisory.`

	defaultEmergencyHeader = `⚠️ Context limit reached — compress now. Prioritize consumed tool outputs.`

	defaultT2Guidance = `Your tier-1 compression summaries have accumulated. Distill them into a single
denser tier-2 summary. Use block IDs as boundaries (startId and endId as bN).`

	defaultT3Guidance = `Your tier-2 compression summaries have accumulated. Condense them further into a
tier-3 ultra-condensed summary. Use block IDs as boundaries (startId and endId as bN).`
)

// truncationNotice tells the model why some tool output looks cut, and what not
// to do about it.
//
// The instinct on seeing a truncated result is to re-run the tool — which
// re-inflates the context at the exact moment it is overflowing. The closing
// sentence is true and load-bearing: truncation applies only to what is sent,
// so it undoes itself once usage drops, and there is nothing to recover.
func truncationNotice(n int) string {
	return fmt.Sprintf(`%d tool result(s) were mechanically truncated to keep this request valid — look for
"%s". Do NOT re-run those tools: that re-inflates the
context you are about to overflow. Compress instead. The truncation is applied only
to what is sent, never to the stored history, so it is undone automatically once
usage drops back below the threshold.`, n, TruncationMarker)
}

func sectionOr(o SectionOverride, def string) string {
	if o != nil {
		return *o
	}
	return def
}

// RenderNudgeText turns a decision into the message the model reads.
func RenderNudgeText(d NudgeDecision, s NudgeSections) (NudgeVoice, string) {
	breakdown := FormatBreakdown(d)
	blockMap := FormatBlockMap(d.ActiveBlockSpans)
	isEmergency := d.Breakdown.Emergency || d.Breakdown.OverLimit

	if d.Tier >= 2 {
		return renderTierNudge(d, s, breakdown, isEmergency)
	}
	if isEmergency {
		return VoiceEmergency, renderEmergencyNudge(d, s, breakdown, blockMap)
	}
	return VoiceGentle, renderGentleNudge(d, s, breakdown, blockMap)
}

func renderTierNudge(d NudgeDecision, s NudgeSections, breakdown string, isEmergency bool) (NudgeVoice, string) {
	isT2 := d.Tier == 2
	word := "CONDENSATION"
	if isT2 {
		word = "DISTILLATION"
	}
	trigger := fmt.Sprintf("[TIER %d %s TRIGGER]", d.Tier, word)
	voice := VoiceGentle
	if isEmergency {
		voice = VoiceEmergency
		trigger = fmt.Sprintf("[EMERGENCY — TIER %d %s] Context limit reached — distill NOW into a denser summary to reclaim tokens.",
			d.Tier, word)
	}
	guidance := sectionOr(s.T2Guidance, defaultT2Guidance)
	rules := Tier2Rules
	if !isT2 {
		guidance = sectionOr(s.T3Guidance, defaultT3Guidance)
		rules = Tier3Rules
	}

	startID, endID := "b1", "b5"
	if len(d.TierTargetBlocks) > 0 {
		startID = d.TierTargetBlocks[0].BlockID
		endID = d.TierTargetBlocks[len(d.TierTargetBlocks)-1].BlockID
	}

	// A tier nudge omits WHEN TO / WHEN NOT TO COMPRESS: it consolidates blocks,
	// not raw messages, so the timing guidance for raw content does not apply.
	parts := []string{
		sectionOr(s.EfficiencyNote, defaultEfficiencyNote),
		"",
		CompressPhilosophy,
		"",
		breakdown,
		"",
		trigger,
		guidance,
		"",
		sectionMultiTier,
		"",
		FormatTierTargetBlocks(d.TierTargetBlocks),
		"",
		sectionCompressTool,
		fmt.Sprintf(`Example: Compress({ content: [{ startId: %q, endId: %q, summary: "..." }] })`, startID, endID),
		"",
		HowToCompressRules,
		"",
		rules,
	}
	if d.TruncatedCount > 0 {
		parts = append([]string{truncationNotice(d.TruncatedCount), ""}, parts...)
	}
	return voice, strings.Join(compact(parts), "\n")
}

func renderEmergencyNudge(d NudgeDecision, s NudgeSections, breakdown, blockMap string) string {
	parts := []string{
		sectionOr(s.EmergencyHeader, defaultEmergencyHeader),
		"",
	}
	if d.TruncatedCount > 0 {
		parts = append(parts, truncationNotice(d.TruncatedCount), "")
	}
	parts = append(parts,
		CompressPhilosophy,
		"",
		breakdown,
		"",
		// At this pressure "should I compress" is settled; "what must I not
		// compress" is the question that still matters.
		sectionWhenNotToCompress,
		"",
		sectionCompressTool,
		`{ "topic": "...", "content": [{ "startId": "<ID>", "endId": "<ID>", "summary": "..." }] }`,
		"Only use IDs from visible messages above. Compress older work first.",
		"",
		HowToCompressRules,
		"",
		FormatRanges(d.CompressibleRanges, d.ProtectedRanges),
	)
	if blockMap != "" {
		parts = append(parts, "", blockMap)
	}
	return strings.Join(compact(parts), "\n")
}

func renderGentleNudge(d NudgeDecision, s NudgeSections, breakdown, blockMap string) string {
	parts := []string{
		sectionOr(s.EfficiencyNote, defaultEfficiencyNote),
		"",
		CompressPhilosophy,
		"",
		breakdown,
		"",
		sectionWhenToCompress,
		"",
		sectionWhenNotToCompress,
		"",
		sectionCompressTool,
		"",
		HowToCompressRules,
		"",
		FormatRanges(d.CompressibleRanges, d.ProtectedRanges),
	}
	if blockMap != "" {
		parts = append(parts, "", blockMap)
	}
	parts = append(parts, "",
		"💡 Compress all ranges in one call (pass multiple content entries: `content: [{...}, {...}]`).")
	return strings.Join(compact(parts), "\n")
}

// compact drops leading blank lines left by a removed section.
func compact(parts []string) []string {
	for len(parts) > 0 && parts[0] == "" {
		parts = parts[1:]
	}
	return parts
}

// FormatBreakdown shows where the context went. Empty buckets are omitted: a
// zero tells the model nothing and costs a line.
func FormatBreakdown(d NudgeDecision) string {
	order := []string{"system", "tool", "summaries", "code", "text"}
	var parts []string
	for _, k := range order {
		if n := d.ContextBreakdown[k]; n > 0 {
			parts = append(parts, fmt.Sprintf("%s %s", FormatTokens(n), k))
		}
	}
	if len(parts) == 0 {
		return ""
	}
	out := "Context breakdown: " + strings.Join(parts, " | ")
	if d.Breakdown.Growth > 0 {
		out += fmt.Sprintf("\n+%s since last nudge", FormatTokens(d.Breakdown.Growth))
	}
	return out
}

// mergedRange is a compressible and/or protected span after merging.
type mergedRange struct {
	startRef, endRef   string
	startNum, endNum   int
	count, tokens      int
	userMsgs           int
	compressibleTokens int
	protectedTokens    int
	protectedCount     int
	protectedTools     []string
	toolPct, textPct   int
	dangerous          bool
}

// FormatRanges lists what may and may not be compressed, oldest first.
//
// Compressible and protected spans are merged into ONE ordered list rather than
// two sections. Splitting them loses the time order and hides overlap — a span
// can be partly compressible and partly protected, which only the merged view
// shows correctly.
func FormatRanges(compressible []RangeInfo, protected []ProtectedRange) string {
	if len(compressible) == 0 && len(protected) == 0 {
		return "[No specific ranges detected — compress any consumed content.]"
	}
	var entries []mergedRange
	for _, r := range compressible {
		s, _ := RefToIndex(r.StartRef)
		e, _ := RefToIndex(r.EndRef)
		entries = append(entries, mergedRange{
			startRef: r.StartRef, endRef: r.EndRef, startNum: s, endNum: e,
			count: r.Count, tokens: r.Tokens, userMsgs: r.UserMsgs,
			compressibleTokens: r.Tokens, toolPct: r.ToolPct, textPct: r.TextPct,
			dangerous: r.Dangerous,
		})
	}
	for _, r := range protected {
		s, _ := RefToIndex(r.StartRef)
		e, _ := RefToIndex(r.EndRef)
		entries = append(entries, mergedRange{
			startRef: r.StartRef, endRef: r.EndRef, startNum: s, endNum: e,
			count: r.Count, tokens: r.Tokens,
			protectedTokens: r.Tokens, protectedCount: r.Count,
			protectedTools: append([]string(nil), r.Tools...),
		})
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].startNum < entries[j].startNum })

	var merged []mergedRange
	for _, e := range entries {
		if n := len(merged); n > 0 && e.startNum <= merged[n-1].endNum+1 {
			last := &merged[n-1]
			last.endRef = e.endRef
			last.endNum = max(last.endNum, e.endNum)
			last.count += e.count
			last.tokens += e.tokens
			last.userMsgs += e.userMsgs
			last.compressibleTokens += e.compressibleTokens
			last.protectedTokens += e.protectedTokens
			last.protectedCount += e.protectedCount
			last.dangerous = last.dangerous || e.dangerous
			for _, tl := range e.protectedTools {
				if !containsString(last.protectedTools, tl) {
					last.protectedTools = append(last.protectedTools, tl)
				}
			}
			continue
		}
		merged = append(merged, e)
	}

	lines := make([]string, 0, len(merged))
	for _, e := range merged {
		lines = append(lines, formatRangeLine(e))
	}
	return fmt.Sprintf("Compressible ranges (%d, oldest first):\n%s", len(merged), strings.Join(lines, "\n"))
}

func formatRangeLine(e mergedRange) string {
	suffix := ""
	if e.dangerous && e.compressibleTokens > 0 {
		suffix = "  ⚠️ NOT recommended unless you are certain."
	}
	userNote := ""
	if e.userMsgs > 0 {
		s := ""
		if e.userMsgs > 1 {
			s = "s"
		}
		userNote = fmt.Sprintf(" · %d user msg%s", e.userMsgs, s)
	}
	switch {
	case e.protectedTokens > 0 && e.compressibleTokens == 0:
		return fmt.Sprintf("  %s–%s  %d msgs  %s [PROTECTED: %s — not compressible]%s",
			e.startRef, e.endRef, e.count, FormatTokens(e.tokens), strings.Join(e.protectedTools, ", "), suffix)
	case e.protectedTokens > 0 && e.compressibleTokens > 0:
		return fmt.Sprintf("  %s–%s  %d msgs  %s [%s compressible | %s protected: %s]%s%s",
			e.startRef, e.endRef, e.count, FormatTokens(e.tokens),
			FormatTokens(e.compressibleTokens), FormatTokens(e.protectedTokens),
			strings.Join(e.protectedTools, ", "), userNote, suffix)
	default:
		return fmt.Sprintf("  %s–%s  %d msgs  %s [tool %d%% | text %d%%]%s%s",
			e.startRef, e.endRef, e.count, FormatTokens(e.tokens), e.toolPct, e.textPct, userNote, suffix)
	}
}

// blockMapMaxShown bounds the block map. Older blocks are the least likely
// compression targets, so they are the ones folded away.
const blockMapMaxShown = 8

// FormatBlockMap lists the active blocks so the model can address them.
func FormatBlockMap(spans []BlockSpan) string {
	if len(spans) == 0 {
		return ""
	}
	hidden := max(len(spans)-blockMapMaxShown, 0)
	shown := spans
	if hidden > 0 {
		shown = spans[len(spans)-blockMapMaxShown:]
	}
	items := make([]string, 0, len(shown))
	for _, s := range shown {
		item := fmt.Sprintf("%s(T%d)=%s–%s", s.BlockID, s.Tier, s.StartRef, s.EndRef)
		items = append(items, item)
	}
	prefix := ""
	if hidden > 0 {
		prefix = fmt.Sprintf("…+%d older · ", hidden)
	}
	return fmt.Sprintf("Active blocks (%d): %s%s", len(spans), prefix, strings.Join(items, " · "))
}

// FormatTierTargetBlocks lists what a tier nudge is asking to consolidate.
func FormatTierTargetBlocks(blocks []CompressionBlock) string {
	if len(blocks) == 0 {
		return "Target blocks: (none — no tier blocks found)"
	}
	lines := make([]string, 0, len(blocks))
	for _, b := range blocks {
		summaryTokens := DefaultCountTokens(b.Summary)
		topic := ""
		if b.Topic != "" {
			topic = fmt.Sprintf(`  %q`, b.Topic)
		}
		lines = append(lines, fmt.Sprintf("  %s  %d msgs  %s→%s%s",
			b.BlockID, len(b.EffectiveMessageIDs),
			FormatTokens(b.CompressedTokens), FormatTokens(summaryTokens), topic))
	}
	return fmt.Sprintf("Target tier-%d blocks to distill (%d):\n%s",
		blocks[0].Tier, len(blocks), strings.Join(lines, "\n"))
}
