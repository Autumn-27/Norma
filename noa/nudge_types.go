package noa

// RangeInfo is one contiguous span the model may compress, as reported to it in
// the nudge's range table.
type RangeInfo struct {
	StartRef string
	EndRef   string
	// Count is how many messages the span covers.
	Count int
	// Tokens and Chars size the span; Chars drives the per-batch minimum.
	Tokens int
	Chars  int
	// ToolPct and TextPct are rounded percentages of the span's composition —
	// a tool-heavy span is usually the cheapest thing to compress.
	ToolPct int
	TextPct int
	// UserMsgs counts user messages inside the span, surfaced so the model
	// treats intent-bearing spans with more care.
	UserMsgs int
	// Dangerous marks a span the model should not compress without certainty.
	Dangerous bool
}

// ProtectedRange is a span excluded from compression, reported so the model can
// see why a gap exists rather than guessing.
type ProtectedRange struct {
	StartRef string
	EndRef   string
	Count    int
	Tokens   int
	// Tools names the hard-protected tools responsible for the exclusion.
	Tools []string
}

// BlockSpan is one active block's coverage, for the nudge's block map.
type BlockSpan struct {
	BlockID  string
	Tier     Tier
	StartRef string
	EndRef   string
}

// NudgeBreakdown is the diagnostic detail behind a nudge decision. It is
// surfaced in logs and in the decision's Reason, and is the first thing to look
// at when nudges fire too often or not at all.
type NudgeBreakdown struct {
	Usage           float64
	Growth          int
	GrowthReference int
	GrowthFloor     int
	AdaptiveGrowth  int

	OverLimit bool
	Emergency bool

	MinPressureBenefit int

	PendingT1 int
	PendingT2 int
	PendingT3 int

	CountT1 int
	CountT2 int
}

// NudgeDecision is the structured output of DecideNudge. Rendering it to text
// is a separate step (RenderNudgeText), so a host can present it differently
// without touching the decision logic.
type NudgeDecision struct {
	ShouldInject bool
	// Reason carries the decision diagnostics whether or not it injected —
	// a suppressed nudge is often the thing being debugged.
	Reason string
	// Tier is the tier being asked for; 0 when nothing is injected.
	Tier         Tier
	ContextUsage float64

	CompressibleRanges []RangeInfo
	ProtectedRanges    []ProtectedRange
	ActiveBlockSpans   []BlockSpan
	// TierTargetBlocks are the blocks a tier-2/3 nudge asks to consolidate.
	//
	// Deviation from docs §6.5, which types this []string: the renderer
	// (§6.7 formatTierTargetBlocks) prints each block's message count, token
	// delta and topic, none of which an id alone carries. §6.7 is the binding
	// requirement.
	TierTargetBlocks []CompressionBlock

	Breakdown NudgeBreakdown
	// ContextBreakdown buckets the context by kind: summaries, tool, system,
	// code, text.
	ContextBreakdown map[string]int
	// TruncatedCount is how many tool results emergency-truncate shortened this
	// turn, so the nudge can tell the model not to re-run them.
	TruncatedCount int
}
