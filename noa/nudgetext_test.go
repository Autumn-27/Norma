package noa

import (
	"strings"
	"testing"
)

func sampleDecision(tier Tier, emergency bool) NudgeDecision {
	d := NudgeDecision{
		ShouldInject: true,
		Tier:         tier,
		ContextUsage: 0.6,
		CompressibleRanges: []RangeInfo{
			{StartRef: "m00012", EndRef: "m00045", Count: 34, Tokens: 12300, ToolPct: 84, TextPct: 16, UserMsgs: 2},
			{StartRef: "m00051", EndRef: "m00080", Count: 28, Tokens: 8100, ToolPct: 61, TextPct: 39},
		},
		ProtectedRanges: []ProtectedRange{
			{StartRef: "m00090", EndRef: "m00095", Count: 6, Tokens: 9700, Tools: []string{"Read", "Bash"}},
		},
		ActiveBlockSpans: []BlockSpan{{BlockID: "b3", Tier: 1, StartRef: "m00001", EndRef: "m00011"}},
		ContextBreakdown: map[string]int{"tool": 61400, "text": 9300, "summaries": 12100},
		Breakdown:        NudgeBreakdown{Growth: 24100, OverLimit: emergency, Emergency: emergency},
	}
	if tier >= 2 {
		d.TierTargetBlocks = []CompressionBlock{
			{BlockID: "b3", Tier: 1, Summary: strings.Repeat("s", 1200), Topic: "JWT 认证方案",
				EffectiveMessageIDs: make([]string, 34), CompressedTokens: 12400},
			{BlockID: "b7", Tier: 1, Summary: strings.Repeat("s", 900),
				EffectiveMessageIDs: make([]string, 28), CompressedTokens: 9100},
		}
	}
	return d
}

// Gentle is the most frequently fired form, so it must carry everything the
// model needs to act — none of it lives in the system prompt.
func TestRenderGentleNudgeIsSelfContained(t *testing.T) {
	voice, text := RenderNudgeText(sampleDecision(1, false), NudgeSections{})
	if voice != VoiceGentle {
		t.Fatalf("voice = %s, want gentle", voice)
	}
	for _, want := range []string{
		"efficiency nudge",
		"Compression Philosophy:",
		"Context breakdown:",
		"WHEN TO COMPRESS",
		"WHEN NOT TO COMPRESS",
		"THE COMPRESS TOOL",
		"HOW TO COMPRESS",
		"Compressible ranges",
		"Active blocks",
		"Compress all ranges in one call",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("gentle nudge is missing %q", want)
		}
	}
	// Upstream's gentle form has no imperative at all, making the most common
	// nudge the least directive.
	if !strings.Contains(text, "compress them now") {
		t.Error("the gentle nudge must actually ask for the action")
	}
	// But a bare imperative would push the model into compressing something the
	// current step still needs.
	if !strings.Contains(text, "If none qualify, continue working") {
		t.Error("the gentle nudge must offer a legitimate way to decline")
	}
}

// At this pressure "should I compress" is settled; "what must I not compress"
// is the question that still matters.
func TestRenderEmergencyNudge(t *testing.T) {
	voice, text := RenderNudgeText(sampleDecision(1, true), NudgeSections{})
	if voice != VoiceEmergency {
		t.Fatalf("voice = %s, want emergency", voice)
	}
	if !strings.Contains(text, "Context limit reached") {
		t.Error("the emergency nudge must open with the alert")
	}
	if !strings.Contains(text, "WHEN NOT TO COMPRESS") {
		t.Error("the emergency nudge must keep the do-not-compress guidance")
	}
	if strings.Contains(text, "WHEN TO COMPRESS\n") {
		t.Error("the emergency nudge should drop the when-to guidance; the decision is already made")
	}
	if !strings.Contains(text, `"startId"`) {
		t.Error("the emergency nudge must carry the JSON skeleton")
	}
}

func TestRenderTierNudge(t *testing.T) {
	_, text := RenderNudgeText(sampleDecision(2, false), NudgeSections{})
	for _, want := range []string{
		"[TIER 2 DISTILLATION TRIGGER]",
		"MULTI-TIER COMPRESSION",
		"Target tier-1 blocks to distill (2)",
		"TIER 2 COMPRESSION — DISTILLATION",
		`Example: Compress({ content: [{ startId: "b3", endId: "b7"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("tier-2 nudge is missing %q", want)
		}
	}
	// It consolidates blocks, not raw messages, so timing guidance for raw
	// content does not apply.
	if strings.Contains(text, "WHEN TO COMPRESS\n\n- A sub-agent") {
		t.Error("a tier nudge should not carry the raw-message timing guidance")
	}
}

func TestRenderTier3Nudge(t *testing.T) {
	d := sampleDecision(3, false)
	d.TierTargetBlocks[0].Tier = 2
	d.TierTargetBlocks[1].Tier = 2
	_, text := RenderNudgeText(d, NudgeSections{})
	if !strings.Contains(text, "[TIER 3 CONDENSATION TRIGGER]") {
		t.Error("tier-3 trigger line missing")
	}
	if !strings.Contains(text, "TIER 3 COMPRESSION — ULTRA-CONDENSATION") {
		t.Error("tier-3 rules missing")
	}
}

func TestRenderTierNudgeEmergencyVoice(t *testing.T) {
	voice, text := RenderNudgeText(sampleDecision(2, true), NudgeSections{})
	if voice != VoiceEmergency {
		t.Fatalf("voice = %s, want emergency", voice)
	}
	if !strings.Contains(text, "[EMERGENCY — TIER 2 DISTILLATION]") {
		t.Errorf("emergency tier trigger missing:\n%s", text[:200])
	}
}

func TestNudgeSectionsOverride(t *testing.T) {
	custom := "CUSTOM EFFICIENCY NOTE"
	_, text := RenderNudgeText(sampleDecision(1, false), NudgeSections{EfficiencyNote: &custom})
	if !strings.Contains(text, custom) {
		t.Error("the override was not applied")
	}
	if strings.Contains(text, "efficiency nudge to compress early") {
		t.Error("the default was not replaced")
	}
}

func TestNudgeSectionsRemoval(t *testing.T) {
	empty := ""
	_, text := RenderNudgeText(sampleDecision(1, false), NudgeSections{EfficiencyNote: &empty})
	if strings.Contains(text, "efficiency nudge to compress early") {
		t.Error("an empty override must remove the section")
	}
	if !strings.HasPrefix(text, "Compression Philosophy:") {
		t.Fatalf("a removed leading section must not leave a blank line: %q", text[:40])
	}
}

func TestFormatBreakdownOmitsEmptyBuckets(t *testing.T) {
	d := NudgeDecision{ContextBreakdown: map[string]int{"tool": 61400, "code": 0, "text": 9300}}
	got := FormatBreakdown(d)
	if strings.Contains(got, "code") {
		t.Fatalf("an empty bucket was rendered: %q", got)
	}
	if !strings.Contains(got, "61K tool") || !strings.Contains(got, "9.3K text") {
		t.Fatalf("breakdown = %q", got)
	}
}

func TestFormatBreakdownShowsGrowth(t *testing.T) {
	d := NudgeDecision{
		ContextBreakdown: map[string]int{"tool": 1000},
		Breakdown:        NudgeBreakdown{Growth: 24100},
	}
	if !strings.Contains(FormatBreakdown(d), "+24K since last nudge") {
		t.Fatalf("breakdown = %q, want the growth line", FormatBreakdown(d))
	}
}

func TestFormatRangesThreeLineTypes(t *testing.T) {
	got := FormatRanges(
		[]RangeInfo{{StartRef: "m00012", EndRef: "m00045", Count: 34, Tokens: 12300, ToolPct: 84, TextPct: 16, UserMsgs: 2}},
		[]ProtectedRange{{StartRef: "m00090", EndRef: "m00095", Count: 6, Tokens: 9700, Tools: []string{"Read", "Bash"}}},
	)
	if !strings.Contains(got, "m00012–m00045  34 msgs  12K [tool 84% | text 16%] · 2 user msgs") {
		t.Errorf("compressible line malformed:\n%s", got)
	}
	if !strings.Contains(got, "[PROTECTED: Read, Bash — not compressible]") {
		t.Errorf("protected line malformed:\n%s", got)
	}
}

// Merging compressible and protected into one ordered list is what reveals a
// span that is partly each; two separate sections would hide it.
func TestFormatRangesMergesAdjacent(t *testing.T) {
	got := FormatRanges(
		[]RangeInfo{{StartRef: "m00001", EndRef: "m00010", Count: 10, Tokens: 5000, ToolPct: 50, TextPct: 50}},
		[]ProtectedRange{{StartRef: "m00011", EndRef: "m00015", Count: 5, Tokens: 3000, Tools: []string{"Read"}}},
	)
	if !strings.Contains(got, "Compressible ranges (1,") {
		t.Fatalf("adjacent spans were not merged:\n%s", got)
	}
	if !strings.Contains(got, "compressible |") || !strings.Contains(got, "protected: Read") {
		t.Fatalf("the merged line must show both parts:\n%s", got)
	}
}

func TestFormatRangesEmpty(t *testing.T) {
	if got := FormatRanges(nil, nil); !strings.Contains(got, "No specific ranges detected") {
		t.Fatalf("empty ranges = %q", got)
	}
}

func TestFormatBlockMapFoldsOldest(t *testing.T) {
	var spans []BlockSpan
	for i := range 12 {
		spans = append(spans, BlockSpan{
			BlockID: "b" + string(rune('1'+i)), Tier: 1,
			StartRef: IndexToRef(i*10 + 1), EndRef: IndexToRef(i*10 + 9),
		})
	}
	got := FormatBlockMap(spans)
	if !strings.Contains(got, "Active blocks (12):") {
		t.Fatalf("block map = %q", got)
	}
	if !strings.Contains(got, "…+4 older · ") {
		t.Fatalf("the fold prefix is missing: %q", got)
	}
	if strings.Count(got, "=") != blockMapMaxShown {
		t.Fatalf("rendered %d blocks, want at most %d", strings.Count(got, "="), blockMapMaxShown)
	}
}

// Every displayed block id carries its tier, so the model can apply the
// one-tier-at-a-time rule before choosing a range.
func TestFormatBlockMapShowsTier(t *testing.T) {
	got := FormatBlockMap([]BlockSpan{{BlockID: "b3", Tier: 2, StartRef: "m00001", EndRef: "m00009"}})
	if !strings.Contains(got, "b3(T2)=") {
		t.Fatalf("block map = %q, want the tier label", got)
	}
}

func TestFormatBlockMapEmpty(t *testing.T) {
	if got := FormatBlockMap(nil); got != "" {
		t.Fatalf("empty block map = %q, want an empty string", got)
	}
}

func TestFormatTierTargetBlocks(t *testing.T) {
	got := FormatTierTargetBlocks([]CompressionBlock{{
		BlockID: "b3", Tier: 1, Topic: "JWT", Summary: strings.Repeat("s", 1200),
		EffectiveMessageIDs: make([]string, 34), CompressedTokens: 12400,
	}})
	if !strings.Contains(got, "Target tier-1 blocks to distill (1):") {
		t.Fatalf("header = %q", got)
	}
	if !strings.Contains(got, `b3  34 msgs  12K→300  "JWT"`) {
		t.Fatalf("block line malformed: %q", got)
	}
}

func TestFormatTierTargetBlocksEmpty(t *testing.T) {
	if got := FormatTierTargetBlocks(nil); !strings.Contains(got, "none") {
		t.Fatalf("empty target list = %q", got)
	}
}
