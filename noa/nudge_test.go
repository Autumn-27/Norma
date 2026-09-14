package noa

import (
	"strings"
	"testing"
)

// nudgeAt builds a decision at a given context size with a given tier-1 backlog.
func nudgeAt(t *testing.T, tokenCount, t1Tokens int, mutate func(*CompressionState, *Config)) NudgeDecision {
	t.Helper()
	cfg := DefaultConfig(200000)
	st := CreateInitialState("s", "/a")
	if mutate != nil {
		mutate(&st, &cfg)
	}
	rec := Recommendation{}
	if t1Tokens > 0 {
		rec.CompressibleRanges = []RangeInfo{{
			StartRef: "m00002", EndRef: "m00050", Count: 20,
			Tokens: t1Tokens, Chars: t1Tokens * 4, ToolPct: 80, TextPct: 20,
		}}
	}
	return DecideNudge(DecideNudgeInput{
		State: st, Config: cfg, TokenCount: tokenCount, Recommendation: rec,
	})
}

func TestResolveAdaptiveGrowth(t *testing.T) {
	n := DefaultConfig(200000).Nudge
	// min(50000, max(50000, round(200000*0.05)=10000)) = 50000
	if got := resolveAdaptiveGrowth(200000, n); got != 50000 {
		t.Fatalf("adaptive growth for a 200k window = %d, want 50000", got)
	}
	if got := resolveAdaptiveGrowth(0, n); got != n.GrowthFloor {
		t.Fatalf("adaptive growth with no limit = %d, want the floor", got)
	}
}

func TestGrowthFloor(t *testing.T) {
	n := DefaultConfig(200000).Nudge
	// max(20000, 0.45*50000=22500) = 22500
	if got := growthFloorOf(50000, n); got != 22500 {
		t.Fatalf("growth floor = %d, want 22500", got)
	}
}

// Below the cadence gate the nudge stays quiet, however much is compressible.
func TestNudgeHoldsBelowCadence(t *testing.T) {
	d := nudgeAt(t, 100000, 60000, func(st *CompressionState, _ *Config) {
		st.Nudge.LastNudgeShownTokens = 95000 // grew only 5000
		st.Nudge.LastPerMessageNudgeTokens = 95000
	})
	if d.ShouldInject {
		t.Fatalf("injected despite growing only 5000 of 22500: %s", d.Reason)
	}
	if !strings.Contains(d.Reason, "below cadence") {
		t.Fatalf("reason = %q, want it to name the cadence gate", d.Reason)
	}
}

func TestNudgeFiresOnGrowthWithBacklog(t *testing.T) {
	d := nudgeAt(t, 120000, 60000, func(st *CompressionState, _ *Config) {
		st.Nudge.LastNudgeShownTokens = 90000 // grew 30000 > 22500
		st.Nudge.LastPerMessageNudgeTokens = 90000
	})
	if !d.ShouldInject || d.Tier != 1 {
		t.Fatalf("decision = inject:%v tier:%d, want a tier-1 nudge: %s", d.ShouldInject, d.Tier, d.Reason)
	}
}

// Growth alone is not enough: there must be enough to reclaim.
func TestNudgeHoldsWhenBacklogTooSmall(t *testing.T) {
	d := nudgeAt(t, 120000, 5000, func(st *CompressionState, _ *Config) {
		st.Nudge.LastNudgeShownTokens = 90000
		st.Nudge.LastPerMessageNudgeTokens = 90000
	})
	if d.ShouldInject {
		t.Fatalf("injected with only 5000 pending against a 50000 threshold: %s", d.Reason)
	}
}

// A session arriving with a large backlog has no baseline to have grown from;
// without this branch the cadence gate would hold it back indefinitely.
func TestNudgeFirstSightBypassesCadence(t *testing.T) {
	d := nudgeAt(t, 100000, 60000, nil) // 50% usage, no baselines at all
	if !d.ShouldInject {
		t.Fatalf("first sight with a large backlog must inject: %s", d.Reason)
	}
}

func TestNudgeFirstSightNeedsUsageFloor(t *testing.T) {
	// 40% usage is below MinContextLimitPct (45%).
	d := nudgeAt(t, 80000, 60000, nil)
	if d.ShouldInject {
		t.Fatalf("first sight below the usage floor must not inject: %s", d.Reason)
	}
}

// Past the pressure band there is no growth gate — waiting for more growth is
// not a strategy when the context is nearly full.
func TestNudgePressureIgnoresCadence(t *testing.T) {
	d := nudgeAt(t, 160000, 30000, func(st *CompressionState, _ *Config) {
		st.Nudge.LastNudgeShownTokens = 159000 // grew only 1000
		st.Nudge.LastPerMessageNudgeTokens = 159000
	})
	if !d.ShouldInject {
		t.Fatalf("pressure must inject regardless of cadence: %s", d.Reason)
	}
	if !d.Breakdown.OverLimit {
		t.Fatal("OverLimit must be set at 80% usage")
	}
}

// Under pressure with nothing worth reclaiming, nudging would burn a turn and
// leave the context exactly as full.
func TestNudgePressureRespectsMinimumBenefit(t *testing.T) {
	d := nudgeAt(t, 160000, 1000, nil) // below minPressureBenefit (5000)
	if d.ShouldInject {
		t.Fatalf("injected under pressure for a 1000-token benefit: %s", d.Reason)
	}
	if !strings.Contains(d.Reason, "minimum benefit") {
		t.Fatalf("reason = %q, want it to name the benefit floor", d.Reason)
	}
}

func TestNudgeEmergencyFlag(t *testing.T) {
	d := nudgeAt(t, 195000, 30000, nil) // 97.5%
	if !d.Breakdown.Emergency {
		t.Fatal("Emergency must be set past 95%")
	}
	if !d.ShouldInject {
		t.Fatalf("emergency with a real backlog must inject: %s", d.Reason)
	}
}

// Enough tier-1 blocks have accumulated to be worth consolidating.
func TestNudgeTier2OnBlockCount(t *testing.T) {
	d := nudgeAt(t, 120000, 0, func(st *CompressionState, _ *Config) {
		for i := range 5 {
			b := block("b"+string(rune('1'+i)), 1, "gone")
			b.Summary = strings.Repeat("summary text ", 40)
			st.Blocks = append(st.Blocks, b)
		}
		st.Nudge.LastNudgeShownTokens = 90000
		st.Nudge.LastPerMessageNudgeTokens = 90000
	})
	if !d.ShouldInject || d.Tier != 2 {
		t.Fatalf("decision = inject:%v tier:%d, want tier 2: %s", d.ShouldInject, d.Tier, d.Reason)
	}
	if len(d.TierTargetBlocks) != 5 {
		t.Fatalf("TierTargetBlocks = %d, want the five tier-1 blocks", len(d.TierTargetBlocks))
	}
}

// Block count alone is not enough: a nearly empty context has nothing to gain
// from consolidation, and the turn would be wasted.
func TestNudgeTier2NeedsUsageFloor(t *testing.T) {
	d := nudgeAt(t, 60000, 0, func(st *CompressionState, _ *Config) { // 30% usage
		for i := range 5 {
			b := block("b"+string(rune('1'+i)), 1, "gone")
			b.Summary = strings.Repeat("summary text ", 40)
			st.Blocks = append(st.Blocks, b)
		}
		st.Nudge.LastNudgeShownTokens = 30000
		st.Nudge.LastPerMessageNudgeTokens = 30000
	})
	if d.ShouldInject && d.Tier == 2 {
		t.Fatalf("tier 2 fired below the usage floor: %s", d.Reason)
	}
}

// Each tier is paced separately, so an ignored tier-2 nudge does not reappear
// on every following turn.
func TestNudgeTierCadence(t *testing.T) {
	setup := func(st *CompressionState, _ *Config) {
		for i := range 5 {
			b := block("b"+string(rune('1'+i)), 1, "gone")
			b.Summary = strings.Repeat("summary text ", 40)
			st.Blocks = append(st.Blocks, b)
		}
		st.Nudge.LastNudgeShownTokens = 90000
		st.Nudge.LastPerMessageNudgeTokens = 90000
		st.Nudge.LastShownByTier = map[Tier]int{2: 119000} // shown 1000 tokens ago
	}
	d := nudgeAt(t, 120000, 0, setup)
	if d.ShouldInject {
		t.Fatalf("tier 2 re-fired only 1000 tokens after the last one: %s", d.Reason)
	}
	if !strings.Contains(d.Reason, "cadence") {
		t.Fatalf("reason = %q, want it to name the tier cadence", d.Reason)
	}
}

func TestNudgeTiersDisabled(t *testing.T) {
	d := nudgeAt(t, 120000, 0, func(st *CompressionState, cfg *Config) {
		cfg.Tiers.Enabled = false
		for i := range 8 {
			b := block("b"+string(rune('1'+i)), 1, "gone")
			b.Summary = strings.Repeat("summary text ", 40)
			st.Blocks = append(st.Blocks, b)
		}
		st.Nudge.LastNudgeShownTokens = 90000
		st.Nudge.LastPerMessageNudgeTokens = 90000
	})
	if d.ShouldInject && d.Tier >= 2 {
		t.Fatalf("a tier nudge fired with tiers disabled: %s", d.Reason)
	}
}

// The reason string is the diagnostic surface; it must be present whether or
// not the nudge fired.
func TestNudgeAlwaysExplainsItself(t *testing.T) {
	for _, d := range []NudgeDecision{
		nudgeAt(t, 100000, 60000, nil),
		nudgeAt(t, 10000, 0, nil),
		nudgeAt(t, 160000, 1000, nil),
	} {
		if d.Reason == "" {
			t.Fatalf("decision has no reason: %+v", d.Breakdown)
		}
		if !strings.Contains(d.Reason, "ready:") {
			t.Fatalf("reason = %q, want the per-tier backlog summary", d.Reason)
		}
	}
}

func TestComputeContextBreakdown(t *testing.T) {
	msgs := []CoreMessage{
		{Role: RoleUser, ContentType: CTText, Text: strings.Repeat("a", 400)},
		{Role: RoleTool, ContentType: CTToolResult, Text: strings.Repeat("b", 800)},
		{Role: RoleUser, ContentType: CTText, Text: SummaryHeader + strings.Repeat("c", 400)},
		{Role: RoleAssistant, ContentType: CTText, Text: "```go\n" + strings.Repeat("d", 400) + "\n```"},
	}
	bd := ComputeContextBreakdown(msgs, nil)
	if bd["text"] == 0 || bd["tool"] == 0 || bd["summaries"] == 0 || bd["code"] == 0 {
		t.Fatalf("breakdown = %v, want all four buckets populated", bd)
	}
}

// A collapsed context must re-anchor, or growth reads negative forever and the
// nudge never fires again.
func TestStampNudgeReanchorsAfterCollapse(t *testing.T) {
	cfg := DefaultConfig(200000)
	st := CreateInitialState("s", "/a")
	st.Nudge.LastPerMessageNudgeTokens = 150000
	st.Nudge.LastNudgeShownTokens = 148000
	st.Nudge.LastShownByTier = map[Tier]int{2: 140000}

	stampNudge(&st, cfg, 60000, NudgeDecision{}) // dropped by more than one growth step

	if st.Nudge.LastPerMessageNudgeTokens != 60000 {
		t.Fatalf("baseline = %d, want it re-anchored to 60000", st.Nudge.LastPerMessageNudgeTokens)
	}
	if st.Nudge.LastNudgeShownTokens != 0 || len(st.Nudge.LastShownByTier) != 0 {
		t.Fatalf("nudge stamps = %+v, want them cleared on re-anchor", st.Nudge)
	}
}

func TestStampNudgeInitialisesBaseline(t *testing.T) {
	cfg := DefaultConfig(200000)
	st := CreateInitialState("s", "/a")
	stampNudge(&st, cfg, 42000, NudgeDecision{})
	if st.Nudge.LastPerMessageNudgeTokens != 42000 {
		t.Fatalf("baseline = %d, want it initialised to the current size", st.Nudge.LastPerMessageNudgeTokens)
	}
}

func TestStampNudgeRecordsInjection(t *testing.T) {
	cfg := DefaultConfig(200000)
	st := CreateInitialState("s", "/a")
	st.Nudge.LastPerMessageNudgeTokens = 100000
	stampNudge(&st, cfg, 120000, NudgeDecision{ShouldInject: true, Tier: 2})
	if st.Nudge.LastNudgeShownTokens != 120000 {
		t.Fatalf("LastNudgeShownTokens = %d, want 120000", st.Nudge.LastNudgeShownTokens)
	}
	if st.Nudge.LastShownByTier[2] != 120000 {
		t.Fatalf("LastShownByTier[2] = %d, want 120000", st.Nudge.LastShownByTier[2])
	}
}
