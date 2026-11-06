package noa

import (
	"fmt"

	"strings"
)

// resolveAdaptiveGrowth is the growth unit the pacing is expressed in: a
// fraction of the window, clamped into a band.
func resolveAdaptiveGrowth(limit int, n NudgeConfig) int {
	if limit <= 0 {
		return n.GrowthFloor
	}
	return min(n.GrowthCap, max(n.GrowthFloor, roundHalfUp(float64(limit)*n.GrowthRatio)))
}

// growthFloorOf is the cadence gate: how much the context must grow before
// another nudge is warranted.
func growthFloorOf(adaptive int, n NudgeConfig) int {
	return max(n.MinGrowthFloor, int(n.MinGrowthRatio*float64(adaptive)))
}

// DecideNudgeInput is one turn's pressure picture.
type DecideNudgeInput struct {
	Messages   []CoreMessage
	State      CompressionState
	Config     Config
	TokenCount int
	// Recommendation is the compressible/protected split for this turn.
	Recommendation Recommendation
	CountTokens    TokenCountFn
	// TruncatedCount is how many tool results were mechanically shortened, so
	// an emergency nudge can tell the model not to re-run them.
	TruncatedCount int
}

// tierPending is one tier's backlog.
type tierPending struct {
	pending int
	count   int
	blocks  []CompressionBlock
}

// DecideNudge decides whether to ask the model to compress, and for what.
//
// Two bands, and they behave differently on purpose:
//
//   - Under pressure (usage past MaxContextLimitPct) there is no growth gate.
//     The context is nearly full; waiting for it to grow further is not a
//     strategy. Only a minimum-benefit floor applies, so the model is not asked
//     to spend a turn reclaiming nothing.
//   - Below pressure, the nudge is an efficiency suggestion and must not become
//     nagging: it fires only once the context has grown by a full cadence step
//     AND there is a worthwhile amount to reclaim.
func DecideNudge(in DecideNudgeInput) NudgeDecision {
	cfg := in.Config
	count := in.CountTokens
	if count == nil {
		count = DefaultCountTokens
	}
	limit := cfg.ModelContextLimit
	usage := 0.0
	if limit > 0 {
		usage = float64(in.TokenCount) / float64(limit)
	}

	adaptive := resolveAdaptiveGrowth(limit, cfg.Nudge)
	floor := growthFloorOf(adaptive, cfg.Nudge)
	minBen := cfg.minPressureBenefit()

	overLimit := usage >= cfg.Nudge.MaxContextLimitPct
	emergency := usage >= cfg.Nudge.EmergencyThresholdPct
	pressure := overLimit || emergency

	baseline := in.State.Nudge.LastPerMessageNudgeTokens
	growthRef := in.State.Nudge.LastNudgeShownTokens
	if growthRef == 0 {
		if baseline > 0 {
			growthRef = baseline
		} else {
			growthRef = in.TokenCount
		}
	}
	growthSince := in.TokenCount - growthRef

	tiers := computeTierPending(in, count)

	d := NudgeDecision{
		ContextUsage:       usage,
		CompressibleRanges: in.Recommendation.CompressibleRanges,
		ProtectedRanges:    in.Recommendation.ProtectedRanges,
		ActiveBlockSpans:   ActiveBlockSpans(in.State),
		ContextBreakdown:   ComputeContextBreakdown(in.Messages, count),
		TruncatedCount:     in.TruncatedCount,
		Breakdown: NudgeBreakdown{
			Usage: usage, Growth: growthSince, GrowthReference: growthRef,
			GrowthFloor: floor, AdaptiveGrowth: adaptive,
			OverLimit: overLimit, Emergency: emergency,
			MinPressureBenefit: minBen,
			PendingT1:          tiers[1].pending, PendingT2: tiers[2].pending, PendingT3: tiers[3].pending,
			CountT1: tiers[2].count, CountT2: tiers[3].count,
		},
	}

	if pressure {
		decidePressure(&d, tiers, cfg, minBen)
		return d
	}

	// First sight: a session that arrives already carrying a large backlog has
	// no baseline to have grown from, so the cadence gate would hold it back
	// indefinitely.
	firstSight := in.State.Nudge.LastNudgeShownTokens == 0 && baseline == 0 &&
		usage >= cfg.Nudge.MinContextLimitPct &&
		max(tiers[1].pending, max(tiers[2].pending, tiers[3].pending)) >= adaptive
	growthReady := firstSight || growthSince >= floor
	if !growthReady {
		d.Reason = fmt.Sprintf("below cadence: grew %d of %d since last nudge; %s",
			growthSince, floor, pendingSummary(tiers))
		return d
	}
	decideGrowth(&d, in, tiers, cfg, adaptive, floor)
	return d
}

// decidePressure picks a tier to compress under pressure.
//
// The default rule is "reclaim the most tokens this turn", which always names
// tier 1: it is raw content, while tier 2 is the ~10:1 summary of it. That rule
// therefore never consolidates, and the block count grows without bound — a
// session living in the pressure band ends up with dozens of tier-1 blocks and
// no tier 2 at all.
//
// The count triggers that exist to prevent exactly this lived only in
// decideGrowth, which the pressure path skips. They are applied here first: once
// enough lower-tier blocks exist, consolidating them is the job, even though it
// reclaims fewer tokens than compressing more raw content would.
func decidePressure(d *NudgeDecision, tiers map[Tier]tierPending, cfg Config, minBen int) {
	candidates := []Tier{1}
	if cfg.Tiers.Enabled {
		candidates = append(candidates, 2, 3)
	}

	// Consolidation first, by count — but only once the summaries it would fold
	// are themselves worth the turn. Reclaiming raw content is strictly better
	// per token, so taking consolidation too early starves the reclaim loop and
	// the view grows instead of shrinking.
	if cfg.Tiers.Enabled {
		for _, t := range []Tier{2, 3} {
			trigger := cfg.Tiers.Tier2Trigger
			if t == 3 {
				trigger = cfg.Tiers.Tier3Trigger
			}
			// The summaries a consolidation folds are ~10x smaller than the raw
			// content they replaced, so the full minimum-benefit floor would
			// never be cleared and the block count would grow unbounded. Half of
			// it, floored at 1% of the window, is the point where folding them
			// starts to matter.
			// Fold them once they outweigh what is left to reclaim: reclaiming raw
			// content is strictly better per token, so consolidation is only the
			// right move when there is less raw content left than summary to fold.
			if tiers[t].count >= trigger && len(tiers[t].blocks) > 0 && tiers[t].pending >= tiers[1].pending {
				d.ShouldInject = true
				d.Tier = t
				d.TierTargetBlocks = tiers[t].blocks
				d.Reason = fmt.Sprintf("pressure at %.0f%%: %d block(s) ready to consolidate into tier %d; %s",
					d.ContextUsage*100, tiers[t].count, t, pendingSummary(tiers))
				return
			}
		}
	}

	best, bestPending := Tier(0), -1
	for _, t := range candidates {
		if tiers[t].pending > bestPending {
			best, bestPending = t, tiers[t].pending
		}
	}
	if bestPending < minBen {
		// Under pressure but nothing worth reclaiming. Nudging anyway would burn
		// a turn and leave the context exactly as full.
		d.Reason = fmt.Sprintf("pressure at %.0f%% but best tier reclaims only %d (< %d minimum benefit); %s",
			d.ContextUsage*100, max(bestPending, 0), minBen, pendingSummary(tiers))
		return
	}
	d.ShouldInject = true
	d.Tier = best
	if best >= 2 {
		d.TierTargetBlocks = tiers[best].blocks
	}
	d.Reason = fmt.Sprintf("pressure at %.0f%%: tier %d reclaims %d; %s",
		d.ContextUsage*100, best, bestPending, pendingSummary(tiers))
}

// decideGrowth picks a tier in the efficiency band, in priority order.
func decideGrowth(d *NudgeDecision, in DecideNudgeInput, tiers map[Tier]tierPending,
	cfg Config, adaptive, floor int) {

	tier2Threshold := roundHalfUp(float64(adaptive) * cfg.Nudge.Tier2GrowthMultiplier)
	usageFloor := cfg.Nudge.MinContextLimitPct

	// Consolidation that has crossed its count trigger goes first. Raw content is
	// worth more per token, so "tier 1 has pending" would otherwise win every
	// time and the block count would grow without bound — a session can sit at a
	// quarter full with a hundred summaries and never consolidate.
	//
	// The count trigger is held back by a usage floor (with five blocks in a
	// nearly empty context, folding them buys nothing). A block count at twice
	// the trigger overrides that floor: then the thing being bounded IS the
	// block count, whatever the usage says.
	t2CountReady := tiers[2].count >= cfg.Tiers.Tier2Trigger &&
		(d.ContextUsage >= usageFloor || tiers[2].count >= 2*cfg.Tiers.Tier2Trigger)
	t3CountReady := tiers[3].count >= cfg.Tiers.Tier3Trigger &&
		(d.ContextUsage >= usageFloor || tiers[3].count >= 2*cfg.Tiers.Tier3Trigger)

	var blockedTier Tier
	if cfg.Tiers.Enabled {
		for _, ready := range []struct {
			ok   bool
			tier Tier
		}{{t2CountReady && len(tiers[2].blocks) > 0, 2}, {t3CountReady && len(tiers[3].blocks) > 0, 3}} {
			if !ready.ok {
				continue
			}
			if !tierCadenceReady(in.State, ready.tier, in.TokenCount, floor) {
				// Paced. Tier 1 may still have work this turn, so do not stop
				// here — but remember why, in case nothing else fires.
				blockedTier = ready.tier
				continue
			}
			d.ShouldInject = true
			d.Tier = ready.tier
			d.TierTargetBlocks = tiers[ready.tier].blocks
			d.Reason = fmt.Sprintf("growth: tier %d ready (%d block(s), %d pending); %s",
				ready.tier, tiers[ready.tier].count, tiers[ready.tier].pending, pendingSummary(tiers))
			return
		}
	}

	if tiers[1].pending >= adaptive {
		d.ShouldInject = true
		d.Tier = 1
		d.Reason = fmt.Sprintf("growth: tier 1 has %d pending (>= %d); %s", tiers[1].pending, adaptive, pendingSummary(tiers))
		return
	}
	if cfg.Tiers.Enabled {
		if tiers[2].pending >= tier2Threshold && tiers[2].pending > tiers[1].pending {
			if tierCadenceReady(in.State, 2, in.TokenCount, floor) {
				d.ShouldInject = true
				d.Tier = 2
				d.TierTargetBlocks = tiers[2].blocks
				d.Reason = fmt.Sprintf("growth: tier 2 ready (%d block(s), %d pending); %s",
					tiers[2].count, tiers[2].pending, pendingSummary(tiers))
				return
			}
			d.Reason = "blocked: T2 (cadence); " + pendingSummary(tiers)
			return
		}
		if tiers[3].pending >= tier2Threshold &&
			tiers[3].pending > tiers[2].pending && tiers[3].pending > tiers[1].pending {
			if tierCadenceReady(in.State, 3, in.TokenCount, floor) {
				d.ShouldInject = true
				d.Tier = 3
				d.TierTargetBlocks = tiers[3].blocks
				d.Reason = fmt.Sprintf("growth: tier 3 ready (%d block(s), %d pending); %s",
					tiers[3].count, tiers[3].pending, pendingSummary(tiers))
				return
			}
			d.Reason = "blocked: T3 (cadence); " + pendingSummary(tiers)
			return
		}
	}
	if blockedTier > 0 {
		d.Reason = fmt.Sprintf("blocked: T%d (cadence); %s", blockedTier, pendingSummary(tiers))
		return
	}
	d.Reason = "growth ready but no tier qualifies; " + pendingSummary(tiers)
}

// tierCadenceReady paces each tier independently, so a tier-2 nudge the model
// ignored does not reappear on every subsequent turn.
func tierCadenceReady(state CompressionState, tier Tier, tokenCount, floor int) bool {
	last := state.Nudge.LastShownByTier[tier]
	return last == 0 || tokenCount-last >= floor
}

// computeTierPending measures what each tier could reclaim.
//
// Tier 1 counts compressible raw content; tiers 2 and 3 count the summary text
// of the blocks they would absorb, because that is what consolidation actually
// removes from the context.
func computeTierPending(in DecideNudgeInput, count TokenCountFn) map[Tier]tierPending {
	out := map[Tier]tierPending{}

	t1 := 0
	for _, r := range in.Recommendation.CompressibleRanges {
		if r.Chars >= in.Config.Compress.MinCompressRange {
			t1 += r.Tokens
		}
	}
	out[1] = tierPending{pending: t1}

	for _, src := range []struct {
		tier Tier
		of   Tier
	}{{2, 1}, {3, 2}} {
		var tp tierPending
		for _, b := range in.State.Blocks {
			if b.Active && b.Tier == src.of {
				tp.pending += count(b.Summary)
				tp.count++
				tp.blocks = append(tp.blocks, b)
			}
		}
		out[src.tier] = tp
	}
	return out
}

func pendingSummary(tiers map[Tier]tierPending) string {
	return fmt.Sprintf("ready: T1(%s)/T2(%s)/T3(%s)",
		FormatTokens(tiers[1].pending), FormatTokens(tiers[2].pending), FormatTokens(tiers[3].pending))
}

// ComputeContextBreakdown buckets the context so the model can see where its
// tokens went, rather than only how many there are.
func ComputeContextBreakdown(msgs []CoreMessage, count TokenCountFn) map[string]int {
	if count == nil {
		count = DefaultCountTokens
	}
	out := map[string]int{}
	for _, m := range msgs {
		n := count(m.Text)
		switch {
		case strings.HasPrefix(m.Text, SummaryHeader):
			out["summaries"] += n
		case m.ContentType == CTToolCall || m.ContentType == CTToolResult:
			out["tool"] += n
		case strings.Contains(m.Text, "```"):
			out["code"] += n
		default:
			out["text"] += n
		}
	}
	return out
}

// nudgeInjectNode runs the decision and does the stamp bookkeeping.
func nudgeInjectNode() PipelineNode {
	return nodeFunc{
		name: "nudge-inject",
		run: func(io NodeIO, ctx PipelineContext) NodeIO {
			rec := BuildRecommendation(io.Messages, io.State, ctx.Config, ctx.CountTokens)
			d := DecideNudge(DecideNudgeInput{
				Messages: io.Messages, State: io.State, Config: ctx.Config,
				TokenCount: ctx.TokenCount, Recommendation: rec,
				CountTokens: ctx.CountTokens, TruncatedCount: io.TruncatedCount,
			})

			state := CloneState(io.State)
			stampNudge(&state, ctx.Config, ctx.TokenCount, d)
			io.State = state

			if d.ShouldInject {
				dec := d
				io.Nudge = &dec
			}
			return io
		},
	}
}

// stampNudge updates the pacing baselines. The order matters: a re-anchor must
// happen before the first-time initialisation, or a collapsed context would be
// initialised to its new size and then immediately re-anchored to it again.
func stampNudge(state *CompressionState, cfg Config, tokenCount int, d NudgeDecision) {
	baseline := state.Nudge.LastPerMessageNudgeTokens
	g := resolveAdaptiveGrowth(cfg.ModelContextLimit, cfg.Nudge)

	// The context collapsed by a full growth step — a compression landed, or the
	// host trimmed history. Comparing against the old baseline would report
	// negative growth forever, so re-anchor to the new scale.
	if baseline > 0 && tokenCount < baseline-g {
		state.Nudge.LastPerMessageNudgeTokens = tokenCount
		state.Nudge.LastNudgeShownTokens = 0
		state.Nudge.LastShownByTier = map[Tier]int{}
	}
	if state.Nudge.LastPerMessageNudgeTokens == 0 {
		state.Nudge.LastPerMessageNudgeTokens = tokenCount
	}
	if d.ShouldInject {
		state.Nudge.LastNudgeShownTokens = tokenCount
		if d.Tier != 0 {
			state.Nudge.LastShownByTier[d.Tier] = tokenCount
		}
	}
}

// NudgeGrowthFloor is the cadence step a host uses to decide when a suppressed
// nudge may resume.
//
// It is the pacing unit, clamped to a tenth of the window. The two jobs pull in
// opposite directions and cannot share one number: the cadence gate wants to be
// large (do not nag), while the release wants to be small — suppression engages
// in the pressure band, so a step that is a big share of the window can never
// complete before the request overflows, and nudging stays silent exactly when
// it matters most. The clamp is what makes "resume after a failure" always
// reachable.
func NudgeGrowthFloor(cfg Config) int {
	floor := growthFloorOf(resolveAdaptiveGrowth(cfg.ModelContextLimit, cfg.Nudge), cfg.Nudge)
	if cap := cfg.ModelContextLimit / 10; cap > 0 && floor > cap {
		return cap
	}
	return floor
}
