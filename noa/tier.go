package noa

// TierDecision is the outcome of classifying one resolved range.
type TierDecision struct {
	// TargetTier is the layer being compressed: the LOWEST tier present in the
	// range. It decides which blocks are absorbed.
	TargetTier Tier
	// OutputTier is the new block's tier, one above the target and capped at
	// MaxTier.
	OutputTier Tier
	// ConsumedBlockIDs are the blocks this compression absorbs — only those at
	// TargetTier.
	ConsumedBlockIDs []string
	// SurvivingBlockIDs are blocks inside the range that are NOT absorbed
	// because they sit above TargetTier. They stay active, and the panel reports
	// them so the model understands why they are still there.
	SurvivingBlockIDs []string
}

// DecideTier classifies a resolved range.
//
// ONE TIER AT A TIME. A compression absorbs only the blocks at the lowest tier
// present in the range; anything higher stays active. Re-summarising an
// already-distilled tier-2 summary alongside raw tier-1 summaries would put the
// same content through three lossy passes, and detail disappears fast.
//
// The rule is also what makes tier 3 reachable at all: if a tier-2 block could
// be absorbed by a same-tier sibling, TargetTier would be pinned at 1 forever
// and OutputTier would never exceed 2.
func DecideTier(r ResolvedRange, state CompressionState, cfg Config) TierDecision {
	maxTier := cfg.Tiers.MaxTier
	if maxTier < 2 {
		maxTier = 3
	}

	d := TierDecision{TargetTier: 1, OutputTier: 1}
	if r.Kind != BoundaryBlock {
		// A pure message range always produces a tier-1 block.
		return d
	}

	lowest := Tier(0)
	for _, id := range r.NestedBlockIDs {
		b := FindBlock(&state, id)
		if b == nil || !b.Active {
			continue
		}
		if lowest == 0 || b.Tier < lowest {
			lowest = b.Tier
		}
	}
	if lowest > 0 {
		d.TargetTier = lowest
	}
	d.OutputTier = min(maxTier, d.TargetTier+1)

	for _, id := range r.NestedBlockIDs {
		b := FindBlock(&state, id)
		if b == nil || !b.Active {
			continue
		}
		if b.Tier == d.TargetTier {
			d.ConsumedBlockIDs = append(d.ConsumedBlockIDs, id)
		} else {
			d.SurvivingBlockIDs = append(d.SurvivingBlockIDs, id)
		}
	}
	return d
}

// IsTerminalRewrite reports a compression that would absorb terminal-tier blocks
// into another terminal-tier block without folding in any new message.
//
// That reclaims nothing — the same summaries go in and come out one pass lossier
// — and a model that tries it once will try it again. Checking TargetTier alone
// is sufficient: by DecideTier, TargetTier == MaxTier implies OutputTier ==
// MaxTier.
func IsTerminalRewrite(d TierDecision, directCount int, cfg Config) bool {
	maxTier := cfg.Tiers.MaxTier
	if maxTier < 2 {
		maxTier = 3
	}
	return d.TargetTier == maxTier && directCount == 0
}
