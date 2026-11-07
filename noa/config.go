package noa

import "fmt"

// TierConfig controls multi-tier consolidation.
type TierConfig struct {
	Enabled bool
	// MaxTier is the terminal level, 2 or 3 (default 3). It is not an extension
	// point — the tier rule prompts are hand-written per tier and stop at 3.
	// It exists as a TEST lever: reaching the terminal-tier guard at MaxTier=3
	// needs ~100 compressions of synthetic conversation, at MaxTier=2 about ten.
	MaxTier Tier
	// Tier2Trigger is how many active tier-1 blocks trigger a tier-2 nudge.
	Tier2Trigger int
	// Tier3Trigger is how many active tier-2 blocks trigger a tier-3 nudge.
	Tier3Trigger int
}

// NudgeConfig paces how often the model is asked to compress.
type NudgeConfig struct {
	// MaxContextLimitPct is the bottom of the pressure band.
	MaxContextLimitPct float64
	// MinContextLimitPct is both the first-sight floor and the usage floor for
	// count-based tier triggers.
	MinContextLimitPct float64
	// EmergencyThresholdPct is where the voice turns urgent.
	EmergencyThresholdPct float64

	// GrowthRatio/Floor/Cap derive the adaptive growth unit from the window.
	GrowthRatio float64
	GrowthFloor int
	GrowthCap   int

	// MinGrowthFloor and MinGrowthRatio derive the cadence gate: how much the
	// context must grow before another nudge is warranted.
	MinGrowthFloor int
	MinGrowthRatio float64

	// Tier2GrowthMultiplier scales the adaptive growth into the pending-token
	// threshold for tier 2 and 3.
	Tier2GrowthMultiplier float64

	// MinPressureBenefitTokens is the minimum reclaimable amount worth spending a
	// turn on while under pressure. nil means max(5000, round(limit*0.01)).
	MinPressureBenefitTokens *int
}

// TruncateConfig controls the mechanical last resort.
type TruncateConfig struct {
	Threshold float64
}

// CompressConfig bounds what the model may submit.
type CompressConfig struct {
	// MinCompressRange is the per-batch character floor, applied atomically:
	// a batch below it is rejected whole. Block-boundary ranges are exempt —
	// tier 2/3 compress summaries, which are short by construction.
	MinCompressRange int
	MinSummaryLength int
	MaxSummaryLength int
}

// Config is the full knob set. DefaultConfig supplies values converged on
// upstream over months of production use; changing one is rarely a good idea.
type Config struct {
	ModelContextLimit int

	Tiers    TierConfig
	Nudge    NudgeConfig
	Truncate TruncateConfig
	Compress CompressConfig

	// ProtectedTools are hard-excluded from every compression range.
	ProtectedTools []string
	// IsToolProtected, when set, extends ProtectedTools with a host predicate.
	IsToolProtected func(toolName string) bool

	// PreserveRecentMessages and PreserveRecentTokens define the soft protected
	// zone at the tail of the conversation.
	PreserveRecentMessages int
	PreserveRecentTokens   int

	// MaxCompressAttempts is the consecutive failure/ignore limit before nudging
	// is suppressed. Suppression lifts once the context grows by growthFloor.
	MaxCompressAttempts int
}

// DefaultConfig returns the tuned defaults for a given context window.
func DefaultConfig(modelContextLimit int) Config {
	return Config{
		ModelContextLimit: modelContextLimit,
		Tiers: TierConfig{
			Enabled:      true,
			MaxTier:      3,
			Tier2Trigger: 5,
			Tier3Trigger: 10,
		},
		Nudge: NudgeConfig{
			MaxContextLimitPct:    0.75,
			MinContextLimitPct:    0.45,
			EmergencyThresholdPct: 0.95,
			// The band must have width, or GrowthRatio can never move the
			// result and "adaptive" growth is a constant: a 40K window then paces
			// itself exactly like a 400K one, and the small one overflows before
			// its cadence gate ever opens.
			GrowthRatio:           0.05,
			GrowthFloor:           1_000,
			GrowthCap:             20_000,
			MinGrowthFloor:        10_000,
			MinGrowthRatio:        0.45,
			Tier2GrowthMultiplier: 1.5,
		},
		Truncate: TruncateConfig{Threshold: 0.95},
		Compress: CompressConfig{
			MinCompressRange: 5_000,
			MinSummaryLength: 50,
			MaxSummaryLength: 20_000,
		},
		PreserveRecentMessages: 5,
		PreserveRecentTokens:   5_000,
		MaxCompressAttempts:    3,
	}
}

// ValidateConfig returns warnings, not errors: a questionable knob degrades
// behaviour but must never take down a session. Callers log and continue.
func ValidateConfig(c Config) []string {
	var w []string
	add := func(format string, args ...any) { w = append(w, fmt.Sprintf(format, args...)) }

	if c.ModelContextLimit <= 0 {
		add("ModelContextLimit must be positive, got %d", c.ModelContextLimit)
	}
	if c.Nudge.MinContextLimitPct > c.Nudge.MaxContextLimitPct {
		add("Nudge.MinContextLimitPct (%g) must not exceed MaxContextLimitPct (%g)",
			c.Nudge.MinContextLimitPct, c.Nudge.MaxContextLimitPct)
	}
	if c.Nudge.MaxContextLimitPct > c.Nudge.EmergencyThresholdPct {
		add("Nudge.MaxContextLimitPct (%g) must not exceed EmergencyThresholdPct (%g)",
			c.Nudge.MaxContextLimitPct, c.Nudge.EmergencyThresholdPct)
	}
	if c.Truncate.Threshold <= 0 || c.Truncate.Threshold > 1 {
		add("Truncate.Threshold must be in (0, 1], got %g", c.Truncate.Threshold)
	}
	if c.Tiers.Tier2Trigger < 1 {
		add("Tiers.Tier2Trigger must be >= 1, got %d", c.Tiers.Tier2Trigger)
	}
	if c.Tiers.Tier3Trigger <= c.Tiers.Tier2Trigger {
		add("Tiers.Tier3Trigger (%d) must exceed Tier2Trigger (%d)",
			c.Tiers.Tier3Trigger, c.Tiers.Tier2Trigger)
	}
	if c.Tiers.MaxTier != 2 && c.Tiers.MaxTier != 3 {
		add("Tiers.MaxTier must be 2 or 3, got %d — the tier rule prompts only cover up to 3",
			c.Tiers.MaxTier)
	}
	if c.Compress.MinSummaryLength >= c.Compress.MaxSummaryLength {
		add("Compress.MinSummaryLength (%d) must be below MaxSummaryLength (%d)",
			c.Compress.MinSummaryLength, c.Compress.MaxSummaryLength)
	}
	if c.Nudge.MinPressureBenefitTokens != nil && *c.Nudge.MinPressureBenefitTokens < 0 {
		add("Nudge.MinPressureBenefitTokens must be >= 0, got %d", *c.Nudge.MinPressureBenefitTokens)
	}
	if c.MaxCompressAttempts < 1 {
		add("MaxCompressAttempts must be >= 1, got %d", c.MaxCompressAttempts)
	}
	return w
}

// preserveRecentTokens resolves the tail protection in tokens, capped so a
// small window still has room left to compress.
//
// A flat 5000 protects 83% of a 6k window, and compression can then never run
// at all — the whole point of the floor is "leave the newest work alone", not
// "leave everything alone". Large windows are unaffected: 5000 stays well under
// a quarter of them.
func (c Config) preserveRecentTokens() int {
	n := c.PreserveRecentTokens
	if c.ModelContextLimit <= 0 {
		return n
	}
	if lim := max(c.ModelContextLimit/4, 500); n > lim {
		return lim
	}
	return n
}

// minCompressRangeChars resolves the per-batch floor in characters (~4 per
// token), capped at a twentieth of the window.
//
// The floor exists to reject a trivial range, not to demand a fifth of the
// context. On a 6k window the flat 5000 made every compressible range too small
// to qualify; above ~62k it is already under a twentieth, so nothing changes.
func (c Config) minCompressRangeChars() int {
	n := c.Compress.MinCompressRange
	if c.ModelContextLimit <= 0 {
		return n
	}
	if lim := max(c.ModelContextLimit*4/20, 800); n > lim {
		return lim
	}
	return n
}

// minPressureBenefit resolves the configured floor or its default.
// minPressureBenefit resolves the smallest reclaim it is worth interrupting for.
//
// The 5000-token default only scales above a 500k window — the max() means it is
// a constant everywhere else. On a 6k window that demands a single reclaimable
// range worth 83% of the context, so the pressure nudge never fires and nothing
// is ever compressed. Cap it at a quarter of the window; the default is already
// under that for any window above 20k, so nothing else changes.
func (c Config) minPressureBenefit() int {
	n := max(5000, roundHalfUp(float64(c.ModelContextLimit)*0.01))
	if c.Nudge.MinPressureBenefitTokens != nil {
		return *c.Nudge.MinPressureBenefitTokens
	}
	if c.ModelContextLimit <= 0 {
		return n
	}
	return min(n, max(c.ModelContextLimit/4, 800))
}
