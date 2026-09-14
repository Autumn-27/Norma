package noa

import "testing"

// The nudge pacing is expressed as a fraction of the window so that a small
// window gets proportionally tighter cadence than a large one. These tests check
// that the shipped defaults actually deliver that, because every pressure
// behaviour downstream — cadence, suppression release, tier spacing — is derived
// from this one number.

// GrowthRatio says the growth unit is 5% of the window. The clamp band is meant
// to keep that sane at the extremes, not to replace it.
func TestAdaptiveGrowthTracksTheWindow(t *testing.T) {
	small := DefaultConfig(40_000)
	large := DefaultConfig(400_000)

	a := resolveAdaptiveGrowth(small.ModelContextLimit, small.Nudge)
	b := resolveAdaptiveGrowth(large.ModelContextLimit, large.Nudge)

	if a == b {
		t.Fatalf("a %d-token window and a %d-token window both produce a growth unit of %d.\n"+
			"GrowthFloor (%d) and GrowthCap (%d) are equal, so the clamp band has zero width and "+
			"GrowthRatio (%.2f) can never affect the result — the 'adaptive' growth is a constant.",
			small.ModelContextLimit, large.ModelContextLimit, a,
			small.Nudge.GrowthFloor, small.Nudge.GrowthCap, small.Nudge.GrowthRatio)
	}
}

// Suppression stops nudging after MaxCompressAttempts and lifts once the context
// has grown by one cadence step. That step has to be small relative to the
// window: if it is not, suppression that engages near the ceiling cannot lift
// before the request overflows, and the system goes silent exactly when it has
// the most to say.
func TestSuppressionReleaseFitsInsideTheWindow(t *testing.T) {
	for _, window := range []int{40_000, 64_000, 128_000, 200_000} {
		cfg := DefaultConfig(window)
		floor := NudgeGrowthFloor(cfg)
		pct := float64(floor) * 100 / float64(window)

		// A quarter of the window is already generous: suppression normally
		// engages in the 75%+ pressure band, so anything above 25% cannot lift
		// before the window is full.
		if pct > 25 {
			t.Errorf("window %d: suppression lifts only after %d tokens of growth (%.0f%% of the "+
				"window). Suppression engages in the pressure band, so it cannot lift before "+
				"overflow — nudging stays off while the context runs past the limit.",
				window, floor, pct)
		}
	}
}

// The emergency band exists because at that pressure the next request may not
// fit at all. One nudge costs a couple of thousand tokens; an overflow costs the
// turn. So the failure ladder must not be able to silence the emergency voice.
func TestEmergencyPressureIsNudgedEvenWhenSuppressed(t *testing.T) {
	cfg := DefaultConfig(40_000)
	// 96% of the window: past EmergencyThresholdPct.
	tokens := int(float64(cfg.ModelContextLimit) * 0.96)

	d := DecideNudge(DecideNudgeInput{
		State:      CreateInitialState("s", ""),
		Config:     cfg,
		TokenCount: tokens,
		Recommendation: Recommendation{CompressibleRanges: []RangeInfo{{
			StartRef: "m00002", EndRef: "m00050", Count: 20,
			Tokens: 20_000, Chars: 80_000, ToolPct: 90, TextPct: 10,
		}}},
	})
	if !d.ShouldInject {
		t.Fatalf("no nudge at 96%% of the window even before any suppression; "+
			"the emergency band is not firing (reason: %s)", d.Reason)
	}
	if !d.Breakdown.Emergency {
		t.Fatalf("the decision at 96%% pressure is not flagged emergency: %+v", d.Breakdown)
	}

	// The host-side gate is what suppression acts on. It is a separate decision
	// from DecideNudge, and it currently has no knowledge of pressure at all —
	// see Session.nudgeAllowed, which branches only on attempts and growth.
	// This test documents the seam; the assertion belongs with the host, in
	// noaadapter.TestSuppressionSurvivesIntoTheEmergencyBand.
}
