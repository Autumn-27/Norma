package noa

import (
	"strings"
	"testing"
)

// tierFixture builds a view of block summaries with the given tiers, in order,
// plus a loose message between the second and third.
func tierFixture(tiers ...Tier) ([]CoreMessage, CompressionState) {
	st := CreateInitialState("s", "/archive")
	var msgs []CoreMessage
	for i, tr := range tiers {
		id := "b" + string(rune('1'+i))
		b := block(id, tr, "gone"+id)
		st.Blocks = append(st.Blocks, b)
		msgs = append(msgs, CoreMessage{
			ID: SummaryMessageID(id), Role: RoleUser, ContentType: CTText,
			Text: SummaryHeader + "\nsummary of " + id,
		})
	}
	st.NextBlockID = len(tiers) + 1
	return msgs, st
}

func resolveAll(msgs []CoreMessage, st CompressionState, start, end string) ResolvedRange {
	r, err := ResolveBoundaries(start, end, msgs, st)
	if err != nil {
		panic(err)
	}
	return r
}

func TestDecideTierPureMessageRange(t *testing.T) {
	d := DecideTier(ResolvedRange{Kind: BoundaryMessage}, CreateInitialState("s", "/a"), DefaultConfig(200000))
	if d.TargetTier != 1 || d.OutputTier != 1 {
		t.Fatalf("tier = %d->%d, want 1->1 for a pure message range", d.TargetTier, d.OutputTier)
	}
}

func TestDecideTierAllTierOne(t *testing.T) {
	msgs, st := tierFixture(1, 1, 1)
	d := DecideTier(resolveAll(msgs, st, "b1", "b3"), st, DefaultConfig(200000))
	if d.TargetTier != 1 || d.OutputTier != 2 {
		t.Fatalf("tier = %d->%d, want 1->2", d.TargetTier, d.OutputTier)
	}
	if len(d.ConsumedBlockIDs) != 3 {
		t.Fatalf("consumed = %v, want all three T1 blocks", d.ConsumedBlockIDs)
	}
	if len(d.SurvivingBlockIDs) != 0 {
		t.Fatalf("surviving = %v, want none", d.SurvivingBlockIDs)
	}
}

// ONE TIER AT A TIME. The documented example: compressing b1..b4 over
// [T1, T1, T2, T1] absorbs only the T1 blocks; the T2 block stays active.
func TestDecideTierMixedAbsorbsOnlyLowest(t *testing.T) {
	msgs, st := tierFixture(1, 1, 2, 1)
	d := DecideTier(resolveAll(msgs, st, "b1", "b4"), st, DefaultConfig(200000))

	if d.TargetTier != 1 {
		t.Fatalf("TargetTier = %d, want 1 (the lowest tier present)", d.TargetTier)
	}
	if d.OutputTier != 2 {
		t.Fatalf("OutputTier = %d, want 2", d.OutputTier)
	}
	if strings.Join(d.ConsumedBlockIDs, ",") != "b1,b2,b4" {
		t.Fatalf("consumed = %v, want [b1 b2 b4] — b3 is a tier above and must not be absorbed", d.ConsumedBlockIDs)
	}
	if strings.Join(d.SurvivingBlockIDs, ",") != "b3" {
		t.Fatalf("surviving = %v, want [b3]", d.SurvivingBlockIDs)
	}
}

func TestDecideTierAllTierTwoProducesTierThree(t *testing.T) {
	msgs, st := tierFixture(2, 2, 2)
	d := DecideTier(resolveAll(msgs, st, "b1", "b3"), st, DefaultConfig(200000))
	if d.TargetTier != 2 || d.OutputTier != 3 {
		t.Fatalf("tier = %d->%d, want 2->3", d.TargetTier, d.OutputTier)
	}
}

// Tier 3 is terminal: the output cannot climb past MaxTier.
func TestDecideTierCapsAtMaxTier(t *testing.T) {
	msgs, st := tierFixture(3, 3)
	d := DecideTier(resolveAll(msgs, st, "b1", "b2"), st, DefaultConfig(200000))
	if d.TargetTier != 3 || d.OutputTier != 3 {
		t.Fatalf("tier = %d->%d, want 3->3 — tier 3 is terminal", d.TargetTier, d.OutputTier)
	}
}

// MaxTier=2 is the supported test lever: it makes the terminal case reachable
// with two blocks instead of the ~100 compressions the default would need.
func TestDecideTierRespectsMaxTierTwo(t *testing.T) {
	cfg := DefaultConfig(200000)
	cfg.Tiers.MaxTier = 2
	msgs, st := tierFixture(2, 2)
	d := DecideTier(resolveAll(msgs, st, "b1", "b2"), st, cfg)
	if d.OutputTier != 2 {
		t.Fatalf("OutputTier = %d, want 2 under MaxTier=2", d.OutputTier)
	}
	if !IsTerminalRewrite(d, 0, cfg) {
		t.Fatal("absorbing terminal blocks with no new content must be flagged as a terminal rewrite")
	}
}

func TestIsTerminalRewrite(t *testing.T) {
	cfg := DefaultConfig(200000)
	term := TierDecision{TargetTier: 3, OutputTier: 3}
	if !IsTerminalRewrite(term, 0, cfg) {
		t.Fatal("T3 absorbing T3 with no new messages reclaims nothing and must be refused")
	}
	if IsTerminalRewrite(term, 5, cfg) {
		t.Fatal("a terminal range that also folds in new messages does reclaim something")
	}
	if IsTerminalRewrite(TierDecision{TargetTier: 1, OutputTier: 2}, 0, cfg) {
		t.Fatal("a non-terminal tier must not be flagged")
	}
}

// End to end: the surviving higher-tier block is reported so the model can see
// why it is still there.
func TestApplyCompressionReportsSurvivingHigherTier(t *testing.T) {
	st := CreateInitialState("s", "/archive")
	var msgs []CoreMessage
	msgs = append(msgs, msg("u0", RoleUser, CTText, "the task"))
	for i, tr := range []Tier{1, 1, 2, 1} {
		id := "b" + string(rune('1'+i))
		st.Blocks = append(st.Blocks, block(id, tr, "gone"+id))
		msgs = append(msgs, CoreMessage{
			ID: SummaryMessageID(id), Role: RoleUser, ContentType: CTText,
			Text: SummaryHeader + "\nsummary of " + id,
		})
	}
	st.NextBlockID = 5
	tailBody := strings.Repeat("t", 4000)
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, tailBody))
	}
	pt := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})

	res := ApplyCompression(ApplyInput{
		Ranges:   []CompressRange{{StartRef: "b1", EndRef: "b4", Summary: longSummary, Topic: "consolidated"}},
		Messages: pt.Messages, State: pt.State, Config: DefaultConfig(200000),
		CallID: "toolu_1", Archiver: newMemArchiver(),
	})
	if len(res.Errors) != 0 {
		t.Fatalf("errors = %v", res.Errors)
	}
	if len(res.BlocksCreated) != 1 {
		t.Fatalf("BlocksCreated = %d, want 1", len(res.BlocksCreated))
	}
	nb := res.BlocksCreated[0]
	if nb.Tier != 2 {
		t.Fatalf("new block tier = %d, want 2", nb.Tier)
	}
	if strings.Join(nb.DirectBlockIDs, ",") != "b1,b2,b4" {
		t.Fatalf("DirectBlockIDs = %v, want [b1 b2 b4]", nb.DirectBlockIDs)
	}
	if strings.Join(res.SurvivingBlockIDs, ",") != "b3" {
		t.Fatalf("SurvivingBlockIDs = %v, want [b3]", res.SurvivingBlockIDs)
	}
	if b3 := FindBlock(&res.State, "b3"); b3 == nil || !b3.Active {
		t.Fatal("b3 is a tier above the target and must stay active")
	}
	for _, id := range []string{"b1", "b2", "b4"} {
		if b := FindBlock(&res.State, id); b == nil || b.Active {
			t.Fatalf("%s should have been absorbed and deactivated", id)
		}
	}
}

// The terminal guard must fire end to end, not just in the predicate.
func TestApplyCompressionRefusesTerminalRewrite(t *testing.T) {
	cfg := DefaultConfig(200000)
	cfg.Tiers.MaxTier = 2

	st := CreateInitialState("s", "/archive")
	msgs := []CoreMessage{msg("u0", RoleUser, CTText, "the task")}
	for i, tr := range []Tier{2, 2} {
		id := "b" + string(rune('1'+i))
		st.Blocks = append(st.Blocks, block(id, tr, "gone"+id))
		msgs = append(msgs, CoreMessage{
			ID: SummaryMessageID(id), Role: RoleUser, ContentType: CTText,
			Text: SummaryHeader + "\nsummary of " + id,
		})
	}
	st.NextBlockID = 3
	tailBody := strings.Repeat("t", 4000)
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, tailBody))
	}
	pt := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: cfg})

	arch := newMemArchiver()
	res := ApplyCompression(ApplyInput{
		Ranges:   []CompressRange{{StartRef: "b1", EndRef: "b2", Summary: longSummary}},
		Messages: pt.Messages, State: pt.State, Config: cfg,
		CallID: "toolu_1", Archiver: arch,
	})
	if len(res.BlocksCreated) != 0 {
		t.Fatalf("BlocksCreated = %d, want 0 — a terminal rewrite reclaims nothing", len(res.BlocksCreated))
	}
	joined := strings.Join(res.Errors, "\n")
	if !strings.Contains(joined, "terminal layer") {
		t.Fatalf("errors = %v, want the terminal-tier refusal", res.Errors)
	}
	if arch.count() != 0 {
		t.Fatal("a refused compression must not write an archive")
	}
}
