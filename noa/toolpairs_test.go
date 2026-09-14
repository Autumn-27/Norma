package noa

import (
	"strings"
	"testing"
)

func call(id, callID, tool string) CoreMessage {
	return CoreMessage{ID: id, Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: callID, ToolName: tool}
}
func result(id, callID, tool string) CoreMessage {
	return CoreMessage{ID: id, Role: RoleTool, ContentType: CTToolResult, ToolCallID: callID, ToolName: tool}
}
func think(id string) CoreMessage {
	return CoreMessage{ID: id, Role: RoleAssistant, ContentType: CTReasoning, Text: "reasoning " + id}
}

func TestAdjustBoundariesForToolPairsPullsResultForward(t *testing.T) {
	msgs := []CoreMessage{
		msg("u", RoleUser, CTText, "q"),
		call("c1", "t1", "Read"),
		result("r1", "t1", "Read"),
		msg("after", RoleUser, CTText, "later"),
	}
	// The range takes the call but not its result.
	s, e := AdjustBoundariesForToolPairs(1, 1, msgs, ToolPairMaxScan)
	if s != 1 || e != 2 {
		t.Fatalf("adjusted = %d..%d, want 1..2 — the result must come along", s, e)
	}
}

func TestAdjustBoundariesForToolPairsPullsCallBackward(t *testing.T) {
	msgs := []CoreMessage{
		call("c1", "t1", "Read"),
		result("r1", "t1", "Read"),
	}
	s, e := AdjustBoundariesForToolPairs(1, 1, msgs, ToolPairMaxScan)
	if s != 0 || e != 1 {
		t.Fatalf("adjusted = %d..%d, want 0..1 — the call must come along", s, e)
	}
}

// The walk stops at the first message that does not belong: the pair is
// contiguous, so a gap means we are past it.
func TestAdjustBoundariesForToolPairsStopsAtGap(t *testing.T) {
	msgs := []CoreMessage{
		call("c1", "t1", "Read"),
		msg("unrelated", RoleUser, CTText, "interruption"),
		result("r1", "t1", "Read"),
	}
	s, e := AdjustBoundariesForToolPairs(0, 0, msgs, ToolPairMaxScan)
	if s != 0 || e != 0 {
		t.Fatalf("adjusted = %d..%d, want 0..0 — the walk stops at the unrelated message", s, e)
	}
}

func TestAdjustBoundariesForToolPairsRespectsMaxScan(t *testing.T) {
	msgs := []CoreMessage{call("c1", "t1", "Read")}
	for i := range 30 {
		msgs = append(msgs, result("r"+idFor(i), "t1", "Read"))
	}
	s, e := AdjustBoundariesForToolPairs(0, 0, msgs, 5)
	if s != 0 || e != 5 {
		t.Fatalf("adjusted = %d..%d, want the scan capped at 5", s, e)
	}
}

// Compress is hard-protected and never part of a compressible range, so its
// exchange must not drag the boundary around.
func TestAdjustBoundariesForToolPairsIgnoresCompress(t *testing.T) {
	msgs := []CoreMessage{
		call("c1", "t1", CompressToolName),
		result("r1", "t1", CompressToolName),
	}
	s, e := AdjustBoundariesForToolPairs(0, 0, msgs, ToolPairMaxScan)
	if s != 0 || e != 0 {
		t.Fatalf("adjusted = %d..%d, want 0..0 — a Compress exchange must not widen the range", s, e)
	}
}

func TestAdjustBoundariesForReasoningPairsPullsBurst(t *testing.T) {
	msgs := []CoreMessage{
		think("t1"),
		call("c1", "x1", "Read"),
		msg("after", RoleUser, CTText, "later"),
	}
	s, e := AdjustBoundariesForReasoningPairs(0, 0, msgs)
	if s != 0 || e != 1 {
		t.Fatalf("adjusted = %d..%d, want the assistant act pulled in", s, e)
	}
}

func TestAdjustBoundariesForReasoningPairsPullsRunBackward(t *testing.T) {
	msgs := []CoreMessage{
		think("t1"),
		think("t2"),
		call("c1", "x1", "Read"),
	}
	s, e := AdjustBoundariesForReasoningPairs(2, 2, msgs)
	if s != 0 || e != 2 {
		t.Fatalf("adjusted = %d..%d, want the whole reasoning run pulled in", s, e)
	}
}

func TestApplyPairBoundaryAdjustmentsCombines(t *testing.T) {
	msgs := []CoreMessage{
		msg("u", RoleUser, CTText, "q"),
		think("t1"),
		call("c1", "x1", "Read"),
		result("r1", "x1", "Read"),
		msg("after", RoleUser, CTText, "later"),
	}
	// Starting from the call alone, both widenings must apply.
	s, e := ApplyPairBoundaryAdjustments(2, 2, msgs)
	if s != 1 || e != 3 {
		t.Fatalf("adjusted = %d..%d, want 1..3 — reasoning before, result after", s, e)
	}
}

func TestComputeTurnGroups(t *testing.T) {
	msgs := []CoreMessage{
		msg("u", RoleUser, CTText, "q"),
		think("t1"),
		call("c1", "x1", "Read"),
		result("r1", "x1", "Read"),
		msg("u2", RoleUser, CTText, "next"),
		think("t2"),
		msg("a2", RoleAssistant, CTText, "answer"),
	}
	groups := ComputeTurnGroups(msgs)
	if len(groups) != 2 {
		t.Fatalf("got %d turn groups, want 2", len(groups))
	}
	if len(groups[0].ReasoningIdx) != 1 || len(groups[0].ActIdx) != 1 || len(groups[0].ResultIdx) != 1 {
		t.Fatalf("group 0 = %+v, want one of each", groups[0])
	}
	if len(groups[1].ReasoningIdx) != 1 || len(groups[1].ActIdx) != 1 || len(groups[1].ResultIdx) != 0 {
		t.Fatalf("group 1 = %+v, want reasoning + act with no result", groups[1])
	}
}

// A visible tool call whose reasoning run was compressed away makes strict
// providers reject the rebuilt request, so the whole turn is withdrawn.
func TestWithdrawSplitTurns(t *testing.T) {
	msgs := []CoreMessage{
		think("t1"),
		call("c1", "x1", "Read"),
		result("r1", "x1", "Read"),
	}
	folded := map[string]bool{"t1": true} // only the reasoning is being compressed
	kept, withdrawn := WithdrawSplitTurns(msgs, folded)
	if len(kept) != 0 {
		t.Fatalf("kept = %v, want the split turn withdrawn entirely", kept)
	}
	if len(withdrawn) != 1 || withdrawn[0] != "t1" {
		t.Fatalf("withdrawn = %v, want [t1]", withdrawn)
	}
}

func TestWithdrawSplitTurnsKeepsWholeTurn(t *testing.T) {
	msgs := []CoreMessage{
		think("t1"),
		call("c1", "x1", "Read"),
		result("r1", "x1", "Read"),
	}
	// The whole turn is being compressed — nothing is left dangling.
	folded := map[string]bool{"t1": true, "c1": true, "r1": true}
	kept, withdrawn := WithdrawSplitTurns(msgs, folded)
	if len(kept) != 3 {
		t.Fatalf("kept = %v, want the intact turn preserved", kept)
	}
	if len(withdrawn) != 0 {
		t.Fatalf("withdrawn = %v, want none", withdrawn)
	}
}

func TestWithdrawSplitTurnsIgnoresTurnsWithoutReasoning(t *testing.T) {
	msgs := []CoreMessage{
		call("c1", "x1", "Read"),
		result("r1", "x1", "Read"),
	}
	folded := map[string]bool{"r1": true}
	kept, withdrawn := WithdrawSplitTurns(msgs, folded)
	if len(kept) != 1 || len(withdrawn) != 0 {
		t.Fatalf("kept = %v withdrawn = %v, want no withdrawal — there is no reasoning to lose", kept, withdrawn)
	}
}

// End to end: a range that would split a turn compresses less rather than
// producing a request the provider will reject.
func TestApplyCompressionWithdrawsSplitTurn(t *testing.T) {
	body := strings.Repeat("x", 3000)
	msgs := []CoreMessage{
		msg("u0", RoleUser, CTText, "the task"),
		{ID: "think1", Role: RoleAssistant, ContentType: CTReasoning, Text: body},
		call("callA", "tA", "Read"),
		result("resA", "tA", "Read"),
		msg("filler", RoleAssistant, CTText, body),
	}
	tailBody := strings.Repeat("t", 4000)
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, tailBody))
	}
	st := CreateInitialState("s", "/archive")
	pt := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})

	// m00002 is the reasoning; compressing it alone would orphan callA.
	// Pair adjustment should widen the range so the turn stays whole.
	res := ApplyCompression(ApplyInput{
		Ranges:   []CompressRange{{StartRef: "m00002", EndRef: "m00002", Summary: longSummary}},
		Messages: pt.Messages, State: pt.State, Config: DefaultConfig(200000),
		CallID: "toolu_1", Archiver: newMemArchiver(),
	})
	if len(res.BlocksCreated) == 1 {
		covered := map[string]bool{}
		for _, id := range res.BlocksCreated[0].EffectiveMessageIDs {
			covered[id] = true
		}
		if covered["think1"] && !covered["callA"] {
			t.Fatal("the reasoning was compressed while its tool call stayed visible — the provider would reject the rebuilt request")
		}
		return
	}
	// Refusing outright is also correct.
	if len(res.Errors) == 0 {
		t.Fatalf("neither a block nor an error: %+v", res)
	}
}
