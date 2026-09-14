package noa

import "testing"

// block builds a minimal active block covering the given message ids.
func block(id string, tier Tier, covered ...string) CompressionBlock {
	return CompressionBlock{
		BlockID:             id,
		Tier:                tier,
		Summary:             "summary of " + id,
		DirectMessageIDs:    append([]string(nil), covered...),
		EffectiveMessageIDs: append([]string(nil), covered...),
		Active:              true,
	}
}

func syncWith(msgs []CoreMessage, st CompressionState) CompressionState {
	io := syncBlocks(NodeIO{Messages: msgs, State: st}, PipelineContext{Config: DefaultConfig(200000)})
	return io.State
}

func TestSyncBlocksDeactivatesConsumed(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	b1 := block("b1", 1, "m1")
	b2 := block("b2", 2, "m1")
	b2.DirectBlockIDs = []string{"b1"}
	st.Blocks = []CompressionBlock{b1, b2}

	out := syncWith([]CoreMessage{msg("m1", RoleUser, CTText, "x")}, st)
	if FindBlock(&out, "b1").Active {
		t.Fatal("b1 was absorbed by b2 and must be inactive")
	}
	if !FindBlock(&out, "b2").Active {
		t.Fatal("b2 covers a present message and must stay active")
	}
}

func TestSyncBlocksStillPresentViaSummaryID(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}

	// The covered message is no longer in view, but its rendered summary is —
	// which is exactly the steady state after a compression.
	msgs := []CoreMessage{{
		ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText,
		Text: SummaryHeader + "\nsomething",
	}}
	out := syncWith(msgs, st)
	if !FindBlock(&out, "b1").Active {
		t.Fatal("a block whose summary is in view must stay active")
	}
}

func TestSyncBlocksDeactivatesWhenNothingPresent(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}
	out := syncWith([]CoreMessage{msg("other", RoleUser, CTText, "x")}, st)
	if FindBlock(&out, "b1").Active {
		t.Fatal("a block with neither covered messages nor its summary in view must go inactive")
	}
}

func TestSyncBlocksReactivates(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	b := block("b1", 1, "m1")
	b.Active = false
	st.Blocks = []CompressionBlock{b}
	out := syncWith([]CoreMessage{msg("m1", RoleUser, CTText, "x")}, st)
	if !FindBlock(&out, "b1").Active {
		t.Fatal("a block whose coverage is back in view must become active again")
	}
}

func TestSyncBlocksPrunesTokenSnapshot(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.MessageRefs.ByRaw = map[string]string{"live": "m00001", "dead": "m00002"}
	st.MessageRefs.ByRef = map[string]string{"m00001": "live", "m00002": "dead"}
	st.TokenSnapshot = map[string]int{"m00001": 10, "m00002": 20, "m00003": 30}
	st.Blocks = []CompressionBlock{block("b1", 1, "live")}

	out := syncWith([]CoreMessage{msg("live", RoleUser, CTText, "x")}, st)
	if len(out.TokenSnapshot) != 1 {
		t.Fatalf("TokenSnapshot = %v, want only the live ref", out.TokenSnapshot)
	}
	if out.TokenSnapshot["m00001"] != 10 {
		t.Fatalf("live snapshot entry = %d, want 10 — the frozen number must not move", out.TokenSnapshot["m00001"])
	}
}

// BLOCKED refs never enter the snapshot, so they must not confuse pruning.
func TestSyncBlocksSnapshotIgnoresBlocked(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.MessageRefs.ByRaw = map[string]string{"a": "m00001", "p": BlockedRef}
	st.MessageRefs.ByRef = map[string]string{"m00001": "a"}
	st.TokenSnapshot = map[string]int{"m00001": 5}
	st.Blocks = []CompressionBlock{block("b1", 1, "a")}

	msgs := []CoreMessage{msg("a", RoleUser, CTText, "x"), msg("p", RoleAssistant, CTToolCall, "y")}
	out := syncWith(msgs, st)
	if out.TokenSnapshot["m00001"] != 5 {
		t.Fatalf("TokenSnapshot = %v, want m00001 kept", out.TokenSnapshot)
	}
}

func TestSyncBlocksDoesNotMutateInput(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}
	before := st.Blocks[0].Active

	_ = syncWith([]CoreMessage{msg("other", RoleUser, CTText, "x")}, st)
	if st.Blocks[0].Active != before {
		t.Fatal("syncBlocks mutated the caller's state; it must work on a clone")
	}
}

func TestCoveredMessageIDsOnlyActive(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	active := block("b1", 1, "a", "b")
	inactive := block("b2", 1, "c")
	inactive.Active = false
	st.Blocks = []CompressionBlock{active, inactive}

	covered := CoveredMessageIDs(st)
	if !covered["a"] || !covered["b"] {
		t.Fatalf("covered = %v, want the active block's ids", covered)
	}
	if covered["c"] {
		t.Fatal("an inactive block's coverage must not be reported as covered")
	}
}

func TestCloneStateIsDeep(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "a")}
	st.MessageRefs.ByRaw["a"] = "m00001"
	st.TokenSnapshot["m00001"] = 7
	st.Nudge.LastShownByTier[2] = 123

	c := CloneState(st)
	c.Blocks[0].EffectiveMessageIDs[0] = "mutated"
	c.Blocks[0].Active = false
	c.MessageRefs.ByRaw["a"] = "m00099"
	c.TokenSnapshot["m00001"] = 99
	c.Nudge.LastShownByTier[2] = 999

	if st.Blocks[0].EffectiveMessageIDs[0] != "a" {
		t.Error("EffectiveMessageIDs slice is shared with the clone")
	}
	if !st.Blocks[0].Active {
		t.Error("block Active is shared with the clone")
	}
	if st.MessageRefs.ByRaw["a"] != "m00001" {
		t.Error("ByRaw map is shared with the clone")
	}
	if st.TokenSnapshot["m00001"] != 7 {
		t.Error("TokenSnapshot map is shared with the clone")
	}
	if st.Nudge.LastShownByTier[2] != 123 {
		t.Error("LastShownByTier map is shared with the clone")
	}
}
