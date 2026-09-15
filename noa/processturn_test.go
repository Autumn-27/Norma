package noa

import (
	"reflect"
	"strings"
	"testing"
)

func TestProcessTurnAssignsRefsAndLeavesViewIntact(t *testing.T) {
	in := ProcessTurnInput{
		Messages: []CoreMessage{
			msg("a", RoleUser, CTText, "task"),
			msg("b", RoleAssistant, CTText, "reply"),
		},
		State:      CreateInitialState("s", "/tmp"),
		Config:     DefaultConfig(200000),
		TokenCount: 1000,
	}
	got := ProcessTurn(in)
	if len(got.Messages) != 2 {
		t.Fatalf("Messages = %v, want the view unchanged with no blocks", ids(got.Messages))
	}
	if got.State.MessageRefs.ByRaw["a"] != "m00001" || got.State.MessageRefs.ByRaw["b"] != "m00002" {
		t.Fatalf("ByRaw = %v, want sequential refs", got.State.MessageRefs.ByRaw)
	}
}

// ProcessTurn must not touch its inputs: the host keeps the originals and
// re-projects from them every turn.
func TestProcessTurnDoesNotMutateInput(t *testing.T) {
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, "task"),
		msg("b", RoleAssistant, CTText, "covered"),
	}
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "b")}

	_ = ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})

	if len(msgs) != 2 || msgs[1].ID != "b" {
		t.Fatalf("input messages were mutated: %v", ids(msgs))
	}
	if len(st.MessageRefs.ByRaw) != 0 {
		t.Fatalf("input state ref map was mutated: %v", st.MessageRefs.ByRaw)
	}
}

// The steady state: a compressed range renders as its summary, and re-running
// the projection on the same input produces byte-identical output. This is what
// keeps the prefix cache alive between turns.
func TestProcessTurnIsIdempotent(t *testing.T) {
	msgs := []CoreMessage{
		msg("m1", RoleUser, CTText, "the task"),
		msg("m2", RoleAssistant, CTText, "work a"),
		msg("m3", RoleAssistant, CTText, "work b"),
		msg("m4", RoleUser, CTText, "recent"),
	}
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{{
		BlockID: "b1", Tier: 1, Summary: "did work", Active: true,
		DirectMessageIDs:    []string{"m2", "m3"},
		EffectiveMessageIDs: []string{"m2", "m3"},
		StartRef:            "m00002", EndRef: "m00003",
		ArchivePath: "/tmp/s/tier1/b1.md",
	}}
	cfg := DefaultConfig(200000)

	first := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: cfg})
	second := ProcessTurn(ProcessTurnInput{Messages: msgs, State: first.State, Config: cfg})

	if !reflect.DeepEqual(first.Messages, second.Messages) {
		t.Fatalf("projection is not idempotent:\nfirst  = %v\nsecond = %v",
			ids(first.Messages), ids(second.Messages))
	}
	want := []string{"m1", SummaryMessageID("b1"), "m4"}
	if strings.Join(ids(first.Messages), ",") != strings.Join(want, ",") {
		t.Fatalf("view = %v, want %v", ids(first.Messages), want)
	}
}

// A summary already in view must be re-anchored in place, not duplicated.
func TestProcessTurnDoesNotDuplicateSummaries(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}
	msgs := []CoreMessage{
		msg("first", RoleUser, CTText, "task"),
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader + "\nold"},
	}
	got := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})
	n := 0
	for _, m := range got.Messages {
		if m.ID == SummaryMessageID("b1") {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("summary rendered %d times, want 1: %v", n, ids(got.Messages))
	}
}

// Synthetic messages must never consume ref numbers, or the numbering would
// shift every time a block is created or a nudge fires.
func TestProcessTurnSyntheticMessagesGetNoRefs(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}
	msgs := []CoreMessage{
		msg("real1", RoleUser, CTText, "task"),
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
		{ID: NudgeMessageID, Role: RoleUser, ContentType: CTText, Text: "please compress"},
		msg("real2", RoleUser, CTText, "more"),
	}
	got := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})
	refs := got.State.MessageRefs.ByRaw
	if len(refs) != 2 {
		t.Fatalf("ByRaw = %v, want refs only for the two real messages", refs)
	}
	if refs["real1"] != "m00001" || refs["real2"] != "m00002" {
		t.Fatalf("ByRaw = %v, want contiguous refs unaffected by synthetic messages", refs)
	}
}

// The Compress call is hard-protected: BLOCKED, and never addressable.
func TestProcessTurnBlocksCompressTool(t *testing.T) {
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, "task"),
		{ID: "call", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t1", ToolName: CompressToolName},
		{ID: "res", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t1", ToolName: CompressToolName},
	}
	got := ProcessTurn(ProcessTurnInput{Messages: msgs, State: CreateInitialState("s", "/tmp"), Config: DefaultConfig(200000)})
	refs := got.State.MessageRefs.ByRaw
	if refs["call"] != BlockedRef || refs["res"] != BlockedRef {
		t.Fatalf("ByRaw = %v, want the Compress exchange BLOCKED", refs)
	}
	if refs["a"] != "m00001" {
		t.Fatalf("ref for the real message = %q, want m00001 — BLOCKED consumes no number", refs["a"])
	}
}

// Pruning a range that splits a tool exchange must not leave an unpaired block
// behind; strict providers reject such a request.
func TestProcessTurnRepairsPairsAfterPrune(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	// The block covers only the result, orphaning the call.
	st.Blocks = []CompressionBlock{block("b1", 1, "res")}
	msgs := []CoreMessage{
		msg("u", RoleUser, CTText, "task"),
		{ID: "call", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t1", ToolName: "Read"},
		{ID: "res", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t1", ToolName: "Read"},
	}
	got := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})
	for _, m := range got.Messages {
		if m.ID == "call" {
			t.Fatalf("view = %v, want the orphaned tool call dropped", ids(got.Messages))
		}
	}
}

func TestProcessTurnHandlesNilMaps(t *testing.T) {
	// A state loaded from disk or built by hand may have nil maps.
	st := CompressionState{}
	got := ProcessTurn(ProcessTurnInput{
		Messages: []CoreMessage{msg("a", RoleUser, CTText, "x")},
		State:    st,
		Config:   DefaultConfig(200000),
	})
	if got.State.MessageRefs.ByRaw["a"] != "m00001" {
		t.Fatalf("ByRaw = %v, want the nil maps to have been initialised", got.State.MessageRefs.ByRaw)
	}
	if got.State.NextBlockID != 1 {
		t.Fatalf("NextBlockID = %d, want it normalised to 1", got.State.NextBlockID)
	}
}

func TestProcessTurnEmptyInput(t *testing.T) {
	got := ProcessTurn(ProcessTurnInput{State: CreateInitialState("s", "/tmp"), Config: DefaultConfig(200000)})
	if len(got.Messages) != 0 {
		t.Fatalf("Messages = %v, want empty", ids(got.Messages))
	}
	if got.Nudge != nil {
		t.Fatal("Nudge must be nil when nothing is injected")
	}
}
