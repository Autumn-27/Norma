package noa

import (
	"strings"
	"testing"
)

// recommendFixture builds a view with refs assigned.
func recommendFixture(t *testing.T, msgs []CoreMessage) ([]CoreMessage, CompressionState) {
	t.Helper()
	res := ProcessTurn(ProcessTurnInput{
		Messages: msgs, State: CreateInitialState("s", "/a"), Config: DefaultConfig(200000),
	})
	return res.Messages, res.State
}

func TestProtectedRefsCoverRecentMessages(t *testing.T) {
	var msgs []CoreMessage
	for i := range 10 {
		msgs = append(msgs, msg("m"+idFor(i), RoleUser, CTText, strings.Repeat("x", 100)))
	}
	view, st := recommendFixture(t, msgs)
	got := ComputeProtectedRefs(view, st, DefaultConfig(200000))
	// The last five are protected by count; the token budget reaches further
	// back because these messages are small.
	for i := 5; i < 10; i++ {
		ref := st.MessageRefs.ByRaw["m"+idFor(i)]
		if !got[ref] {
			t.Errorf("ref %s (message %d) is not protected", ref, i)
		}
	}
}

func TestProtectedRefsAlwaysIncludeLastUser(t *testing.T) {
	msgs := []CoreMessage{msg("u", RoleUser, CTText, "ask")}
	for i := range 20 {
		msgs = append(msgs, CoreMessage{
			ID: "a" + idFor(i), Role: RoleAssistant, ContentType: CTText,
			Text: strings.Repeat("x", 40000),
		})
	}
	view, st := recommendFixture(t, msgs)
	got := ComputeProtectedRefs(view, st, DefaultConfig(200000))
	// The live instruction must never be compressed away, however far back it is.
	if !got[st.MessageRefs.ByRaw["u"]] {
		t.Fatal("the most recent user message must be protected")
	}
}

// Bulky, re-obtainable output must not push real content out of the recency
// window — but it stays fully compressible.
func TestProtectedRefsSkipNeverPreserveTools(t *testing.T) {
	var msgs []CoreMessage
	for i := range 3 {
		msgs = append(msgs, msg("keep"+idFor(i), RoleUser, CTText, "real content"))
	}
	for i := range 10 {
		msgs = append(msgs, CoreMessage{
			ID: "read" + idFor(i), Role: RoleTool, ContentType: CTToolResult,
			ToolCallID: "t" + idFor(i), ToolName: "Read", Text: strings.Repeat("y", 200),
		})
	}
	view, st := recommendFixture(t, msgs)
	got := ComputeProtectedRefs(view, st, DefaultConfig(200000))

	// The Read results occupy no slots, so the three real messages stay in the
	// window despite ten tool results after them.
	for i := range 3 {
		ref := st.MessageRefs.ByRaw["keep"+idFor(i)]
		if !got[ref] {
			t.Errorf("real message %d lost its protection to bulky tool output", i)
		}
	}
}

func TestBuildCompressibleRangesGroupsAndBreaks(t *testing.T) {
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, strings.Repeat("x", 2000)),
		msg("b", RoleAssistant, CTText, strings.Repeat("x", 2000)),
		msg("c", RoleAssistant, CTText, strings.Repeat("x", 2000)),
		msg("d", RoleAssistant, CTText, strings.Repeat("x", 2000)),
		// A user message after three messages starts a new group: user turns are
		// natural seams, and one summary spanning several requests reads badly.
		msg("e", RoleUser, CTText, strings.Repeat("x", 2000)),
		msg("f", RoleAssistant, CTText, strings.Repeat("x", 2000)),
	}
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, strings.Repeat("t", 4000)))
	}
	view, st := recommendFixture(t, msgs)
	cfg := DefaultConfig(200000)
	protected := ComputeProtectedRefs(view, st, cfg)
	ranges, _ := BuildCompressibleRanges(view, st, cfg, protected, nil)

	if len(ranges) < 2 {
		t.Fatalf("got %d ranges, want the group to break at the user message: %+v", len(ranges), ranges)
	}
}

func TestBuildCompressibleRangesReportsProtectedTools(t *testing.T) {
	cfg := DefaultConfig(200000)
	cfg.ProtectedTools = []string{"Write"}
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, strings.Repeat("x", 2000)),
		{ID: "w", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t1", ToolName: "Write", Text: "{}"},
		{ID: "wr", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t1", ToolName: "Write", Text: "ok"},
	}
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, strings.Repeat("t", 4000)))
	}
	res := ProcessTurn(ProcessTurnInput{Messages: msgs, State: CreateInitialState("s", "/a"), Config: cfg})
	protected := ComputeProtectedRefs(res.Messages, res.State, cfg)
	_, protectedRuns := BuildCompressibleRanges(res.Messages, res.State, cfg, protected, nil)

	// Hard-protected tools are BLOCKED and carry no ref, so they cannot appear
	// in a reported range — they are invisible to the model by design.
	for _, p := range protectedRuns {
		if containsString(p.Tools, "Write") {
			t.Fatal("a hard-protected tool should never be addressable enough to report")
		}
	}
}

func TestMergeRangesToThreshold(t *testing.T) {
	ranges := []RangeInfo{
		{StartRef: "m00001", EndRef: "m00005", Count: 5, Tokens: 500, Chars: 2000, ToolPct: 100},
		{StartRef: "m00006", EndRef: "m00010", Count: 5, Tokens: 500, Chars: 2000, ToolPct: 0},
		{StartRef: "m00011", EndRef: "m00015", Count: 5, Tokens: 500, Chars: 2000, ToolPct: 50},
	}
	got := MergeRangesToThreshold(ranges, 5000)
	if len(got) != 1 {
		t.Fatalf("got %d ranges, want them merged to clear the 5000-char floor: %+v", len(got), got)
	}
	if got[0].StartRef != "m00001" || got[0].EndRef != "m00015" {
		t.Fatalf("merged span = %s–%s, want m00001–m00015", got[0].StartRef, got[0].EndRef)
	}
	// The merged tool percentage is a count-weighted average, not a plain one.
	if got[0].ToolPct != 50 {
		t.Fatalf("merged ToolPct = %d, want the weighted average 50", got[0].ToolPct)
	}
}

// A fragment dragged into a batch makes the whole batch fail the size gate, and
// models tend to resubmit the identical batch.
func TestViableRangesDropsFragments(t *testing.T) {
	got := ViableRanges([]RangeInfo{
		{StartRef: "m1", EndRef: "m2", Tokens: 50},
		{StartRef: "m3", EndRef: "m9", Tokens: 5000},
		{StartRef: "m10", EndRef: "m11", Tokens: ViableRangeMinTokens - 1},
		{StartRef: "m12", EndRef: "m19", Tokens: ViableRangeMinTokens},
	})
	if len(got) != 2 {
		t.Fatalf("got %d ranges, want the two at or above %d tokens: %+v", len(got), ViableRangeMinTokens, got)
	}
}

func TestBuildRecommendationEndToEnd(t *testing.T) {
	var msgs []CoreMessage
	msgs = append(msgs, msg("u0", RoleUser, CTText, "the task"))
	for i := range 6 {
		msgs = append(msgs, CoreMessage{
			ID: "tool" + idFor(i), Role: RoleTool, ContentType: CTToolResult,
			ToolCallID: "t" + idFor(i), ToolName: "Bash", Text: strings.Repeat("output ", 500),
		})
	}
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, strings.Repeat("t", 4000)))
	}
	view, st := recommendFixture(t, msgs)
	rec := BuildRecommendation(view, st, DefaultConfig(200000), nil)

	if rec.NothingToCompress {
		t.Fatal("six large tool results must yield something compressible")
	}
	if len(rec.CompressibleRanges) == 0 {
		t.Fatal("no compressible ranges reported")
	}
	toolHeavy := false
	for _, r := range rec.CompressibleRanges {
		if r.Tokens < ViableRangeMinTokens {
			t.Fatalf("a fragment survived the viability filter: %+v", r)
		}
		if r.ToolPct >= 50 {
			toolHeavy = true
		}
	}
	// The composition figures are what let the model pick the cheapest range;
	// the run of tool results must read as tool-heavy.
	if !toolHeavy {
		t.Fatalf("no range was reported as tool-heavy: %+v", rec.CompressibleRanges)
	}
}

func TestActiveBlockSpansSorted(t *testing.T) {
	st := CreateInitialState("s", "/a")
	st.Blocks = []CompressionBlock{
		{BlockID: "b2", Tier: 1, StartRef: "m00050", EndRef: "m00060", Active: true},
		{BlockID: "b1", Tier: 2, StartRef: "m00010", EndRef: "m00020", Active: true},
		{BlockID: "b3", Tier: 1, StartRef: "m00070", EndRef: "m00080", Active: false},
	}
	got := ActiveBlockSpans(st)
	if len(got) != 2 {
		t.Fatalf("got %d spans, want only the active blocks", len(got))
	}
	if got[0].BlockID != "b1" || got[1].BlockID != "b2" {
		t.Fatalf("spans = %+v, want them ordered by start ref", got)
	}
}
