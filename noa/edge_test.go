package noa

import (
	"strings"
	"testing"
)

// --- Degenerate inputs ---------------------------------------------------

func TestProcessTurnSingleMessage(t *testing.T) {
	res := ProcessTurn(ProcessTurnInput{
		Messages: []CoreMessage{msg("only", RoleUser, CTText, "hi")},
		State:    CreateInitialState("s", "/a"), Config: DefaultConfig(200000),
	})
	if len(res.Messages) != 1 || res.State.MessageRefs.ByRaw["only"] != "m00001" {
		t.Fatalf("single message: %v / %v", ids(res.Messages), res.State.MessageRefs.ByRaw)
	}
}

func TestProcessTurnAllMessagesEmptyText(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "a", Role: RoleUser, ContentType: CTText},
		{ID: "b", Role: RoleAssistant, ContentType: CTText},
	}
	res := ProcessTurn(ProcessTurnInput{Messages: msgs, State: CreateInitialState("s", "/a"), Config: DefaultConfig(200000)})
	if len(res.Messages) != 2 {
		t.Fatalf("empty-text messages were dropped: %v", ids(res.Messages))
	}
}

func TestProcessTurnZeroContextLimit(t *testing.T) {
	cfg := DefaultConfig(0)
	res := ProcessTurn(ProcessTurnInput{
		Messages: []CoreMessage{msg("a", RoleUser, CTText, strings.Repeat("x", 100000))},
		State:    CreateInitialState("s", "/a"), Config: cfg, TokenCount: 25000,
	})
	// A zero limit means "unknown", not "everything is an emergency".
	if res.TruncatedCount != 0 {
		t.Fatalf("truncated %d messages with no configured window", res.TruncatedCount)
	}
	if res.Nudge != nil && res.Nudge.ShouldInject {
		t.Fatal("a nudge fired with no configured window")
	}
}

// A block whose coverage vanished must not leave a summary standing for
// nothing.
func TestPruneWithBlockCoveringNothing(t *testing.T) {
	st := CreateInitialState("s", "/a")
	st.Blocks = []CompressionBlock{block("b1", 1, "vanished")}
	res := ProcessTurn(ProcessTurnInput{
		Messages: []CoreMessage{msg("present", RoleUser, CTText, "here")},
		State:    st, Config: DefaultConfig(200000),
	})
	if FindBlock(&res.State, "b1").Active {
		t.Fatal("a block covering nothing stayed active")
	}
	for _, m := range res.Messages {
		if strings.HasPrefix(m.ID, SummaryIDPrefix) {
			t.Fatalf("an inactive block still rendered a summary: %v", ids(res.Messages))
		}
	}
}

// Every message covered means the view is just the summary plus the protected
// first user message.
func TestPruneEverythingCovered(t *testing.T) {
	st := CreateInitialState("s", "/a")
	st.Blocks = []CompressionBlock{block("b1", 1, "a", "b", "c")}
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, "first"),
		msg("b", RoleAssistant, CTText, "second"),
		msg("c", RoleAssistant, CTText, "third"),
	}
	got := pruneWith(msgs, st)
	if len(got) != 2 {
		t.Fatalf("got %v, want the summary plus the preserved first user message", ids(got))
	}
}

// --- Ref space -----------------------------------------------------------

func TestRefSpaceBoundaries(t *testing.T) {
	if IndexToRef(MinRefIndex) != "m00001" {
		t.Errorf("the lowest ref is %q", IndexToRef(MinRefIndex))
	}
	if IndexToRef(MaxRefIndex) != "m99999" {
		t.Errorf("the highest ref is %q", IndexToRef(MaxRefIndex))
	}
	if r := IndexToRef(MaxRefIndex + 1); r != "" {
		t.Errorf("IndexToRef past the ceiling = %q, want an empty string", r)
	}
	n, ok := RefToIndex("m99999")
	if !ok || n != MaxRefIndex {
		t.Errorf("RefToIndex(m99999) = %d,%v", n, ok)
	}
	if _, ok := RefToIndex("m100000"); ok {
		t.Error("a six-digit ref was accepted")
	}
}

// Running out of refs must degrade, not corrupt: earlier messages keep theirs.
func TestAssignRefsNearExhaustionKeepsExisting(t *testing.T) {
	existing := MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}}
	for i := MinRefIndex; i < MaxRefIndex; i++ {
		existing.ByRef[IndexToRef(i)] = "filler"
	}
	existing.ByRaw["old"] = "m00001"

	res := AssignRefs([]CoreMessage{
		msg("new1", RoleUser, CTText, "a"),
		msg("new2", RoleUser, CTText, "b"),
	}, AssignRefsOptions{Existing: existing, NextIndex: MaxRefIndex})

	if res.Map.ByRaw["old"] != "m00001" {
		t.Fatal("an existing ref was disturbed by exhaustion")
	}
	if res.NewlyAssigned != 1 {
		t.Fatalf("NewlyAssigned = %d, want exactly the one remaining slot", res.NewlyAssigned)
	}
}

// --- Block ids -----------------------------------------------------------

func TestAllocateBlockIDNormalisesCounter(t *testing.T) {
	st := CreateInitialState("s", "/a")
	st.NextBlockID = 0 // a hand-built or legacy state
	if id := AllocateBlockID(&st); id != "b1" {
		t.Fatalf("first id = %q, want b1", id)
	}
	if st.NextBlockID != 2 {
		t.Fatalf("NextBlockID = %d, want 2", st.NextBlockID)
	}
}

func TestBlockIDRoundTrip(t *testing.T) {
	for _, n := range []int{1, 9, 10, 99, 100, 999999999} {
		id := "b" + itoa(n)
		got, ok := ParseBlockID(id)
		if !ok || got != id {
			t.Errorf("ParseBlockID(%q) = %q,%v", id, got, ok)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// --- Summary bounds ------------------------------------------------------

// The bounds are inclusive at the minimum and at the maximum.
func TestSummaryLengthBoundaries(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	cfg := DefaultConfig(200000)

	exactMin := strings.Repeat("s", cfg.Compress.MinSummaryLength)
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: exactMin,
	})
	if len(res.Errors) != 0 {
		t.Fatalf("a summary of exactly the minimum was rejected: %v", res.Errors)
	}

	justUnder := strings.Repeat("s", cfg.Compress.MinSummaryLength-1)
	res = applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: justUnder,
	})
	if len(res.Errors) == 0 {
		t.Fatal("a summary one character under the minimum was accepted")
	}

	exactMax := strings.Repeat("s", cfg.Compress.MaxSummaryLength)
	res = applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: exactMax,
	})
	if len(res.Errors) != 0 {
		t.Fatalf("a summary of exactly the maximum was rejected: %v", res.Errors)
	}
}

// Length is counted in runes, so a CJK summary is not rejected for being three
// times its apparent size.
func TestSummaryLengthCountsRunes(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	cjk := strings.Repeat("认证方案设计与实现决策记录", 5) // 65 runes, 195 bytes
	if len([]rune(cjk)) < 50 || len(cjk) < 150 {
		t.Fatalf("fixture is wrong: %d runes, %d bytes", len([]rune(cjk)), len(cjk))
	}
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: cjk,
	})
	if len(res.Errors) != 0 {
		t.Fatalf("a CJK summary above the rune minimum was rejected: %v", res.Errors)
	}
}

// --- Range shapes --------------------------------------------------------

func TestCompressSingleMessageRange(t *testing.T) {
	msgs, st := fixture(t, 3, 30000) // one message alone clears the size gate
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00002", Summary: longSummary,
	})
	if len(res.Errors) != 0 || len(res.BlocksCreated) != 1 {
		t.Fatalf("a single-message range failed: %v", res.Errors)
	}
	if res.BlocksCreated[0].StartRef != res.BlocksCreated[0].EndRef {
		t.Fatalf("span = %s..%s, want both ends equal",
			res.BlocksCreated[0].StartRef, res.BlocksCreated[0].EndRef)
	}
}

func TestCompressBlockRefsWhenNoBlocksExist(t *testing.T) {
	msgs, st := fixture(t, 4, 2000)
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "b1", EndRef: "b2", Summary: longSummary,
	})
	if len(res.BlocksCreated) != 0 {
		t.Fatal("block refs resolved against a session with no blocks")
	}
	if len(res.Errors) == 0 {
		t.Fatal("no error for unresolvable block refs")
	}
}

// A range whose endpoints are the same block is a no-op consolidation, not a
// crash.
func TestCompressSameBlockBothEnds(t *testing.T) {
	st := CreateInitialState("s", "/a")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}
	st.NextBlockID = 2
	msgs := []CoreMessage{
		msg("u0", RoleUser, CTText, "the task"),
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader + "\nsummary"},
	}
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, strings.Repeat("t", 4000)))
	}
	pt := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})
	res := ApplyCompression(ApplyInput{
		Ranges:   []CompressRange{{StartRef: "b1", EndRef: "b1", Summary: longSummary}},
		Messages: pt.Messages, State: pt.State, Config: DefaultConfig(200000),
		CallID: "x", Archiver: newMemArchiver(),
	})
	// Either it promotes the block to tier 2, or it refuses; both are coherent.
	// What it must not do is produce a block covering nothing.
	for _, b := range res.BlocksCreated {
		if len(b.EffectiveMessageIDs) == 0 && len(b.DirectBlockIDs) == 0 {
			t.Fatalf("created a block covering nothing: %+v", b)
		}
	}
}

// --- Unicode and odd content --------------------------------------------

func TestCompressPreservesUnicodeInArchive(t *testing.T) {
	content := "错误：认证失败 🔐 at auth/jwt.go:42\n" + strings.Repeat("日志行内容。", 900)
	msgs := []CoreMessage{
		msg("u0", RoleUser, CTText, "the task"),
		msg("cjk", RoleAssistant, CTText, content),
	}
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, strings.Repeat("t", 4000)))
	}
	st := CreateInitialState("s", "/a")
	pt := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})

	arch := newMemArchiver()
	res := ApplyCompression(ApplyInput{
		Ranges:   []CompressRange{{StartRef: "m00002", EndRef: "m00002", Summary: longSummary}},
		Messages: pt.Messages, State: pt.State, Config: DefaultConfig(200000),
		CallID: "x", Archiver: arch,
	})
	if len(res.BlocksCreated) != 1 {
		t.Fatalf("no block: %v", res.Errors)
	}
	body := arch.content(res.BlocksCreated[0].ArchivePath)
	if !strings.Contains(body, "错误：认证失败 🔐") {
		t.Fatal("the archive mangled non-ASCII content")
	}
	if !strings.Contains(body, "auth/jwt.go:42") {
		t.Fatal("the archive lost the file reference")
	}
}

// Tool output frequently contains markdown fences; the archive must not let
// them terminate its own fence.
func TestArchiveFenceSurvivesNestedFences(t *testing.T) {
	body := "```go\npackage main\n```\nand more ```inline``` text"
	out := RenderArchive(ArchiveRenderInput{
		BlockID: "b1", Tier: 1, StartRef: "m1", EndRef: "m1",
		Entries: []ArchiveEntry{{
			Message: &CoreMessage{ID: "x", Role: RoleTool, ContentType: CTToolResult, ToolName: "Bash", Text: body},
			Ref:     "m00001",
		}},
	})
	s := string(out)
	if !strings.Contains(s, body) {
		t.Fatal("the nested fences were altered")
	}
	if !strings.Contains(s, "````") {
		t.Fatalf("the wrapping fence was not lengthened past the nested one:\n%s", s)
	}
}

// --- Config extremes -----------------------------------------------------

func TestNudgeWithTinyWindow(t *testing.T) {
	cfg := DefaultConfig(1000)
	d := DecideNudge(DecideNudgeInput{
		State: CreateInitialState("s", "/a"), Config: cfg, TokenCount: 990,
		Recommendation: Recommendation{CompressibleRanges: []RangeInfo{
			{StartRef: "m1", EndRef: "m9", Tokens: 500, Chars: 6000},
		}},
	})
	// minPressureBenefit is max(5000, 1% of 1000) = 5000, above anything a
	// 1000-token window could ever reclaim — so it correctly declines.
	if d.ShouldInject {
		t.Fatalf("injected in a window too small to benefit: %s", d.Reason)
	}
	if d.Reason == "" {
		t.Fatal("no reason recorded")
	}
}

func TestTruncateWithZeroPreserveRecent(t *testing.T) {
	cfg := DefaultConfig(200000)
	cfg.PreserveRecentMessages = 0
	msgs := []CoreMessage{bigToolResult("a", 200000)}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if r.TruncatedCount != 1 {
		t.Fatalf("truncated %d, want the only message cut when nothing is protected", r.TruncatedCount)
	}
}

func TestProtectedRefsWithZeroConfig(t *testing.T) {
	cfg := DefaultConfig(200000)
	cfg.PreserveRecentMessages = 0
	cfg.PreserveRecentTokens = 0
	msgs := []CoreMessage{msg("a", RoleUser, CTText, "x"), msg("b", RoleUser, CTText, "y")}
	view, st := recommendFixture(t, msgs)
	if got := ComputeProtectedRefs(view, st, cfg); len(got) != 0 {
		t.Fatalf("protected = %v, want nothing protected when both knobs are zero", got)
	}
}

// --- Idempotence under repetition ---------------------------------------

// Re-running the projection many times must not drift; the prefix cache lives
// on this being exactly true.
func TestProcessTurnStableOverManyIterations(t *testing.T) {
	st := CreateInitialState("s", "/a")
	st.Blocks = []CompressionBlock{{
		BlockID: "b1", Tier: 1, Summary: "did the work", Active: true,
		DirectMessageIDs: []string{"m2"}, EffectiveMessageIDs: []string{"m2"},
		StartRef: "m00002", EndRef: "m00002", ArchivePath: "/a/b1.md",
	}}
	msgs := []CoreMessage{
		msg("m1", RoleUser, CTText, "task"),
		msg("m2", RoleAssistant, CTText, "work"),
		msg("m3", RoleUser, CTText, "more"),
	}
	cfg := DefaultConfig(200000)

	first := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: cfg})
	want := renderAll(first.Messages)
	state := first.State
	for i := range 20 {
		got := ProcessTurn(ProcessTurnInput{Messages: msgs, State: state, Config: cfg})
		if renderAll(got.Messages) != want {
			t.Fatalf("iteration %d drifted:\n%s\n---\n%s", i+2, want, renderAll(got.Messages))
		}
		state = got.State
	}
}

func renderAll(msgs []CoreMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(m.ID)
		b.WriteString("\x00")
		b.WriteString(m.Text)
		b.WriteString("\x01")
	}
	return b.String()
}
