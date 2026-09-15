package noa

import (
	"strings"
	"testing"
)

func pruneWith(msgs []CoreMessage, st CompressionState) []CoreMessage {
	io := prune(NodeIO{Messages: msgs, State: st}, PipelineContext{Config: DefaultConfig(200000)})
	return io.Messages
}

func ids(msgs []CoreMessage) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.ID
	}
	return out
}

func TestPruneNoBlocksIsNoop(t *testing.T) {
	msgs := []CoreMessage{msg("a", RoleUser, CTText, "1"), msg("b", RoleAssistant, CTText, "2")}
	got := pruneWith(msgs, CreateInitialState("s", "/tmp"))
	if len(got) != 2 {
		t.Fatalf("prune with no blocks changed the view: %v", ids(got))
	}
}

func TestPruneReplacesCoveredWithSummary(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "m2", "m3")}
	msgs := []CoreMessage{
		msg("m1", RoleUser, CTText, "first user"),
		msg("m2", RoleAssistant, CTText, "compressed a"),
		msg("m3", RoleAssistant, CTText, "compressed b"),
		msg("m4", RoleUser, CTText, "recent"),
	}
	got := pruneWith(msgs, st)
	want := []string{"m1", SummaryMessageID("b1"), "m4"}
	if strings.Join(ids(got), ",") != strings.Join(want, ",") {
		t.Fatalf("prune = %v, want %v", ids(got), want)
	}
	if !strings.HasPrefix(got[1].Text, SummaryHeader) {
		t.Fatalf("summary text = %q, want it to open with the header", got[1].Text)
	}
}

// The first user message is the session's task definition; losing it costs the
// model the reason it is doing any of this.
func TestPruneAlwaysKeepsFirstUserMessage(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "m1", "m2")}
	msgs := []CoreMessage{
		msg("m1", RoleUser, CTText, "the task"),
		msg("m2", RoleAssistant, CTText, "work"),
	}
	got := pruneWith(msgs, st)
	if got[0].ID != SummaryMessageID("b1") {
		t.Fatalf("first message = %q, want the summary anchored at index 0", got[0].ID)
	}
	found := false
	for _, m := range got {
		if m.ID == "m1" {
			found = true
		}
	}
	if !found {
		t.Fatalf("prune = %v, want the first user message preserved even though a block covers it", ids(got))
	}
}

func TestPruneAnchorsAtExistingSummaryPosition(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}
	msgs := []CoreMessage{
		msg("u", RoleUser, CTText, "first"),
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader + "\nold rendering"},
		msg("later", RoleUser, CTText, "after"),
	}
	got := pruneWith(msgs, st)
	want := []string{"u", SummaryMessageID("b1"), "later"}
	if strings.Join(ids(got), ",") != strings.Join(want, ",") {
		t.Fatalf("prune = %v, want %v — the summary must stay where it already is", ids(got), want)
	}
	count := 0
	for _, m := range got {
		if m.ID == SummaryMessageID("b1") {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("summary appears %d times, want exactly 1", count)
	}
}

func TestPruneOrdersAnchorsStably(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	// Both blocks anchor at the same index (their coverage is absent), so the
	// tie is broken by block number.
	st.Blocks = []CompressionBlock{block("b2", 1, "gone2"), block("b1", 1, "gone1")}
	msgs := []CoreMessage{msg("u", RoleUser, CTText, "only")}
	got := pruneWith(msgs, st)
	if got[0].ID != SummaryMessageID("b1") || got[1].ID != SummaryMessageID("b2") {
		t.Fatalf("prune = %v, want b1 before b2 at the same anchor", ids(got))
	}
}

func TestPruneOrdersByCoveragePosition(t *testing.T) {
	st := CreateInitialState("s", "/tmp")
	st.Blocks = []CompressionBlock{block("b2", 1, "m4"), block("b1", 1, "m2")}
	msgs := []CoreMessage{
		msg("m1", RoleUser, CTText, "first"),
		msg("m2", RoleAssistant, CTText, "a"),
		msg("m3", RoleUser, CTText, "between"),
		msg("m4", RoleAssistant, CTText, "b"),
	}
	got := pruneWith(msgs, st)
	want := []string{"m1", SummaryMessageID("b1"), "m3", SummaryMessageID("b2")}
	if strings.Join(ids(got), ",") != strings.Join(want, ",") {
		t.Fatalf("prune = %v, want %v", ids(got), want)
	}
}

func TestStripOrphanedToolCalls(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "c1", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t1", ToolName: "Read"},
		{ID: "r1", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t1"},
		{ID: "c2", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t2", ToolName: "Bash"},
	}
	got := stripOrphanedToolCalls(msgs)
	if strings.Join(ids(got), ",") != "c1,r1" {
		t.Fatalf("stripOrphanedToolCalls = %v, want the unpaired call dropped", ids(got))
	}
}

// The Compress call is the addressable anchor of a block's summary. It must
// survive even after hide-compress-calls removed its result.
func TestStripOrphanedToolCallsExemptsCompress(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "c1", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t1", ToolName: CompressToolName},
	}
	got := stripOrphanedToolCalls(msgs)
	if len(got) != 1 {
		t.Fatalf("stripOrphanedToolCalls dropped the Compress call: %v", ids(got))
	}
}

func TestStripOrphanedToolResults(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "r1", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "gone"},
		{ID: "c2", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t2", ToolName: "Read"},
		{ID: "r2", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t2"},
	}
	got := stripOrphanedToolResults(msgs)
	if strings.Join(ids(got), ",") != "c2,r2" {
		t.Fatalf("stripOrphanedToolResults = %v, want the unpaired result dropped", ids(got))
	}
}

func TestStripOrphanedReasoningKeepsPairedRun(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "think", Role: RoleAssistant, ContentType: CTReasoning, Text: "hmm"},
		{ID: "act", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t1", ToolName: "Read"},
	}
	got := stripOrphanedReasoning(msgs)
	if len(got) != 2 {
		t.Fatalf("stripOrphanedReasoning = %v, want the run kept — it precedes an assistant act", ids(got))
	}
}

func TestStripOrphanedReasoningDropsDanglingRun(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "t1", Role: RoleAssistant, ContentType: CTReasoning, Text: "a"},
		{ID: "t2", Role: RoleAssistant, ContentType: CTReasoning, Text: "b"},
		{ID: "u", Role: RoleUser, ContentType: CTText, Text: "next question"},
	}
	got := stripOrphanedReasoning(msgs)
	if strings.Join(ids(got), ",") != "u" {
		t.Fatalf("stripOrphanedReasoning = %v, want the whole run dropped", ids(got))
	}
}

func TestStripOrphanedReasoningDropsTrailingRun(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "u", Role: RoleUser, ContentType: CTText, Text: "q"},
		{ID: "t1", Role: RoleAssistant, ContentType: CTReasoning, Text: "a"},
	}
	got := stripOrphanedReasoning(msgs)
	if strings.Join(ids(got), ",") != "u" {
		t.Fatalf("stripOrphanedReasoning = %v, want the trailing run dropped", ids(got))
	}
}

func TestStripOrphanedReasoningHandlesMultipleRuns(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "k1", Role: RoleAssistant, ContentType: CTReasoning, Text: "kept"},
		{ID: "a1", Role: RoleAssistant, ContentType: CTText, Text: "act"},
		{ID: "d1", Role: RoleAssistant, ContentType: CTReasoning, Text: "dropped"},
		{ID: "u", Role: RoleUser, ContentType: CTText, Text: "q"},
		{ID: "k2", Role: RoleAssistant, ContentType: CTReasoning, Text: "kept"},
		{ID: "a2", Role: RoleAssistant, ContentType: CTToolCall, ToolCallID: "t", ToolName: "Read"},
	}
	got := stripOrphanedReasoning(msgs)
	if strings.Join(ids(got), ",") != "k1,a1,u,k2,a2" {
		t.Fatalf("stripOrphanedReasoning = %v, want only the dangling run dropped", ids(got))
	}
}

func TestIsRenderedSummaryMessage(t *testing.T) {
	good := CoreMessage{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader + " — topic"}
	if !isRenderedSummaryMessage(good) {
		t.Fatal("a well-formed rendered summary was not recognised")
	}
	// Each condition alone must not be enough — a host-authored message that
	// merely shares the id prefix must not be silently deleted by a rebuild.
	bad := []CoreMessage{
		{ID: "other", Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
		{ID: SummaryMessageID("b1"), Role: RoleAssistant, ContentType: CTText, Text: SummaryHeader},
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTToolResult, Text: SummaryHeader},
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: "host wrote this"},
	}
	for i, m := range bad {
		if isRenderedSummaryMessage(m) {
			t.Errorf("case %d: %+v was wrongly recognised as a rendered summary", i, m)
		}
	}
}

func TestRenderSummaryMessage(t *testing.T) {
	b := CompressionBlock{
		BlockID: "b5", Tier: 2, Topic: "认证子系统", Summary: "  decided on JWT  ",
		StartRef: "m00012", EndRef: "m00160", ArchivePath: "/data/noa/s/tier2/b5.md",
	}
	m := RenderSummaryMessage(b)
	if m.ID != "noa_summary_b5" || m.Role != RoleUser || m.ContentType != CTText {
		t.Fatalf("rendered summary identity = %+v", m)
	}
	lines := strings.Split(m.Text, "\n")
	if lines[0] != SummaryHeader+" — 认证子系统" {
		t.Fatalf("header line = %q", lines[0])
	}
	if lines[1] != "decided on JWT" {
		t.Fatalf("summary line = %q, want it trimmed", lines[1])
	}
	last := lines[len(lines)-1]
	want := `<noa-archive block="b5" tier="2" range="m00012-m00160" path="/data/noa/s/tier2/b5.md"/>`
	if last != want {
		t.Fatalf("archive tag = %q, want %q", last, want)
	}
}

func TestRenderSummaryMessageWithoutTopicOrSummary(t *testing.T) {
	b := CompressionBlock{BlockID: "b1", Tier: 1, StartRef: "m1", EndRef: "m2", ArchivePath: "/p.md"}
	m := RenderSummaryMessage(b)
	if !strings.HasPrefix(m.Text, SummaryHeader+"\n\n<noa-archive") {
		t.Fatalf("text = %q, want header then archive tag with no topic or body", m.Text)
	}
}

// The archive tag must not look like a ref tag: the adapter's strip regexes key
// on id="mNNNNN", and a collision would corrupt message text.
func TestArchiveTagIsNotARefTag(t *testing.T) {
	tag := ArchiveTag(CompressionBlock{BlockID: "b1", Tier: 1, StartRef: "m00001", EndRef: "m00009", ArchivePath: "/p.md"})
	if strings.Contains(tag, `<noa-ref`) {
		t.Fatalf("archive tag %q must not contain a noa-ref element", tag)
	}
	if strings.Contains(tag, ` id="m`) {
		t.Fatalf("archive tag %q must not carry an id=\"mNNNNN\" attribute", tag)
	}
}
