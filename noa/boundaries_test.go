package noa

import (
	"strings"
	"testing"
)

// stateWithRefs builds a state whose ref map covers the given ids in order.
func stateWithRefs(ids ...string) CompressionState {
	st := CreateInitialState("s", "/tmp")
	for i, id := range ids {
		ref := IndexToRef(i + 1)
		st.MessageRefs.ByRaw[id] = ref
		st.MessageRefs.ByRef[ref] = id
	}
	return st
}

func TestParseBoundaryClassifies(t *testing.T) {
	cases := []struct {
		in   string
		kind BoundaryKind
		raw  string
	}{
		{"m00012", BoundaryMessage, "m00012"},
		{"m12", BoundaryMessage, "m00012"},
		{"b5", BoundaryBlock, "b5"},
		{"b005", BoundaryBlock, "b5"},
		{"b3(T2)", BoundaryBlock, "b3"},
	}
	for _, c := range cases {
		got, ok := ParseBoundary(c.in)
		if !ok || got.kind != c.kind || got.raw != c.raw {
			t.Errorf("ParseBoundary(%q) = %+v,%v, want kind=%s raw=%s", c.in, got, ok, c.kind, c.raw)
		}
	}
	for _, in := range []string{"", "xyz", "m0", "b0"} {
		if _, ok := ParseBoundary(in); ok {
			t.Errorf("ParseBoundary(%q) unexpectedly parsed", in)
		}
	}
}

// A bare number is a block id, not a message ref: message refs always carry the
// m prefix, which is what keeps the two addressable in one field.
func TestParseBoundaryBareNumberIsBlock(t *testing.T) {
	got, ok := ParseBoundary("5")
	if !ok || got.kind != BoundaryBlock || got.raw != "b5" {
		t.Fatalf("ParseBoundary(\"5\") = %+v,%v, want block b5", got, ok)
	}
}

func TestResolveBoundariesMessageRange(t *testing.T) {
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, "1"),
		msg("b", RoleAssistant, CTText, "2"),
		msg("c", RoleUser, CTText, "3"),
	}
	st := stateWithRefs("a", "b", "c")
	got, err := ResolveBoundaries("m00001", "m00002", msgs, st)
	if err != nil {
		t.Fatalf("ResolveBoundaries: %v", err)
	}
	if got.StartIndex != 0 || got.EndIndex != 1 {
		t.Fatalf("indices = %d..%d, want 0..1", got.StartIndex, got.EndIndex)
	}
	if got.Kind != BoundaryMessage {
		t.Fatalf("Kind = %s, want %s", got.Kind, BoundaryMessage)
	}
	if strings.Join(got.MessageIDs, ",") != "a,b" {
		t.Fatalf("MessageIDs = %v, want [a b]", got.MessageIDs)
	}
}

func TestResolveBoundariesSwapsReversed(t *testing.T) {
	msgs := []CoreMessage{msg("a", RoleUser, CTText, "1"), msg("b", RoleUser, CTText, "2")}
	st := stateWithRefs("a", "b")
	got, err := ResolveBoundaries("m00002", "m00001", msgs, st)
	if err != nil {
		t.Fatalf("ResolveBoundaries: %v", err)
	}
	if got.StartIndex != 0 || got.EndIndex != 1 {
		t.Fatalf("indices = %d..%d, want them swapped to 0..1", got.StartIndex, got.EndIndex)
	}
}

func TestResolveBoundariesExcludesRenderedSummaries(t *testing.T) {
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, "1"),
		{ID: SummaryMessageID("b9"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
		msg("c", RoleUser, CTText, "3"),
	}
	st := stateWithRefs("a", "c")
	// a is m00001, c is m00002 — the summary carries no ref.
	got, err := ResolveBoundaries("m00001", "m00002", msgs, st)
	if err != nil {
		t.Fatalf("ResolveBoundaries: %v", err)
	}
	if strings.Join(got.MessageIDs, ",") != "a,c" {
		t.Fatalf("MessageIDs = %v, want the rendered summary excluded", got.MessageIDs)
	}
}

func TestResolveBoundariesUnknownRef(t *testing.T) {
	msgs := []CoreMessage{msg("a", RoleUser, CTText, "1")}
	st := stateWithRefs("a")
	_, err := ResolveBoundaries("m00001", "m09999", msgs, st)
	if err == nil {
		t.Fatal("ResolveBoundaries accepted an unknown ref")
	}
	if err.Kind != NotFoundUnknown {
		t.Fatalf("Kind = %s, want %s", err.Kind, NotFoundUnknown)
	}
	if !strings.Contains(err.Error(), "does not exist in this session") {
		t.Fatalf("message = %q", err.Error())
	}
}

// A ref whose message is covered by an active block is consumed, not unknown —
// the distinction is what tells the model "you already compressed that" instead
// of "that ref is not yours".
func TestResolveBoundariesConsumedRef(t *testing.T) {
	msgs := []CoreMessage{msg("visible", RoleUser, CTText, "1")}
	st := stateWithRefs("visible", "hidden")
	st.Blocks = []CompressionBlock{block("b1", 1, "hidden")}

	_, err := ResolveBoundaries("m00001", "m00002", msgs, st)
	if err == nil {
		t.Fatal("ResolveBoundaries accepted a consumed ref")
	}
	if err.Kind != NotFoundConsumed {
		t.Fatalf("Kind = %s, want %s", err.Kind, NotFoundConsumed)
	}
}

func TestResolveBoundariesBlockRange(t *testing.T) {
	msgs := []CoreMessage{
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
		msg("loose", RoleUser, CTText, "between"),
		{ID: SummaryMessageID("b2"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
	}
	st := stateWithRefs("loose")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone1"), block("b2", 1, "gone2")}

	got, err := ResolveBoundaries("b1", "b2", msgs, st)
	if err != nil {
		t.Fatalf("ResolveBoundaries: %v", err)
	}
	if got.Kind != BoundaryBlock {
		t.Fatalf("Kind = %s, want %s — a block endpoint makes the whole range a block range", got.Kind, BoundaryBlock)
	}
	if got.StartIndex != 0 || got.EndIndex != 2 {
		t.Fatalf("indices = %d..%d, want 0..2", got.StartIndex, got.EndIndex)
	}
	if strings.Join(got.MessageIDs, ",") != "loose" {
		t.Fatalf("MessageIDs = %v, want only the uncompressed message between the blocks", got.MessageIDs)
	}
	if len(got.NestedBlockIDs) != 2 {
		t.Fatalf("NestedBlockIDs = %v, want both blocks", got.NestedBlockIDs)
	}
}

// Mixing a message endpoint with a block endpoint still yields a block range.
func TestResolveBoundariesMixedEndpointsAreBlockKind(t *testing.T) {
	msgs := []CoreMessage{
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
		msg("a", RoleUser, CTText, "1"),
	}
	st := stateWithRefs("a")
	st.Blocks = []CompressionBlock{block("b1", 1, "gone")}

	got, err := ResolveBoundaries("b1", "m00001", msgs, st)
	if err != nil {
		t.Fatalf("ResolveBoundaries: %v", err)
	}
	if got.Kind != BoundaryBlock {
		t.Fatalf("Kind = %s, want %s", got.Kind, BoundaryBlock)
	}
}

func TestResolveBoundariesUnknownBlock(t *testing.T) {
	msgs := []CoreMessage{msg("a", RoleUser, CTText, "1")}
	st := stateWithRefs("a")
	_, err := ResolveBoundaries("b99", "b99", msgs, st)
	if err == nil || err.Kind != NotFoundUnknown {
		t.Fatalf("err = %v, want an unknown-block error", err)
	}
}

// A ref pointing into an absorbed child is not stale: the content is still
// represented one tier up, so the endpoint snaps to the owner with a warning.
func TestResolveBoundariesSnapsToActiveOwner(t *testing.T) {
	msgs := []CoreMessage{
		msg("first", RoleUser, CTText, "1"),
		{ID: SummaryMessageID("b2"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
	}
	st := stateWithRefs("first", "absorbed")
	child := block("b1", 1, "absorbed")
	child.Active = false
	owner := block("b2", 2, "absorbed")
	owner.DirectBlockIDs = []string{"b1"}
	st.Blocks = []CompressionBlock{child, owner}

	got, err := ResolveBoundaries("m00001", "m00002", msgs, st)
	if err != nil {
		t.Fatalf("ResolveBoundaries: %v", err)
	}
	if got.EndIndex != 1 {
		t.Fatalf("EndIndex = %d, want 1 (snapped to b2's anchor)", got.EndIndex)
	}
	if len(got.Warnings) != 1 || !strings.Contains(got.Warnings[0], "b2") {
		t.Fatalf("Warnings = %v, want one mentioning the absorbing block", got.Warnings)
	}
}

func TestResolveBoundariesInvalidRef(t *testing.T) {
	msgs := []CoreMessage{msg("a", RoleUser, CTText, "1")}
	st := stateWithRefs("a")
	for _, bad := range []string{"", "nonsense", "m0"} {
		_, err := ResolveBoundaries(bad, "m00001", msgs, st)
		if err == nil || err.Kind != NotFoundInvalid {
			t.Fatalf("ResolveBoundaries(%q) err = %v, want an invalid-ref error", bad, err)
		}
	}
}
