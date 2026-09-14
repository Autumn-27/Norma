package noa

import "testing"

func msg(id string, role Role, ct ContentType, text string) CoreMessage {
	return CoreMessage{ID: id, Role: role, ContentType: ct, Text: text}
}

func TestIndexToRef(t *testing.T) {
	cases := map[int]string{1: "m00001", 42: "m00042", 99999: "m99999", 0: "", 100000: "", -1: ""}
	for in, want := range cases {
		if got := IndexToRef(in); got != want {
			t.Errorf("IndexToRef(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestRefToIndex(t *testing.T) {
	ok := map[string]int{
		"m00001": 1, "m00042": 42, "m99999": 99999,
		"m1": 1, "m042": 42, // missing padding tolerated
		"  M00007  ": 7, // trimmed and lowercased
	}
	for in, want := range ok {
		got, valid := RefToIndex(in)
		if !valid || got != want {
			t.Errorf("RefToIndex(%q) = %d,%v, want %d,true", in, got, valid, want)
		}
	}
	for _, in := range []string{"", "m", "m0", "m00000", "b5", "00001", "m123456", "mm1", "m1x"} {
		if got, valid := RefToIndex(in); valid {
			t.Errorf("RefToIndex(%q) = %d,true, want invalid", in, got)
		}
	}
}

func TestAssignRefsIsIdempotent(t *testing.T) {
	msgs := []CoreMessage{
		msg("a", RoleUser, CTText, "one"),
		msg("b", RoleAssistant, CTText, "two"),
	}
	first := AssignRefs(msgs, AssignRefsOptions{Existing: MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}}, NextIndex: 1})
	if first.NewlyAssigned != 2 {
		t.Fatalf("NewlyAssigned = %d, want 2", first.NewlyAssigned)
	}
	second := AssignRefs(msgs, AssignRefsOptions{Existing: first.Map, NextIndex: HighestUsedIndex(first.Map) + 1})
	if second.NewlyAssigned != 0 {
		t.Fatalf("second pass assigned %d refs, want 0 — refs must never be reassigned", second.NewlyAssigned)
	}
	for id, ref := range first.Map.ByRaw {
		if second.Map.ByRaw[id] != ref {
			t.Fatalf("ref for %q changed from %q to %q — the core invariant is that it never does", id, ref, second.Map.ByRaw[id])
		}
	}
}

// A ref stays with its message even as the message set grows and shifts.
func TestAssignRefsSurvivesGrowth(t *testing.T) {
	m := MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}}
	r1 := AssignRefs([]CoreMessage{msg("a", RoleUser, CTText, "1")}, AssignRefsOptions{Existing: m, NextIndex: 1})
	want := r1.Map.ByRaw["a"]

	grown := []CoreMessage{
		msg("x", RoleUser, CTText, "new first"),
		msg("a", RoleUser, CTText, "1"),
		msg("y", RoleUser, CTText, "new last"),
	}
	r2 := AssignRefs(grown, AssignRefsOptions{Existing: r1.Map, NextIndex: HighestUsedIndex(r1.Map) + 1})
	if got := r2.Map.ByRaw["a"]; got != want {
		t.Fatalf("ref for %q = %q after reordering, want %q", "a", got, want)
	}
	if r2.Map.ByRaw["x"] == want || r2.Map.ByRaw["y"] == want {
		t.Fatal("a new message reused an existing ref")
	}
}

func TestAssignRefsBlockedConsumesNoNumber(t *testing.T) {
	msgs := []CoreMessage{
		{ID: "p", Role: RoleAssistant, ContentType: CTToolCall, ToolName: CompressToolName, ToolCallID: "t1"},
		msg("a", RoleUser, CTText, "visible"),
	}
	cfg := DefaultConfig(200000)
	res := AssignRefs(msgs, AssignRefsOptions{
		Existing:    MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}},
		NextIndex:   1,
		IsProtected: func(m CoreMessage) bool { return IsMessageProtected(m, cfg) },
	})
	if res.Map.ByRaw["p"] != BlockedRef {
		t.Fatalf("protected message ref = %q, want %q", res.Map.ByRaw["p"], BlockedRef)
	}
	if res.Map.ByRaw["a"] != "m00001" {
		t.Fatalf("ref after a BLOCKED message = %q, want m00001 — BLOCKED must not consume a number", res.Map.ByRaw["a"])
	}
	if _, ok := res.Map.ByRef[BlockedRef]; ok {
		t.Fatal("BLOCKED leaked into the reverse index")
	}
}

func TestAssignRefsSkipsSyntheticMessages(t *testing.T) {
	msgs := []CoreMessage{
		{ID: SummaryMessageID("b1"), Role: RoleUser, ContentType: CTText, Text: SummaryHeader},
		{ID: NudgeMessageID, Role: RoleUser, ContentType: CTText, Text: "nudge"},
		msg("a", RoleUser, CTText, "real"),
	}
	res := AssignRefs(msgs, AssignRefsOptions{
		Existing:  MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}},
		NextIndex: 1,
		ShouldSkip: func(m CoreMessage) bool {
			return isRenderedSummaryMessage(m) || m.ID == NudgeMessageID
		},
	})
	if len(res.Map.ByRaw) != 1 || res.Map.ByRaw["a"] != "m00001" {
		t.Fatalf("ByRaw = %v, want only the real message at m00001", res.Map.ByRaw)
	}
}

func TestAssignRefsSkipsOccupiedNumbers(t *testing.T) {
	existing := MessageRefMap{
		ByRaw: map[string]string{"old": "m00002"},
		ByRef: map[string]string{"m00002": "old"},
	}
	res := AssignRefs([]CoreMessage{msg("new", RoleUser, CTText, "x")},
		AssignRefsOptions{Existing: existing, NextIndex: 1})
	// Cursor starts at 1, m00001 is free, so it is used; m00002 must be untouched.
	if res.Map.ByRaw["new"] != "m00001" {
		t.Fatalf("ref = %q, want m00001", res.Map.ByRaw["new"])
	}
	if res.Map.ByRef["m00002"] != "old" {
		t.Fatal("an occupied ref was overwritten")
	}
}

func TestAssignRefsSkipsEmptyID(t *testing.T) {
	res := AssignRefs([]CoreMessage{{Role: RoleUser, ContentType: CTText, Text: "no id"}},
		AssignRefsOptions{Existing: MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}}, NextIndex: 1})
	if len(res.Map.ByRaw) != 0 {
		t.Fatalf("ByRaw = %v, want empty for a message with no id", res.Map.ByRaw)
	}
}

func TestHighestUsedIndexIgnoresBlocked(t *testing.T) {
	m := MessageRefMap{ByRaw: map[string]string{
		"a": "m00003",
		"b": BlockedRef,
		"c": "m00007",
		"d": BlockedRef,
	}}
	if got := HighestUsedIndex(m); got != 7 {
		t.Fatalf("HighestUsedIndex = %d, want 7", got)
	}
	if got := HighestUsedIndex(MessageRefMap{ByRaw: map[string]string{"a": BlockedRef}}); got != 0 {
		t.Fatalf("HighestUsedIndex with only BLOCKED = %d, want 0", got)
	}
	if got := HighestUsedIndex(MessageRefMap{}); got != 0 {
		t.Fatalf("HighestUsedIndex of empty map = %d, want 0", got)
	}
}

// Running out of ref space must stop cleanly, not corrupt the map.
func TestAssignRefsExhaustion(t *testing.T) {
	existing := MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}}
	for i := MinRefIndex; i <= MaxRefIndex; i++ {
		ref := IndexToRef(i)
		existing.ByRef[ref] = "filler"
	}
	res := AssignRefs([]CoreMessage{msg("late", RoleUser, CTText, "x")},
		AssignRefsOptions{Existing: existing, NextIndex: MaxRefIndex})
	if _, ok := res.Map.ByRaw["late"]; ok {
		t.Fatal("a ref was handed out from an exhausted space")
	}
	if res.NewlyAssigned != 0 {
		t.Fatalf("NewlyAssigned = %d, want 0", res.NewlyAssigned)
	}
}

func TestParseBlockID(t *testing.T) {
	ok := map[string]string{
		"b1": "b1", "b005": "b5", "5": "b5", " B12 ": "b12",
		"b3(T2)": "b3", "b3(t3)": "b3", // the displayed form parses too
	}
	for in, want := range ok {
		if got, valid := ParseBlockID(in); !valid || got != want {
			t.Errorf("ParseBlockID(%q) = %q,%v, want %q,true", in, got, valid, want)
		}
	}
	for _, in := range []string{"", "b", "b0", "0", "m00001", "bx", "-1"} {
		if got, valid := ParseBlockID(in); valid {
			t.Errorf("ParseBlockID(%q) = %q,true, want invalid", in, got)
		}
	}
}
