package noa

import (
	"strings"
	"testing"
)

func bigToolResult(id string, chars int) CoreMessage {
	return CoreMessage{
		ID: id, Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t" + id, ToolName: "Bash",
		Text: strings.Repeat("x", chars),
	}
}

func TestTruncateBelowThresholdIsNoop(t *testing.T) {
	cfg := DefaultConfig(200000)
	msgs := []CoreMessage{bigToolResult("a", 100000)}
	r := TruncateLargeToolOutputs(msgs, 100000, cfg, nil) // 50%, far below 95%
	if r.TruncatedCount != 0 {
		t.Fatalf("truncated %d messages below the threshold", r.TruncatedCount)
	}
}

func TestTruncateCutsLargestFirst(t *testing.T) {
	cfg := DefaultConfig(200000)
	msgs := []CoreMessage{
		bigToolResult("small", 20000),
		bigToolResult("huge", 200000),
		bigToolResult("medium", 40000),
	}
	for range 5 { // the protected tail
		msgs = append(msgs, msg("tail", RoleUser, CTText, "recent"))
	}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if r.TruncatedCount == 0 {
		t.Fatal("nothing was truncated at 97.5% usage")
	}
	if !strings.Contains(r.Messages[1].Text, TruncationMarker) {
		t.Fatal("the largest result must be cut first")
	}
	if r.SavedTokens <= 0 {
		t.Fatalf("SavedTokens = %d, want a positive figure", r.SavedTokens)
	}
}

func TestTruncateKeepsHeadAndTail(t *testing.T) {
	cfg := DefaultConfig(200000)
	body := strings.Repeat("H", 3000) + strings.Repeat("M", 100000) + strings.Repeat("T", 3000)
	msgs := []CoreMessage{{ID: "big", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t1", Text: body}}
	for range 5 {
		msgs = append(msgs, msg("tail", RoleUser, CTText, "recent"))
	}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	out := r.Messages[0].Text
	if !strings.HasPrefix(out, strings.Repeat("H", 2000)) {
		t.Fatal("the head was not preserved")
	}
	if !strings.HasSuffix(out, strings.Repeat("T", 2000)) {
		t.Fatalf("the tail was not preserved: ...%q", out[max(len(out)-20, 0):])
	}
	if !strings.Contains(out, "original ~") {
		t.Fatalf("the marker must state the original size: %q", out[2000:2200])
	}
}

// Re-truncating reclaims nothing and would erode the head and tail further.
func TestTruncateSkipsAlreadyTruncated(t *testing.T) {
	cfg := DefaultConfig(200000)
	body := strings.Repeat("x", 3000) + "\n\n..." + TruncationMarker + " — original ~50000 tokens]...\n\n" +
		strings.Repeat("y", 3000)
	msgs := []CoreMessage{{ID: "done", Role: RoleTool, ContentType: CTToolResult, ToolCallID: "t1", Text: body}}
	for range 5 {
		msgs = append(msgs, msg("tail", RoleUser, CTText, "recent"))
	}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if r.TruncatedCount != 0 {
		t.Fatal("an already-truncated result was cut again")
	}
}

// The recent tail is where the current step is working; cutting it would break
// the task in progress.
func TestTruncateProtectsRecentMessages(t *testing.T) {
	cfg := DefaultConfig(200000)
	var msgs []CoreMessage
	for range 5 {
		msgs = append(msgs, bigToolResult("recent", 100000))
	}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if r.TruncatedCount != 0 {
		t.Fatalf("truncated %d messages inside the protected tail", r.TruncatedCount)
	}
}

func TestTruncateOnlyToolResults(t *testing.T) {
	cfg := DefaultConfig(200000)
	msgs := []CoreMessage{
		{ID: "u", Role: RoleUser, ContentType: CTText, Text: strings.Repeat("x", 200000)},
	}
	for range 5 {
		msgs = append(msgs, msg("tail", RoleUser, CTText, "recent"))
	}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if r.TruncatedCount != 0 {
		t.Fatal("a user message was truncated; only tool results may be cut")
	}
}

func TestTruncateSkipsSmallResults(t *testing.T) {
	cfg := DefaultConfig(200000)
	msgs := []CoreMessage{bigToolResult("small", 2000)} // ~500 tokens, under the 1000 floor
	for range 5 {
		msgs = append(msgs, msg("tail", RoleUser, CTText, "recent"))
	}
	r := TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if r.TruncatedCount != 0 {
		t.Fatal("a result below the minimum size was truncated for no real gain")
	}
}

// Truncation is a view-level repair; the caller's array must be untouched so
// the next projection sees the originals again.
func TestTruncateDoesNotMutateInput(t *testing.T) {
	cfg := DefaultConfig(200000)
	msgs := []CoreMessage{bigToolResult("big", 200000)}
	for range 5 {
		msgs = append(msgs, msg("tail", RoleUser, CTText, "recent"))
	}
	original := msgs[0].Text
	_ = TruncateLargeToolOutputs(msgs, 195000, cfg, nil)
	if msgs[0].Text != original {
		t.Fatal("the input array was mutated; truncation must apply only to what is sent")
	}
}

// End to end through the pipeline, and the count must reach the nudge so it can
// tell the model not to re-run those tools.
func TestProcessTurnTruncatesAndReports(t *testing.T) {
	cfg := DefaultConfig(200000)
	msgs := []CoreMessage{msg("u0", RoleUser, CTText, "the task")}
	// One result large enough to be cut, plus several that survive and still
	// give the nudge something worth asking about. Truncation stops as soon as
	// the target is met, so only the first is touched.
	msgs = append(msgs, bigToolResult("big", 400000))
	for i := range 3 {
		msgs = append(msgs, bigToolResult("mid"+idFor(i), 30000))
	}
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, strings.Repeat("t", 4000)))
	}
	res := ProcessTurn(ProcessTurnInput{
		Messages: msgs, State: CreateInitialState("s", "/a"), Config: cfg, TokenCount: 196000,
	})
	if res.TruncatedCount == 0 {
		t.Fatal("emergency-truncate did not fire at 98% usage")
	}
	if res.Nudge == nil {
		t.Fatal("no nudge at 98% usage with a large backlog")
	}
	if res.Nudge.TruncatedCount != res.TruncatedCount {
		t.Fatalf("nudge reports %d truncations, pipeline did %d", res.Nudge.TruncatedCount, res.TruncatedCount)
	}
	_, text := RenderNudgeText(*res.Nudge, NudgeSections{})
	if !strings.Contains(text, "Do NOT re-run those tools") {
		t.Fatal("the nudge must tell the model not to re-run truncated tools — that re-inflates the overflowing context")
	}
}
