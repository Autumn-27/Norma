package llm

import "testing"

func toolUseMsg(id string) Message {
	return Message{Role: RoleAssistant, Content: []ContentBlock{
		{Type: BlockToolUse, ID: id, Name: "Bash", Input: []byte(`{}`)},
	}}
}

func toolResultMsg(id string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{
		{Type: BlockToolResult, ToolUseID: id, Content: []ContentBlock{TextBlock("ok")}},
	}}
}

// The reported failure: compaction slices the working set right before a
// tool_result whose tool_use was summarized away. MessagesForAPI must not emit
// that orphan — it is what makes the gateway 400.
func TestMessagesForAPIDropsOrphanToolResult(t *testing.T) {
	msgs := []Message{
		toolUseMsg("t1"), // pre-boundary: summarized away
		BoundaryMessage(BoundaryMeta{Trigger: "auto"}),
		UserText("Summary: ..."),
		toolResultMsg("t1"), // orphan: its tool_use is before the boundary
		UserText("continue"),
	}
	out := MessagesForAPI(msgs)
	for _, m := range out {
		for _, b := range m.Content {
			if b.Type == BlockToolResult {
				t.Fatalf("orphan tool_result survived: %+v", out)
			}
		}
	}
	// The empty message left by dropping the orphan block is removed, so the view
	// is [summary, continue].
	if len(out) != 2 || out[0].Content[0].Text != "Summary: ..." || out[1].Content[0].Text != "continue" {
		t.Fatalf("view=%+v", out)
	}
}

// A dangling tool_use (result beyond the view) is dropped symmetrically.
func TestMessagesForAPIDropsDanglingToolUse(t *testing.T) {
	msgs := []Message{
		UserText("hi"),
		toolUseMsg("t9"), // no matching tool_result anywhere in the view
	}
	out := MessagesForAPI(msgs)
	if len(out) != 1 || out[0].Content[0].Text != "hi" {
		t.Fatalf("view=%+v", out)
	}
}

// A well-formed, intact tool exchange must pass through untouched.
func TestMessagesForAPIKeepsPairedExchange(t *testing.T) {
	msgs := []Message{
		UserText("go"),
		toolUseMsg("t1"),
		toolResultMsg("t1"),
		{Role: RoleAssistant, Content: []ContentBlock{TextBlock("done")}},
	}
	out := MessagesForAPI(msgs)
	if len(out) != 4 {
		t.Fatalf("want 4 messages unchanged, got %d: %+v", len(out), out)
	}
	if out[1].Content[0].Type != BlockToolUse || out[2].Content[0].Type != BlockToolResult {
		t.Fatalf("paired exchange altered: %+v", out)
	}
}

// One user message can carry several tool_results (parallel tools). If some are
// orphaned and some are not, the valid ones survive and the message is kept.
func TestMessagesForAPIMixedToolResultsInOneMessage(t *testing.T) {
	msgs := []Message{
		toolUseMsg("keep"), // in-view tool_use for the valid result
		{Role: RoleUser, Content: []ContentBlock{
			{Type: BlockToolResult, ToolUseID: "orphan", Content: []ContentBlock{TextBlock("x")}},
			{Type: BlockToolResult, ToolUseID: "keep", Content: []ContentBlock{TextBlock("y")}},
		}},
	}
	out := MessagesForAPI(msgs)
	// [assistant tool_use "keep", user with only the "keep" result]
	if len(out) != 2 {
		t.Fatalf("want 2 messages, got %d: %+v", len(out), out)
	}
	last := out[1].Content
	if len(last) != 1 || last[0].ToolUseID != "keep" {
		t.Fatalf("valid result not kept / orphan not dropped: %+v", last)
	}
}

// One assistant message can carry several tool_use blocks (parallel calls). A
// block whose result is missing is dropped; the one with a result is kept.
func TestMessagesForAPIMixedToolUsesInOneMessage(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockToolUse, ID: "done", Name: "Bash", Input: []byte(`{}`)},
			{Type: BlockToolUse, ID: "dangling", Name: "Bash", Input: []byte(`{}`)},
		}},
		toolResultMsg("done"),
	}
	out := MessagesForAPI(msgs)
	if len(out) != 2 {
		t.Fatalf("want 2 messages, got %d: %+v", len(out), out)
	}
	uses := out[0].Content
	if len(uses) != 1 || uses[0].ID != "done" {
		t.Fatalf("dangling tool_use not dropped / paired one not kept: %+v", uses)
	}
}

// Only the LAST boundary defines the working set; an orphan after it is dropped.
func TestMessagesForAPIMultipleBoundaries(t *testing.T) {
	msgs := []Message{
		UserText("old"),
		BoundaryMessage(BoundaryMeta{Trigger: "auto"}),
		toolUseMsg("t1"),
		BoundaryMessage(BoundaryMeta{Trigger: "auto"}),
		UserText("Summary 2"),
		toolResultMsg("t1"), // its tool_use is before the LAST boundary → orphan
	}
	out := MessagesForAPI(msgs)
	if len(out) != 1 || out[0].Content[0].Text != "Summary 2" {
		t.Fatalf("view=%+v", out)
	}
}

// No tools, no boundary: the pass is a no-op and returns the conversation as-is.
func TestMessagesForAPINoToolsNoOp(t *testing.T) {
	msgs := []Message{
		UserText("hi"),
		{Role: RoleAssistant, Content: []ContentBlock{TextBlock("hello")}},
		UserText("bye"),
	}
	out := MessagesForAPI(msgs)
	if len(out) != 3 || out[0].Content[0].Text != "hi" || out[2].Content[0].Text != "bye" {
		t.Fatalf("view changed: %+v", out)
	}
}

// After compaction the view begins with the summary (a user message), so
// dropping the orphaned tool_result that follows it never leaves the view
// starting with an assistant turn.
func TestMessagesForAPIStaysUserFirstAfterDrop(t *testing.T) {
	msgs := []Message{
		toolUseMsg("t1"),
		BoundaryMessage(BoundaryMeta{Trigger: "auto"}),
		UserText("Summary"),
		toolResultMsg("t1"),
		{Role: RoleAssistant, Content: []ContentBlock{TextBlock("next")}},
	}
	out := MessagesForAPI(msgs)
	if len(out) == 0 || out[0].Role != RoleUser {
		t.Fatalf("view must start with a user message, got %+v", out)
	}
}

// The pair kept together in the tail (both after the boundary) stays intact —
// only the boundary-straddling one is dropped.
func TestMessagesForAPIKeepsInTailPair(t *testing.T) {
	msgs := []Message{
		toolUseMsg("old"), // pre-boundary
		BoundaryMessage(BoundaryMeta{Trigger: "auto"}),
		UserText("Summary"),
		toolResultMsg("old"), // orphan → dropped
		toolUseMsg("new"),    // in-tail pair → kept
		toolResultMsg("new"),
	}
	out := MessagesForAPI(msgs)
	var uses, results int
	for _, m := range out {
		for _, b := range m.Content {
			if b.Type == BlockToolUse {
				uses++
			}
			if b.Type == BlockToolResult {
				results++
			}
		}
	}
	if uses != 1 || results != 1 {
		t.Fatalf("want the in-tail pair kept and the orphan dropped, got uses=%d results=%d: %+v", uses, results, out)
	}
}
