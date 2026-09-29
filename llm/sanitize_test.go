package llm

import (
	"strings"
	"testing"
)

func oaCall(id string) oaToolCall {
	tc := oaToolCall{ID: id, Type: "function"}
	tc.Function.Name = "Read"
	tc.Function.Arguments = "{}"
	return tc
}

func countRole(msgs []oaMessage, role string) int {
	n := 0
	for _, m := range msgs {
		if m.Role == role {
			n++
		}
	}
	return n
}

// An assistant message carrying a tool_call with no matching role:tool response
// is what a half-covered compaction leaves behind; the unanswered call must go
// so the provider does not 400 with "tool_calls must be followed by tool
// messages".
func TestSanitizeDropsUnansweredToolCall(t *testing.T) {
	msgs := []oaMessage{
		{Role: "assistant", ToolCalls: []oaToolCall{oaCall("a"), oaCall("b")}},
		{Role: "tool", ToolCallID: "a", Content: "ok"},
	}
	got := sanitizeOpenAIMessages(msgs, false)
	if len(got[0].ToolCalls) != 1 || got[0].ToolCalls[0].ID != "a" {
		t.Fatalf("assistant tool_calls = %+v, want only the answered call a", got[0].ToolCalls)
	}
	if countRole(got, "tool") != 1 {
		t.Fatalf("tool messages = %d, want 1", countRole(got, "tool"))
	}
}

// A role:tool message whose call is gone (its assistant turn was compressed
// away) is an orphan a strict provider also rejects.
func TestSanitizeDropsOrphanToolResult(t *testing.T) {
	msgs := []oaMessage{
		{Role: "assistant", Content: "hi"},
		{Role: "tool", ToolCallID: "ghost", Content: "stale"},
	}
	got := sanitizeOpenAIMessages(msgs, false)
	if countRole(got, "tool") != 0 {
		t.Fatalf("orphan tool result survived: %+v", got)
	}
}

// In thinking mode an assistant turn replaying tool_calls must carry
// reasoning_content; when it was lost upstream the guard fills a placeholder
// rather than let the request fail.
func TestSanitizeFillsReasoningInThinkingMode(t *testing.T) {
	msgs := []oaMessage{
		{Role: "assistant", ToolCalls: []oaToolCall{oaCall("a")}},
		{Role: "tool", ToolCallID: "a", Content: "ok"},
	}
	got := sanitizeOpenAIMessages(msgs, true)
	if got[0].ReasoningContent == "" {
		t.Fatalf("thinking-mode assistant with tool_calls must carry reasoning_content")
	}
}

// The placeholder is a thinking-mode-only concession: without thinking, a
// content-less tool_calls turn is valid and must be left as it is.
func TestSanitizeNoReasoningFillWhenNotThinking(t *testing.T) {
	msgs := []oaMessage{
		{Role: "assistant", ToolCalls: []oaToolCall{oaCall("a")}},
		{Role: "tool", ToolCallID: "a", Content: "ok"},
	}
	got := sanitizeOpenAIMessages(msgs, false)
	if got[0].ReasoningContent != "" {
		t.Fatalf("reasoning_content = %q, want empty when not in thinking mode", got[0].ReasoningContent)
	}
}

// An assistant message emptied by dropping its only (unanswered) tool_call would
// serialize to a bare {"role":"assistant"} and 400; it must gain placeholder
// content instead.
func TestSanitizeEmptiedAssistantGainsContent(t *testing.T) {
	msgs := []oaMessage{
		{Role: "assistant", ToolCalls: []oaToolCall{oaCall("a")}}, // no tool response
	}
	got := sanitizeOpenAIMessages(msgs, false)
	if len(got[0].ToolCalls) != 0 {
		t.Fatalf("unanswered call survived: %+v", got[0].ToolCalls)
	}
	if got[0].Content == "" {
		t.Fatalf("emptied assistant has no content — would serialize to a bare assistant message")
	}
}

// A well-formed transcript must pass through untouched: pairs intact, real
// reasoning_content preserved, nothing added or removed.
func TestSanitizeLeavesValidTranscriptIntact(t *testing.T) {
	msgs := []oaMessage{
		{Role: "user", Content: "task"},
		{Role: "assistant", ReasoningContent: "real thinking", ToolCalls: []oaToolCall{oaCall("a")}},
		{Role: "tool", ToolCallID: "a", Content: "ok"},
		{Role: "assistant", Content: "done"},
	}
	got := sanitizeOpenAIMessages(msgs, true)
	if len(got) != 4 {
		t.Fatalf("len = %d, want 4 (nothing dropped): %+v", len(got), got)
	}
	if got[1].ReasoningContent != "real thinking" {
		t.Fatalf("real reasoning_content was clobbered: %q", got[1].ReasoningContent)
	}
	if len(got[1].ToolCalls) != 1 {
		t.Fatalf("valid tool_call was dropped: %+v", got[1].ToolCalls)
	}
}

// End-to-end: the two failures from issue #174 in one transcript — an unanswered
// tool_call and a thinking turn that lost its reasoning — both repaired by the
// guard reached through buildBody.
func TestBuildBodyAppliesSendGuard(t *testing.T) {
	p := &openaiProvider{cfg: Config{Model: "glm-4.6", ThinkingType: "enabled"}}
	req := CompletionRequest{
		Messages: []Message{
			{Role: RoleAssistant, Content: []ContentBlock{
				{Type: BlockToolUse, ID: "x", Name: "Read", Input: []byte(`{}`)},
			}},
			// No tool_result for "x" — the half-covered case.
			UserText("continue"),
		},
	}
	raw, err := p.buildBody(req, false)
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}
	s := string(raw)
	// The unanswered tool_call must not reach the wire.
	if strings.Contains(s, `"tool_calls"`) {
		t.Fatalf("unanswered tool_call was sent: %s", s)
	}
}
