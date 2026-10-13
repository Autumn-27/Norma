package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// A Gemini-recorded history replayed against Anthropic must not carry
// thought_signature: anthropicReq.Messages is []Message, so ContentBlock goes
// straight onto the wire and an unknown key inside a tool_use block draws
// "Extra inputs are not permitted" on every turn.
func TestAnthropicRequestDropsThoughtSignature(t *testing.T) {
	history := []Message{
		UserText("run it"),
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockToolUse, ID: "c1", Name: "Bash", Input: json.RawMessage(`{"command":"ls"}`), ThoughtSignature: "sig-abc"},
		}},
	}

	out := filterThinkingBlocks(history, false)
	for _, m := range out {
		for _, b := range m.Content {
			if b.ThoughtSignature != "" {
				t.Fatalf("thought_signature survived normalization: %q", b.ThoughtSignature)
			}
		}
	}

	// Belt and braces: the field must be absent from the marshaled body, since
	// that is what Anthropic actually parses.
	body, err := json.Marshal(anthropicReq{Messages: mergeAdjacentSameRole(out)})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "thought_signature") {
		t.Errorf("request body carries thought_signature:\n%s", body)
	}

	// The caller's conversation state must be untouched — it is replayed again
	// next turn, and against Gemini the signature is still required.
	if got := history[1].Content[0].ThoughtSignature; got != "sig-abc" {
		t.Errorf("normalization mutated the caller's history: signature = %q, want %q", got, "sig-abc")
	}
}

// Histories with no signature anywhere must keep sharing their backing array —
// stripping is a no-op that should not allocate a copy per turn.
func TestStripThoughtSignaturesNoCopyWhenUnset(t *testing.T) {
	content := []ContentBlock{
		TextBlock("hi"),
		{Type: BlockToolUse, ID: "c1", Name: "Bash"},
	}
	if got := stripThoughtSignatures(content); &got[0] != &content[0] {
		t.Error("stripThoughtSignatures copied a slice that had nothing to strip")
	}
}

// With parallel tool calls, a signature frame naming index 0 can arrive after
// call 1 has started. Routing by arrival order would put call 0's signature on
// call 1 — a wrong signature on one call and none on the other, which is the
// missing-thought_signature rejection this plumbing exists to prevent.
func TestAccumulatorRoutesSignatureByToolIndex(t *testing.T) {
	acc := NewAccumulator()
	acc.Add(StreamEvent{Type: SEToolUseStart, ToolID: "c0", ToolName: "Read", ToolIndex: 0})
	acc.Add(StreamEvent{Type: SEToolInputJSON, Text: `{"path":"a"}`})
	acc.Add(StreamEvent{Type: SEToolUseStart, ToolID: "c1", ToolName: "Read", ToolIndex: 1})
	acc.Add(StreamEvent{Type: SEToolInputJSON, Text: `{"path":"b"}`})
	// Signatures arrive out of order, after both calls have started.
	acc.Add(StreamEvent{Type: SEToolUseSignature, Text: "sig-1", ToolIndex: 1})
	acc.Add(StreamEvent{Type: SEToolUseSignature, Text: "sig-0", ToolIndex: 0})
	acc.Add(StreamEvent{Type: SEMessageStop})

	msg := acc.Message()
	want := map[string]string{"c0": "sig-0", "c1": "sig-1"}
	seen := 0
	for _, b := range msg.Content {
		if b.Type != BlockToolUse {
			continue
		}
		seen++
		if b.ThoughtSignature != want[b.ID] {
			t.Errorf("%s carries signature %q, want %q", b.ID, b.ThoughtSignature, want[b.ID])
		}
	}
	if seen != 2 {
		t.Fatalf("got %d tool_use blocks, want 2", seen)
	}
}

// Adapters that do not index their tool calls (Anthropic, the Responses API)
// leave ToolIndex at 0, which for them means "the call in flight". The
// fallback must still attach the signature to the most recent call.
func TestAccumulatorSignatureFallsBackToLatestCall(t *testing.T) {
	acc := NewAccumulator()
	acc.Add(StreamEvent{Type: SEToolUseStart, ToolID: "c0", ToolName: "Read"})
	acc.Add(StreamEvent{Type: SEToolInputJSON, Text: `{"path":"a"}`})
	// An index that never started: must not panic, must not be dropped.
	acc.Add(StreamEvent{Type: SEToolUseSignature, Text: "sig-x", ToolIndex: 7})
	acc.Add(StreamEvent{Type: SEMessageStop})

	msg := acc.Message()
	if len(msg.Content) != 1 {
		t.Fatalf("got %d blocks, want 1", len(msg.Content))
	}
	if got := msg.Content[0].ThoughtSignature; got != "sig-x" {
		t.Errorf("signature = %q, want %q", got, "sig-x")
	}
}
