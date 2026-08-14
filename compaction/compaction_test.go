package compaction

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

func bigMsg(role llm.Role, n int) llm.Message {
	return llm.Message{Role: role, Content: []llm.ContentBlock{llm.TextBlock(strings.Repeat("x", n))}}
}

// smallWindow makes thresholds tiny so a few messages exceed them.
func smallWindow() Config {
	return Config{ContextWindow: 16000, MaxOutputTokens: 1000, KeepRecent: 2}
}

func TestAutoCompactIsNonDestructive(t *testing.T) {
	c := New(smallWindow(), func(_ context.Context, msgs []llm.Message) (string, error) {
		return "DIGEST", nil
	})
	var msgs []llm.Message
	for i := 0; i < 8; i++ {
		msgs = append(msgs, bigMsg(llm.RoleUser, 4000))
	}
	out, ok := c.autoCompact(context.Background(), msgs, 1234, "auto")
	if !ok {
		t.Fatal("autoCompact did not run")
	}
	// Non-destructive: full original history retained + boundary + summary.
	if len(out) != len(msgs)+2 {
		t.Fatalf("len(out)=%d, want %d (history + boundary + summary)", len(out), len(msgs)+2)
	}
	if llm.LastBoundaryIndex(out) < 0 {
		t.Fatal("no boundary marker inserted")
	}
	// API view = summary + recent tail (boundary stripped).
	api := llm.MessagesForAPI(out)
	if !strings.Contains(api[0].Text(), "DIGEST") {
		t.Fatalf("API view should start with the summary, got %q", api[0].Text())
	}
	if len(api) != 1+smallWindow().KeepRecent {
		t.Fatalf("API view = summary + %d recent, got %d", smallWindow().KeepRecent, len(api))
	}
	// Retained history (full array) far exceeds what is actually sent.
	if EstimateTokens(out) <= EstimateTokens(api) {
		t.Fatal("retained history should exceed the API view")
	}
}

func TestAutoCompactReinjectsInvokedSkills(t *testing.T) {
	c := New(smallWindow(), func(_ context.Context, msgs []llm.Message) (string, error) {
		return "DIGEST", nil
	})
	// An old skill invocation, then enough bulk to push it behind the boundary.
	skillMsg := llm.SkillInvocationMessage("deploy", "/skills/deploy", "1. build\n2. ship")
	msgs := []llm.Message{skillMsg}
	for i := 0; i < 8; i++ {
		msgs = append(msgs, bigMsg(llm.RoleUser, 4000))
	}
	out, ok := c.autoCompact(context.Background(), msgs, 1234, "auto")
	if !ok {
		t.Fatal("autoCompact did not run")
	}
	// The skill instructions must appear in the API view (post-boundary), verbatim,
	// even though the transcript copy was summarized to "DIGEST".
	api := llm.MessagesForAPI(out)
	var found bool
	for _, m := range api {
		if name, ok := llm.SkillInvocationName(m); ok && name == "deploy" && strings.Contains(m.Text(), "1. build") {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("invoked skill 'deploy' was not re-injected into the post-boundary API view")
	}
}

func TestReinjectSkillsDedupesAndSkipsTail(t *testing.T) {
	// Two invocations of the same skill behind the boundary → keep only the latest.
	old := llm.SkillInvocationMessage("x", "", "OLD BODY")
	newer := llm.SkillInvocationMessage("x", "", "NEW BODY")
	// A different skill whose only invocation is in the preserved tail → not re-injected.
	tail := llm.SkillInvocationMessage("y", "", "TAIL BODY")
	msgs := []llm.Message{old, newer, bigMsg(llm.RoleUser, 10), tail}
	tailStart := 3 // msgs[3:] is the preserved tail

	got := reinjectSkills(msgs, tailStart)
	if len(got) != 1 {
		t.Fatalf("want 1 re-injected skill (latest x, y skipped), got %d", len(got))
	}
	if name, _ := llm.SkillInvocationName(got[0]); name != "x" {
		t.Fatalf("want skill x, got %q", name)
	}
	if !strings.Contains(got[0].Text(), "NEW BODY") || strings.Contains(got[0].Text(), "OLD BODY") {
		t.Fatalf("should keep the latest invocation body, got %q", got[0].Text())
	}
}

func TestReinjectSkillsTruncatesToBudget(t *testing.T) {
	huge := strings.Repeat("z", maxSkillReinjectTokens*4*3) // ~3x the per-skill char cap
	msgs := []llm.Message{
		llm.SkillInvocationMessage("big", "", huge),
		bigMsg(llm.RoleUser, 10),
	}
	got := reinjectSkills(msgs, 1)
	if len(got) != 1 {
		t.Fatalf("want 1 re-injected skill, got %d", len(got))
	}
	if EstimateTokens(got) > maxSkillReinjectTokens+50 {
		t.Fatalf("re-injected skill not truncated to per-skill budget: %d tokens", EstimateTokens(got))
	}
	if !strings.Contains(got[0].Text(), "truncated") {
		t.Fatal("truncated skill should carry a truncation note")
	}
}

func TestCircuitBreakerTrips(t *testing.T) {
	c := New(smallWindow(), func(context.Context, []llm.Message) (string, error) {
		return "", errors.New("boom")
	})
	mk := func() []llm.Message {
		var m []llm.Message
		for i := 0; i < 6; i++ {
			m = append(m, bigMsg(llm.RoleUser, 4000))
		}
		return m
	}
	for i := 0; i < 4; i++ {
		c.autoCompact(context.Background(), mk(), 1000, "auto")
	}
	if !c.tripped {
		t.Fatal("breaker should trip after 3 failures")
	}
}

// resultText returns the concatenated text of a tool_result block's content.
func resultText(b llm.ContentBlock) string {
	var s strings.Builder
	for _, cb := range b.Content {
		s.WriteString(cb.Text)
	}
	return s.String()
}

// readPair builds an assistant Read tool_use + its user tool_result.
func toolPair(name, id, body string) []llm.Message {
	return []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockToolUse, ID: id, Name: name, Input: []byte(`{}`)}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResultText(id, body, false)}},
	}
}

// TestMicroCompactKeepsRecentClearsOld verifies the keep-last-N semantics: with
// microKeepRecent (5) recent compactable results and older ones beyond that, only
// the older results are cleared.
func TestMicroCompactKeepsRecentClearsOld(t *testing.T) {
	c := New(Config{ContextWindow: 16000, MaxOutputTokens: 1000}, nil)
	var msgs []llm.Message
	total := microKeepRecent + 2 // 2 older than the kept window
	for i := 0; i < total; i++ {
		msgs = append(msgs, toolPair("Read", "r"+strconv.Itoa(i), strings.Repeat("y", 500))...)
	}
	out, changed := c.microCompact(msgs)
	if !changed {
		t.Fatal("microCompact should report a change")
	}
	// The first two Read results (oldest) must be cleared; the last five kept.
	cleared := 0
	for i, m := range out {
		if len(m.Content) == 1 && m.Content[0].Type == llm.BlockToolResult {
			if strings.Contains(resultText(m.Content[0]), "cleared") {
				cleared++
			} else if i < 2*2 { // one of the two oldest result messages
				t.Fatalf("oldest result at %d should be cleared", i)
			}
		}
	}
	if cleared != 2 {
		t.Fatalf("want 2 old results cleared, got %d", cleared)
	}
}

// TestMicroCompactSkipsNonCompactableTools verifies non-compactable tool results
// are never cleared, even when older than the kept window.
func TestMicroCompactSkipsNonCompactableTools(t *testing.T) {
	c := New(Config{ContextWindow: 16000, MaxOutputTokens: 1000}, nil)
	var msgs []llm.Message
	msgs = append(msgs, toolPair("Agent", "a0", strings.Repeat("y", 5000))...) // oldest, non-compactable
	for i := 0; i < microKeepRecent+1; i++ {
		msgs = append(msgs, toolPair("Read", "r"+strconv.Itoa(i), strings.Repeat("y", 500))...)
	}
	out, _ := c.microCompact(msgs)
	if strings.Contains(resultText(out[1].Content[0]), "cleared") {
		t.Fatal("Agent (non-compactable) result should NOT be cleared")
	}
}

func TestThresholdMath(t *testing.T) {
	// reserved = min(maxOutputTokens, 20000) = 8000.
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000}, nil)
	if c.effectiveWindow() != 200000-8000 {
		t.Fatalf("effectiveWindow=%d, want 192000", c.effectiveWindow())
	}
	if c.buffer() != 13000 {
		t.Fatalf("buffer=%d", c.buffer())
	}
	if c.autoThreshold() != 192000-13000 {
		t.Fatalf("autoThreshold=%d, want 179000", c.autoThreshold())
	}
	// Large window → larger buffer.
	big := New(Config{ContextWindow: 1000000, MaxOutputTokens: 32000}, nil)
	if big.buffer() != 50000 || big.reservedForSummary() != 20000 {
		t.Fatalf("big: buffer=%d reserved=%d", big.buffer(), big.reservedForSummary())
	}
}

func TestIsOverflow(t *testing.T) {
	if !IsOverflow(errors.New("anthropic: status 400: prompt is too long")) {
		t.Fatal("should detect prompt too long")
	}
	if IsOverflow(errors.New("network timeout")) {
		t.Fatal("false positive")
	}
}
