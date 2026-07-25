package compaction

import (
	"context"
	"errors"
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

func TestMicroCompactClearsCompactableToolResults(t *testing.T) {
	c := New(Config{ContextWindow: 16000, MaxOutputTokens: 1000, KeepRecent: 1, ToolResultBudget: 100}, nil)
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockToolUse, ID: "r1", Name: "Read", Input: []byte(`{}`)}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResultText("r1", strings.Repeat("y", 5000), false)}},
		bigMsg(llm.RoleAssistant, 10), // recent, preserved
	}
	out := c.microCompact(msgs)
	got := flatten(out[1].Content[0].Content)
	if !strings.Contains(got, "cleared") {
		t.Fatalf("compactable tool result not cleared: %q", got)
	}
}

func TestMicroCompactSkipsNonCompactableTools(t *testing.T) {
	c := New(Config{ContextWindow: 16000, MaxOutputTokens: 1000, KeepRecent: 1, ToolResultBudget: 100}, nil)
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockToolUse, ID: "a1", Name: "Agent", Input: []byte(`{}`)}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResultText("a1", strings.Repeat("y", 5000), false)}},
		bigMsg(llm.RoleAssistant, 10),
	}
	out := c.microCompact(msgs)
	if strings.Contains(flatten(out[1].Content[0].Content), "cleared") {
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
