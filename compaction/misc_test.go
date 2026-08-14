package compaction

import (
	"context"
	"errors"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

func TestIsOverflowStrings(t *testing.T) {
	cases := []struct {
		msg  string
		want bool
	}{
		{"", false}, // nil handled separately below
		{"anthropic: status 400: prompt is too long", true},
		{"context length exceeded", true},
		{"maximum context window reached", true},
		{"error: context_length_exceeded", true},
		{"anthropic: status 413", true},
		{"network timeout", false},
		{"rate limited", false},
	}
	for _, tc := range cases {
		if got := IsOverflow(errors.New(tc.msg)); got != tc.want {
			t.Errorf("IsOverflow(%q)=%v, want %v", tc.msg, got, tc.want)
		}
	}
	if IsOverflow(nil) {
		t.Fatal("IsOverflow(nil) must be false")
	}
}

func TestIsOverflowMethodForm(t *testing.T) {
	c := New(Config{ContextWindow: 200000}, nil)
	if !c.IsOverflow(errors.New("prompt is too long")) {
		t.Fatal("method form should detect overflow")
	}
	if c.IsOverflow(errors.New("boom")) {
		t.Fatal("method form false positive")
	}
}

// New applies all defaults for a zero Config: 200000 window, 8000 output,
// 8 recent messages — verified through the derived thresholds and geometry.
func TestDefaultsApplied(t *testing.T) {
	c := New(Config{}, func(context.Context, []llm.Message) (string, error) { return "D", nil })
	if c.effectiveWindow() != 192000 { // 200000 - 8000
		t.Fatalf("effectiveWindow=%d, want 192000 (defaulted window/output)", c.effectiveWindow())
	}
	if c.autoThreshold() != 179000 {
		t.Fatalf("autoThreshold=%d, want 179000", c.autoThreshold())
	}
	// KeepRecent defaulted to 8: summarizing 10 messages preserves the last 8.
	var msgs []llm.Message
	for range 10 {
		msgs = append(msgs, bigMsg(llm.RoleUser, 10))
	}
	out, ok := c.autoCompact(context.Background(), msgs, 100, "auto")
	if !ok {
		t.Fatal("autoCompact should run")
	}
	meta, _ := llm.ParseBoundaryMeta(out[llm.LastBoundaryIndex(out)])
	if meta.MessagesSummarized != 2 { // 10 - 8 kept
		t.Fatalf("messagesSummarized=%d, want 2 (KeepRecent defaulted to 8)", meta.MessagesSummarized)
	}
}
