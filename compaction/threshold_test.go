package compaction

import (
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// raw builds a Compactor with the given cfg WITHOUT New's defaulting, so tests
// can exercise the exact threshold math (incl. the reserved==0 guard).
func raw(cw, mo int) *Compactor {
	return &Compactor{cfg: Config{ContextWindow: cw, MaxOutputTokens: mo}}
}

func TestReservedForSummaryTable(t *testing.T) {
	cases := []struct {
		mo   int
		want int
	}{
		{0, 20000},     // unset → full reservation
		{1, 1},         // below cap → passthrough
		{8000, 8000},   // typical
		{19999, 19999}, // just below cap
		{20000, 20000}, // at cap
		{20001, 20000}, // above cap → clamped
		{64000, 20000}, // well above cap → clamped
	}
	for _, tc := range cases {
		if got := raw(200000, tc.mo).reservedForSummary(); got != tc.want {
			t.Errorf("reservedForSummary(mo=%d)=%d, want %d", tc.mo, got, tc.want)
		}
	}
}

func TestEffectiveWindowTable(t *testing.T) {
	cases := []struct{ cw, mo, want int }{
		{200000, 8000, 192000},
		{200000, 32000, 180000}, // reserved clamped to 20000
		{200000, 0, 180000},     // reserved==0 guard → 20000
		{16000, 1000, 15000},
	}
	for _, tc := range cases {
		if got := raw(tc.cw, tc.mo).effectiveWindow(); got != tc.want {
			t.Errorf("effectiveWindow(cw=%d,mo=%d)=%d, want %d", tc.cw, tc.mo, got, tc.want)
		}
	}
}

// buffer tiers off the EFFECTIVE window, not the raw context window. These cases
// straddle the 400k and 800k boundaries by exactly ±1 token of effective window
// (reserved fixed at 20000).
func TestBufferTierBoundaries(t *testing.T) {
	cases := []struct {
		cw, want int
		eff      int
	}{
		{419999, 13000, 399999}, // just below 400k tier
		{420000, 30000, 400000}, // exactly 400k
		{819999, 30000, 799999}, // just below 800k tier
		{820000, 50000, 800000}, // exactly 800k
	}
	for _, tc := range cases {
		c := raw(tc.cw, 20000)
		if c.effectiveWindow() != tc.eff {
			t.Fatalf("precondition: effectiveWindow=%d, want %d", c.effectiveWindow(), tc.eff)
		}
		if got := c.buffer(); got != tc.want {
			t.Errorf("buffer(eff=%d)=%d, want %d", tc.eff, got, tc.want)
		}
	}
}

func TestAutoAndMicroThresholds(t *testing.T) {
	c := raw(200000, 8000) // eff=192000, buffer=13000
	if got, want := c.autoThreshold(), 179000; got != want {
		t.Errorf("autoThreshold=%d, want %d", got, want)
	}
	if got, want := c.microThreshold(), 115200; got != want { // 192000*6/10
		t.Errorf("microThreshold=%d, want %d", got, want)
	}
}

func TestMaxTurnGrowthTable(t *testing.T) {
	cases := []struct{ mo, want int }{
		{8000, 23000},  // 8000 + 15000
		{20000, 35000}, // 20000 + 15000
		{64000, 35000}, // clamped reserved 20000 + 15000
		{0, 35000},     // reserved==0 guard → 20000 + 15000
		{19999, 34999}, // 19999 + 15000
	}
	for _, tc := range cases {
		if got := raw(200000, tc.mo).maxTurnGrowth(); got != tc.want {
			t.Errorf("maxTurnGrowth(mo=%d)=%d, want %d", tc.mo, got, tc.want)
		}
	}
}

// predictiveOver(tok) == tok + maxTurnGrowth > effectiveWindow. With eff=192000
// and growth=23000, the tipping point is tok > 169000.
func TestPredictiveOverBoundary(t *testing.T) {
	c := raw(200000, 8000)
	cases := []struct {
		tok  int
		want bool
	}{
		{168999, false},
		{169000, false}, // 169000+23000 == 192000, not strictly greater
		{169001, true},
		{500000, true},
		{0, false},
	}
	for _, tc := range cases {
		if got := c.predictiveOver(tc.tok); got != tc.want {
			t.Errorf("predictiveOver(%d)=%v, want %v", tc.tok, got, tc.want)
		}
	}
}

func TestCountPrefersLastInputTokensBranches(t *testing.T) {
	c := New(Config{ContextWindow: 200000}, nil)
	msgs := []llm.Message{llm.UserText(strings.Repeat("x", 40))} // est = 40/4=10 → count = 13
	if got := c.count(msgs, 999); got != 999 {
		t.Errorf("positive lastInputTokens should win: got %d", got)
	}
	est := c.count(msgs, 0)
	if est != 10*4/3 {
		t.Errorf("estimation path: got %d, want %d", est, 10*4/3)
	}
	if neg := c.count(msgs, -5); neg != est {
		t.Errorf("non-positive lastInputTokens should fall back to estimation: got %d, want %d", neg, est)
	}
}

// EstimateTokens counts each block by type: text length, thinking length,
// tool_use name+input, and nested tool_result content; total chars / 4.
func TestEstimateTokensPerBlock(t *testing.T) {
	msgs := []llm.Message{
		llm.UserText("aaaa"), // 4
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.BlockThinking, Thinking: "bbbbbbbb"},                    // 8
			{Type: llm.BlockToolUse, Name: "Read", Input: []byte(`{"k":123}`)}, // 4 + 9 = 13
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			{Type: llm.BlockToolResult, ToolUseID: "x", Content: []llm.ContentBlock{llm.TextBlock("cccc")}}, // 4
		}},
	}
	// total chars = 4 + 8 + 13 + 4 = 29 → 29/4 = 7
	if got := EstimateTokens(msgs); got != 7 {
		t.Fatalf("EstimateTokens=%d, want 7", got)
	}
}

func TestEstimateTokensEmpty(t *testing.T) {
	if got := EstimateTokens(nil); got != 0 {
		t.Fatalf("EstimateTokens(nil)=%d, want 0", got)
	}
	if got := EstimateTokens([]llm.Message{{Role: llm.RoleUser}}); got != 0 {
		t.Fatalf("EstimateTokens(empty content)=%d, want 0", got)
	}
}

// count operates on the API view: only messages after the last boundary, with
// the boundary marker itself excluded.
func TestCountUsesMessagesForAPI(t *testing.T) {
	c := New(Config{ContextWindow: 200000}, nil)
	msgs := []llm.Message{
		llm.UserText(strings.Repeat("a", 4000)), // pre-boundary, must NOT be counted
		llm.BoundaryMessage(llm.BoundaryMeta{}),
		llm.UserText("bbbb"), // 4 chars → est 1 → count 1
	}
	if got := c.count(msgs, 0); got != 1 {
		t.Fatalf("count should ignore pre-boundary history: got %d, want 1", got)
	}
}
