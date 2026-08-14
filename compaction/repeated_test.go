package compaction

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// Simulates a long-running conversation that repeatedly grows past the auto
// threshold: each cycle appends several turns, then Pre is called with the model
// reporting an over-threshold context. Every cycle must insert a NEW boundary,
// the API view (what is actually sent) must stay bounded, and the retained
// history must keep growing (non-destructive).
func TestPreRepeatedCompactionOverGrowingConversation(t *testing.T) {
	rs := &recSummarizer{reply: "SUMMARY"}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)

	var msgs []llm.Message
	const cycles = 6
	for iter := range cycles {
		for j := range 5 { // conversation grows by 5 turns
			msgs = append(msgs, llm.UserText(fmt.Sprintf("turn-%d-%d", iter, j)))
		}
		// The model reports we are above the auto threshold (179000).
		msgs = c.Pre(context.Background(), msgs, 200000)

		if got := boundaryCount(msgs); got != iter+1 {
			t.Fatalf("cycle %d: want %d boundaries, got %d", iter, iter+1, got)
		}
		if api := llm.MessagesForAPI(msgs); len(api) > 8 {
			t.Fatalf("cycle %d: API view unbounded (%d messages)", iter, len(api))
		}
	}

	if rs.calls != cycles {
		t.Fatalf("want %d summarizer calls (one per cycle), got %d", cycles, rs.calls)
	}
	// Full history retained on disk-array; only a small slice reaches the model.
	if len(msgs) <= len(llm.MessagesForAPI(msgs)) {
		t.Fatal("retained history should far exceed the API view")
	}
	// The most recent boundary's summary is what leads the API view.
	if api := llm.MessagesForAPI(msgs); !strings.Contains(api[0].Text(), "SUMMARY") {
		t.Fatalf("API view should start with the latest summary, got %q", api[0].Text())
	}
}

// Same repeated-compaction property, but driven purely by ESTIMATION (no
// last-response hint): genuinely large messages push the estimated size past the
// threshold each cycle, so compaction must keep firing on its own.
func TestPreRepeatedCompactionViaEstimation(t *testing.T) {
	rs := &recSummarizer{reply: "S"}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)

	// autoThreshold = 179000 → count = chars/3 must exceed it (chars > ~537000).
	// Append 4 big turns per cycle (200000 chars each) so the post-boundary window
	// re-crosses the threshold AND exceeds KeepRecent (2) so a head remains.
	big := func(tag string) llm.Message { return llm.UserText(tag + strings.Repeat("x", 200000)) }

	var msgs []llm.Message
	prev := 0
	for iter := range 4 {
		for j := range 4 {
			msgs = append(msgs, big(fmt.Sprintf("c%d-%d", iter, j)))
		}
		msgs = c.Pre(context.Background(), msgs, 0) // no last-response hint → estimation

		got := boundaryCount(msgs)
		if got <= prev {
			t.Fatalf("cycle %d: estimation-driven compaction did not fire (boundaries stayed %d)", iter, got)
		}
		prev = got
		// The API view must stay under the threshold after each compaction.
		if tok := c.count(msgs, 0); tok >= c.autoThreshold() {
			t.Fatalf("cycle %d: API view still above autoThreshold after compaction: %d", iter, tok)
		}
	}
	if prev == 0 {
		t.Fatal("expected estimation-driven compaction to fire at least once")
	}
}
