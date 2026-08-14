package compaction

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// recSummarizer records calls and the messages it was handed.
type recSummarizer struct {
	calls int
	last  []llm.Message
	reply string
	err   error
}

func (r *recSummarizer) fn(_ context.Context, msgs []llm.Message) (string, error) {
	r.calls++
	r.last = msgs
	if r.err != nil {
		return "", r.err
	}
	if r.reply == "" {
		return "DIGEST", nil
	}
	return r.reply, nil
}

func boundaryCount(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		if llm.IsBoundaryMarker(m) {
			n++
		}
	}
	return n
}

// --- autoCompact gating ---

func TestAutoCompactNilSummarizer(t *testing.T) {
	c := New(smallWindow(), nil)
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000)}
	if _, ok := c.autoCompact(context.Background(), msgs, 100, "auto"); ok {
		t.Fatal("nil summarizer → autoCompact must not run")
	}
}

func TestAutoCompactTrippedBreaker(t *testing.T) {
	rs := &recSummarizer{}
	c := New(smallWindow(), rs.fn)
	c.tripped = true
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000)}
	if _, ok := c.autoCompact(context.Background(), msgs, 100, "auto"); ok {
		t.Fatal("tripped breaker → autoCompact must not run")
	}
	if rs.calls != 0 {
		t.Fatalf("summarizer should not be called when tripped, calls=%d", rs.calls)
	}
}

func TestAutoCompactTooFewMessages(t *testing.T) {
	rs := &recSummarizer{}
	c := New(smallWindow(), rs.fn) // KeepRecent=2
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000)}
	if _, ok := c.autoCompact(context.Background(), msgs, 100, "auto"); ok {
		t.Fatal("workingBounds empty (tailStart<=start) → must not run")
	}
	if rs.calls != 0 {
		t.Fatalf("summarizer should not be called, calls=%d", rs.calls)
	}
}

func TestAutoCompactFailureThenSuccessResets(t *testing.T) {
	rs := &recSummarizer{err: errors.New("boom")}
	c := New(smallWindow(), rs.fn)
	mk := func() []llm.Message {
		var m []llm.Message
		for range 6 {
			m = append(m, bigMsg(llm.RoleUser, 4000))
		}
		return m
	}
	c.autoCompact(context.Background(), mk(), 100, "auto") // fail 1
	c.autoCompact(context.Background(), mk(), 100, "auto") // fail 2
	if c.failures != 2 || c.tripped {
		t.Fatalf("after 2 failures: failures=%d tripped=%v", c.failures, c.tripped)
	}
	rs.err = nil // now succeeds
	if _, ok := c.autoCompact(context.Background(), mk(), 100, "auto"); !ok {
		t.Fatal("expected success")
	}
	if c.failures != 0 {
		t.Fatalf("success must reset failures, got %d", c.failures)
	}
}

func TestAutoCompactBoundaryMeta(t *testing.T) {
	rs := &recSummarizer{}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)
	var msgs []llm.Message
	for range 6 {
		msgs = append(msgs, bigMsg(llm.RoleUser, 4000))
	}
	out, ok := c.autoCompact(context.Background(), msgs, 4242, "reactive")
	if !ok {
		t.Fatal("expected run")
	}
	i := llm.LastBoundaryIndex(out)
	meta, _ := llm.ParseBoundaryMeta(out[i])
	if meta.Trigger != "reactive" || meta.PreTokens != 4242 {
		t.Fatalf("meta=%+v", meta)
	}
	// messagesSummarized == len of the summarized head (msgs[start:tailStart]).
	if meta.MessagesSummarized != len(msgs)-2 { // KeepRecent=2 preserved
		t.Fatalf("messagesSummarized=%d, want %d", meta.MessagesSummarized, len(msgs)-2)
	}
	if rs.calls != 1 || len(rs.last) != len(msgs)-2 {
		t.Fatalf("summarizer got %d messages, want %d", len(rs.last), len(msgs)-2)
	}
}

func TestAutoCompactMultipleBoundaries(t *testing.T) {
	rs := &recSummarizer{}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)
	msgs := []llm.Message{llm.UserText("FIRST_UNIQUE")}
	for range 6 {
		msgs = append(msgs, bigMsg(llm.RoleUser, 4000))
	}
	out1, ok := c.autoCompact(context.Background(), msgs, 100, "auto")
	if !ok {
		t.Fatal("first compaction failed")
	}
	// Add more turns after the first boundary, then compact again.
	out1 = append(out1, bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000))
	out2, ok := c.autoCompact(context.Background(), out1, 200, "auto")
	if !ok {
		t.Fatal("second compaction failed")
	}
	if boundaryCount(out2) != 2 {
		t.Fatalf("want 2 boundaries, got %d", boundaryCount(out2))
	}
	if rs.calls != 2 {
		t.Fatalf("want 2 summarizer calls, got %d", rs.calls)
	}
	// The second summary must only cover post-first-boundary messages.
	for _, m := range rs.last {
		if strings.Contains(m.Text(), "FIRST_UNIQUE") {
			t.Fatal("second compaction re-summarized pre-boundary history")
		}
	}
}

// --- Pre orchestration matrix ---

func TestPreDisabledWindow(t *testing.T) {
	rs := &recSummarizer{}
	c := New(Config{ContextWindow: -1}, rs.fn)
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000)}
	out := c.Pre(context.Background(), msgs, 999999)
	if len(out) != len(msgs) || rs.calls != 0 {
		t.Fatal("ContextWindow<=0 disables compaction")
	}
}

func TestPreBelowMicroThresholdNoOp(t *testing.T) {
	rs := &recSummarizer{}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000}, rs.fn)
	msgs := manyReadPairs(microKeepRecent + 5)  // has clearable results...
	out := c.Pre(context.Background(), msgs, 1) // ...but the gate says we're tiny
	if boundaryCount(out) != 0 || countCleared(out) != 0 || rs.calls != 0 {
		t.Fatal("below microThreshold → no micro, no auto")
	}
}

// AutoCompact reached purely via the last-response gate on a micro no-op turn
// (no compactable results), proving the accurate last-response size is used for
// the auto decision, not a re-estimate.
func TestPreAutoViaLastInputTokensNoMicro(t *testing.T) {
	rs := &recSummarizer{}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)
	var msgs []llm.Message
	for range 10 {
		msgs = append(msgs, llm.UserText("small")) // no compactable tools
	}
	out := c.Pre(context.Background(), msgs, 200000) // above autoThreshold (179000)
	if boundaryCount(out) != 1 {
		t.Fatalf("auto should fire from the last-response gate, boundaries=%d", boundaryCount(out))
	}
	if rs.calls != 1 {
		t.Fatalf("summarizer calls=%d, want 1", rs.calls)
	}
}

// MicroCompact-only band: above micro, below auto, not predictive → clears results
// but inserts no boundary.
func TestPreMicroOnlyBand(t *testing.T) {
	rs := &recSummarizer{}
	// eff=96000, micro=57600, auto=83000, growth=19000 → predictive at tok>77000.
	c := New(Config{ContextWindow: 100000, MaxOutputTokens: 4000, KeepRecent: 2}, rs.fn)
	msgs := manyReadPairs(microKeepRecent + 1) // small real size, one clearable
	out := c.Pre(context.Background(), msgs, 60000)
	if rs.calls != 0 || boundaryCount(out) != 0 {
		t.Fatalf("micro-only band should not autoCompact (calls=%d boundaries=%d)", rs.calls, boundaryCount(out))
	}
	if countCleared(out) != 1 {
		t.Fatalf("micro should clear the oldest result, cleared=%d", countCleared(out))
	}
}

func TestPreAutoNilSummarizerNoBoundary(t *testing.T) {
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, nil)
	var msgs []llm.Message
	for range 10 {
		msgs = append(msgs, llm.UserText("m"+strconv.Itoa(len(msgs))))
	}
	out := c.Pre(context.Background(), msgs, 200000)
	if boundaryCount(out) != 0 {
		t.Fatal("nil summarizer → no boundary even above autoThreshold")
	}
}

func TestPreNonDestructive(t *testing.T) {
	rs := &recSummarizer{}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)
	var msgs []llm.Message
	for range 10 {
		msgs = append(msgs, llm.UserText("keep"))
	}
	out := c.Pre(context.Background(), msgs, 200000)
	// Full history retained + boundary + summary.
	if len(out) != len(msgs)+2 {
		t.Fatalf("len(out)=%d, want %d", len(out), len(msgs)+2)
	}
	if EstimateTokens(out) <= EstimateTokens(llm.MessagesForAPI(out)) {
		t.Fatal("retained history should exceed the API view")
	}
}

// --- Reactive edges ---

func TestReactiveTripped(t *testing.T) {
	rs := &recSummarizer{}
	c := New(smallWindow(), rs.fn)
	c.tripped = true
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000)}
	if _, ok := c.Reactive(context.Background(), msgs); ok {
		t.Fatal("tripped → reactive cannot compact")
	}
}

// If the summary is larger than the original working set, Reactive reports it did
// not shrink (ok=false) even though a boundary was inserted.
func TestReactiveNoShrinkWhenSummaryHuge(t *testing.T) {
	rs := &recSummarizer{reply: strings.Repeat("z", 200000)}
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 8000, KeepRecent: 2}, rs.fn)
	var msgs []llm.Message
	for range 6 {
		msgs = append(msgs, bigMsg(llm.RoleUser, 400))
	}
	out, ok := c.Reactive(context.Background(), msgs)
	if ok {
		t.Fatal("a huge summary means the API view did not shrink → ok=false")
	}
	if boundaryCount(out) != 1 {
		t.Fatalf("autoCompact still inserts a boundary, got %d", boundaryCount(out))
	}
}
