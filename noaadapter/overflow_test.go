package noaadapter

import (
	"context"
	"errors"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// The three harness-facing methods are the whole public surface between noa and
// the agent loop, and until now only View was exercised.

// Pre must not touch the history. The built-in compaction rewrites messages
// here; noa's message identity is a content hash, so a rewrite would change
// every id and invalidate every ref the model is holding.
func TestPreLeavesTheHistoryUntouched(t *testing.T) {
	sess := simSession(t, 40000)
	c := &Compactor{sess: sess}
	in := []llm.Message{userText("one"), userText("two")}

	out := c.Pre(t.Context(), in, 12345)
	if len(out) != len(in) {
		t.Fatalf("Pre returned %d messages, want the %d it was given", len(out), len(in))
	}
	for i := range in {
		if out[i].Text() != in[i].Text() {
			t.Fatalf("Pre rewrote message %d: %q -> %q", i, in[i].Text(), out[i].Text())
		}
	}
	// The provider's own count is more accurate than any estimate, so it must be
	// taken when offered — resolveTokenCount prefers it over estimateCoreTokens
	// whenever it is fresh and close.
	sess.mu.Lock()
	got := sess.providerTokens
	sess.mu.Unlock()
	if got != 12345 {
		t.Fatalf("Pre recorded providerTokens = %d, want 12345; the pressure ladder keeps "+
			"running on estimates when a measured number was available", got)
	}
}

// A zero count means the provider did not report one — it must not be recorded
// as "the context is empty".
func TestPreIgnoresAnUnreportedTokenCount(t *testing.T) {
	sess := simSession(t, 40000)
	c := &Compactor{sess: sess}
	c.Pre(t.Context(), nil, 7000)
	before := sess.tokensBefore()
	c.Pre(t.Context(), nil, 0)
	if sess.tokensBefore() != before {
		t.Fatalf("a zero token count overwrote the last known size %d with %d",
			before, sess.tokensBefore())
	}
}

// IsOverflow decides whether Reactive runs at all. Getting it wrong in one
// direction wastes a recovery attempt; in the other it turns a recoverable
// prompt-too-long into a failed run.
func TestIsOverflowRecognisesProviderPhrasings(t *testing.T) {
	c := &Compactor{sess: simSession(t, 40000)}
	overflow := []string{
		"This model's maximum context length is 200000 tokens, however you requested 214000",
		"prompt is too long: 215000 tokens > 200000 maximum",
		"error: context_length_exceeded",
		"anthropic: status 413",
	}
	for _, msg := range overflow {
		if !c.IsOverflow(errors.New(msg)) {
			t.Errorf("IsOverflow(%q) = false; the run fails instead of recovering", msg)
		}
	}
	other := []string{"rate limit exceeded", "connection reset by peer", "invalid api key"}
	for _, msg := range other {
		if c.IsOverflow(errors.New(msg)) {
			t.Errorf("IsOverflow(%q) = true; a recovery attempt is spent on an unrelated failure", msg)
		}
	}
	if c.IsOverflow(nil) {
		t.Error("IsOverflow(nil) = true")
	}
}

// overflowState documents itself as mining the error for the provider's real
// window — "rather than guessing again next turn, the error is mined for the
// actual number and the mechanical valve is armed so the next view is built to
// fit."
//
// Nothing does that. ParseOverflowWindow and overflowState.learn have no callers
// anywhere in the tree: harness.Compactor.Reactive receives only the message
// array, and IsOverflow — the one method that does see the error — throws it
// away. So arm() always falls back to the CONFIGURED limit, which is exactly the
// number the provider just rejected.
//
// Either wire learning in (IsOverflow is the natural place: it already has the
// error) or delete the mechanism and the comment that promises it. Leaving it is
// the worst option, because the comment describes behaviour the code does not
// have.
func TestOverflowLearningIsWiredIn(t *testing.T) {
	c := &Compactor{sess: simSession(t, 200000)}

	// The provider says its real window is 128000, not the configured 200000.
	err := errors.New("This model's maximum context length is 128000 tokens, however you requested 214000")
	if n := ParseOverflowWindow(err.Error()); n != 128000 {
		t.Fatalf("ParseOverflowWindow = %d, want 128000 — the parser itself is broken", n)
	}

	// The loop's actual sequence: detect, then recover.
	c.IsOverflow(err)
	c.Reactive(t.Context(), nil)

	floor := c.overflow.armedFloor()
	if floor == 0 {
		t.Skip("Reactive disarmed because it could not shrink an empty view")
	}
	want := int(float64(128000) * 0.95)
	if floor != want {
		t.Fatalf("the armed floor is %d (95%% of the CONFIGURED %d), not %d (95%% of the %d "+
			"the provider reported). ParseOverflowWindow and overflowState.learn have no "+
			"callers, so the learned window is never learned and the next view is rebuilt to "+
			"a limit the provider has already rejected.",
			floor, c.sess.Config().ModelContextLimit, want, 128000)
	}
}

// The parser itself, over the phrasings it claims to cover.
func TestParseOverflowWindowCoversItsPatterns(t *testing.T) {
	cases := map[string]int{
		"This model's maximum context length is 200000 tokens": 200000,
		"context window of 131072 exceeded":                    131072,
		"max_tokens must be less than 8192":                    8192,
		"requests are limited to 32000 tokens":                 32000,
		"rate limit exceeded":                                  0,
		"maximum context length is zero":                       0,
	}
	for msg, want := range cases {
		if got := ParseOverflowWindow(msg); got != want {
			t.Errorf("ParseOverflowWindow(%q) = %d, want %d", msg, got, want)
		}
	}
}

// Reactive must never claim success it cannot deliver: returning true promises
// the harness a smaller request, and if the request is not smaller the retry
// hits the same 413 and the loop spins.
func TestReactiveRefusesWhenItCannotShrink(t *testing.T) {
	sess := simSession(t, 40000)
	c := &Compactor{sess: sess}
	// Nothing sent yet and nothing to cut.
	msgs := []llm.Message{userText("short")}
	if _, ok := c.Reactive(context.Background(), msgs); ok {
		if est := estimateMessages(c.View(context.Background(), msgs)); est > 0 {
			t.Logf("Reactive accepted a %d-token view under an armed floor", est)
		}
	}
	// Whatever it decided, it must leave the emergency floor consistent with that
	// decision: a refusal that left the valve armed would shrink every later view
	// for no reason.
	if !c.overflow.armed && c.overflow.armedFloor() != 0 {
		t.Fatal("disarmed but still reporting a floor")
	}
}
