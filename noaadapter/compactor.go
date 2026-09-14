package noaadapter

import (
	"context"

	"github.com/Autumn-27/norma/compaction"
	"github.com/Autumn-27/norma/llm"
)

// Compactor adapts a Session to the harness's context-manager interfaces.
//
// It satisfies harness.Compactor and harness.ContextView. Implementing the
// latter is what takes over request construction from llm.MessagesForAPI.
type Compactor struct {
	sess *Session
	// overflow tracks what a prompt-too-long error taught us.
	overflow overflowState
}

// Pre is a no-op on the history.
//
// The built-in compaction rewrites messages here; noa must not. Message
// identity is a content hash over the stored text, and a rewrite would change
// every id derived from it, invalidating every ref the model holds.
func (c *Compactor) Pre(_ context.Context, msgs []llm.Message, lastInputTokens int) []llm.Message {
	if lastInputTokens > 0 {
		c.sess.noteProviderTokens(lastInputTokens)
	}
	return msgs
}

// View builds the request's message array.
func (c *Compactor) View(_ context.Context, msgs []llm.Message) []llm.Message {
	return c.sess.View(msgs)
}

// IsOverflow reports whether err is a context-overflow rejection. The built-in
// compaction already recognises every provider's phrasing, so the detection is
// borrowed rather than duplicated.
func (c *Compactor) IsOverflow(err error) bool {
	return compaction.New(compaction.Config{}, nil).IsOverflow(err)
}

// Reactive recovers from a prompt-too-long error.
//
// noa cannot ask the model to compress here — the model is not in the loop at
// this point. What it can do is arm the mechanical valve and VERIFY the result:
// returning true without actually shrinking the view would produce a
// 413 → retry → 413 loop, which is worse than a clean failure. So the view is
// rebuilt and measured, and false is returned if it did not shrink.
func (c *Compactor) Reactive(ctx context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	before := c.sess.tokensBefore()
	c.overflow.arm(c.sess.Config().ModelContextLimit)
	c.sess.setEmergencyFloor(c.overflow.armedFloor())

	after := estimateMessages(c.View(ctx, msgs))
	if before > 0 && after >= before {
		c.sess.setEmergencyFloor(0)
		c.overflow.disarm()
		return msgs, false
	}
	return msgs, true
}

// estimateMessages sizes a rebuilt request.
func estimateMessages(msgs []llm.Message) int {
	total := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			total += len(b.Text) / 4
			total += len(b.Thinking) / 4
			total += len(b.Input) / 4
			for _, inner := range b.Content {
				total += len(inner.Text) / 4
			}
		}
	}
	return total
}
