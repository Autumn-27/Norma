package agentcore

import (
	"context"
	"iter"
	"testing"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

// usageProvider bills every request the same amount and ends the turn with text.
type usageProvider struct{ in, out int }

func (p usageProvider) Stream(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: "ok"}, nil) {
			return
		}
		yield(llm.StreamEvent{
			Type:       llm.SEMessageDelta,
			StopReason: "end_turn",
			Usage:      llm.Usage{InputTokens: p.in, OutputTokens: p.out},
		}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func (p usageProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	acc := llm.NewAccumulator()
	for ev, err := range p.Stream(ctx, req) {
		if err != nil {
			return llm.Message{}, "", llm.Usage{}, err
		}
		acc.Add(ev)
	}
	return acc.Message(), acc.StopReason, acc.Usage, nil
}

func drain(t *testing.T, seq iter.Seq2[harness.Event, error]) {
	t.Helper()
	for _, err := range seq {
		if err != nil {
			t.Fatalf("prompt: %v", err)
		}
	}
}

// Usage is the session total, not the last turn's. Reset() is the only thing
// that zeroes it, and Resume() restores it, so assigning per turn would silently
// discard everything a caller had already paid for.
func TestUsageAccumulatesAcrossPrompts(t *testing.T) {
	sess := NewSession(Options{
		Provider:     usageProvider{in: 100, out: 10},
		SystemPrompt: []string{"sys"},
	})

	drain(t, sess.Prompt(context.Background(), "one"))
	if got := sess.Usage(); got.InputTokens != 100 || got.OutputTokens != 10 {
		t.Fatalf("after one prompt: %+v, want 100/10", got)
	}

	drain(t, sess.Prompt(context.Background(), "two"))
	if got := sess.Usage(); got.InputTokens != 200 || got.OutputTokens != 20 {
		t.Fatalf("after two prompts: %+v, want 200/20 (a running total)", got)
	}

	sess.Reset()
	if got := sess.Usage(); got != (llm.Usage{}) {
		t.Fatalf("after Reset: %+v, want zero", got)
	}
}
