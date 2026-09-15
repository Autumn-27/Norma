package harness

import (
	"context"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// noopCompactor implements Compactor but NOT ContextView: the loop must fall
// back to llm.MessagesForAPI, i.e. behaviour is unchanged.
type noopCompactor struct{}

func (noopCompactor) Pre(_ context.Context, msgs []llm.Message, _ int) []llm.Message { return msgs }
func (noopCompactor) Reactive(_ context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	return msgs, false
}
func (noopCompactor) IsOverflow(error) bool { return false }

// viewCompactor implements both Compactor and ContextView. It records what it
// was handed and returns a projection, so the test can assert that the loop
// uses the returned slice and never writes it back.
type viewCompactor struct {
	noopCompactor
	sawLen  int
	calls   int
	project func([]llm.Message) []llm.Message
}

func (v *viewCompactor) View(_ context.Context, msgs []llm.Message) []llm.Message {
	v.calls++
	v.sawLen = len(msgs)
	return v.project(msgs)
}

func TestRequestMessagesFallsBackWithoutContextView(t *testing.T) {
	msgs := []llm.Message{
		llm.UserText("a"),
		llm.BoundaryMessage(llm.BoundaryMeta{Trigger: "auto"}),
		llm.UserText("b"),
	}
	l := &loop{ctx: context.Background(), messages: msgs, in: QueryInput{Compactor: noopCompactor{}}}

	got := l.requestMessages()
	want := llm.MessagesForAPI(msgs)
	if len(got) != len(want) {
		t.Fatalf("requestMessages() len = %d, want %d (MessagesForAPI fallback)", len(got), len(want))
	}
	for i := range got {
		if got[i].Text() != want[i].Text() {
			t.Fatalf("requestMessages()[%d] = %q, want %q", i, got[i].Text(), want[i].Text())
		}
	}
}

func TestRequestMessagesUsesContextView(t *testing.T) {
	msgs := []llm.Message{llm.UserText("a"), llm.UserText("b"), llm.UserText("c")}
	v := &viewCompactor{project: func(in []llm.Message) []llm.Message {
		return []llm.Message{llm.UserText("projected")}
	}}
	l := &loop{ctx: context.Background(), messages: msgs, in: QueryInput{Compactor: v}}

	got := l.requestMessages()
	if v.calls != 1 {
		t.Fatalf("View called %d times, want 1", v.calls)
	}
	if v.sawLen != 3 {
		t.Fatalf("View saw %d messages, want the full history (3)", v.sawLen)
	}
	if len(got) != 1 || got[0].Text() != "projected" {
		t.Fatalf("requestMessages() = %+v, want the View projection", got)
	}
	// The projection must not be written back: l.messages is the durable history
	// and message identity (content hashing) depends on it staying byte-stable.
	if len(l.messages) != 3 {
		t.Fatalf("l.messages len = %d after View, want 3 (View must not write back)", len(l.messages))
	}
}

// A ContextView bypasses the boundary slice entirely — it owns the projection.
func TestContextViewBypassesBoundarySlicing(t *testing.T) {
	msgs := []llm.Message{
		llm.UserText("pre-boundary"),
		llm.BoundaryMessage(llm.BoundaryMeta{Trigger: "auto"}),
		llm.UserText("post-boundary"),
	}
	v := &viewCompactor{project: func(in []llm.Message) []llm.Message { return in }}
	l := &loop{ctx: context.Background(), messages: msgs, in: QueryInput{Compactor: v}}

	got := l.requestMessages()
	if len(got) != 3 {
		t.Fatalf("requestMessages() len = %d, want 3 — a ContextView receives the raw history, not the post-boundary slice", len(got))
	}
	if got[0].Text() != "pre-boundary" {
		t.Fatalf("requestMessages()[0] = %q, want the pre-boundary message to be visible to View", got[0].Text())
	}
}

func TestRequestMessagesWithNilCompactor(t *testing.T) {
	msgs := []llm.Message{llm.UserText("a")}
	l := &loop{ctx: context.Background(), messages: msgs, in: QueryInput{}}
	if got := l.requestMessages(); len(got) != 1 || got[0].Text() != "a" {
		t.Fatalf("requestMessages() with nil Compactor = %+v, want the MessagesForAPI fallback", got)
	}
}
