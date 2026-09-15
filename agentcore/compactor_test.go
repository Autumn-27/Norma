package agentcore

import (
	"context"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/compaction"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

type stubCompactor struct{}

func (stubCompactor) Pre(_ context.Context, msgs []llm.Message, _ int) []llm.Message { return msgs }
func (stubCompactor) Reactive(_ context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	return msgs, false
}
func (stubCompactor) IsOverflow(error) bool { return false }

// Compactor and Compaction are mutually exclusive context managers. The session
// holds exactly one, so they can never both run.
func TestCompactorWinsOverCompaction(t *testing.T) {
	var warns []string
	s := NewSession(Options{
		Compactor:  stubCompactor{},
		Compaction: &compaction.Config{ContextWindow: 200000},
		OnWarn:     func(m string) { warns = append(warns, m) },
	})
	if s.compactor == nil {
		t.Fatal("session has no compactor")
	}
	if _, ok := s.compactor.(stubCompactor); !ok {
		t.Fatalf("compactor = %T, want the host-supplied stubCompactor (Compaction must be ignored)", s.compactor)
	}
	if len(warns) != 1 {
		t.Fatalf("OnWarn called %d times, want 1 for the Compactor+Compaction clash: %v", len(warns), warns)
	}
	if !strings.Contains(warns[0], "Compaction is IGNORED") {
		t.Fatalf("warning = %q, want it to say Compaction is ignored", warns[0])
	}
}

// Setting only one of them is not a misconfiguration — no warning.
func TestNoWarnWhenOnlyOneContextManager(t *testing.T) {
	for name, opts := range map[string]Options{
		"compactor only":  {Compactor: stubCompactor{}},
		"compaction only": {Compaction: &compaction.Config{ContextWindow: 200000}},
		"neither":         {},
	} {
		var warns []string
		o := opts
		o.OnWarn = func(m string) { warns = append(warns, m) }
		NewSession(o)
		if len(warns) != 0 {
			t.Fatalf("%s: OnWarn called with %v, want no warnings", name, warns)
		}
	}
}

// The SDK must stay silent when the host did not ask for warnings.
func TestNilOnWarnIsSafe(t *testing.T) {
	s := NewSession(Options{
		Compactor:  stubCompactor{},
		Compaction: &compaction.Config{ContextWindow: 200000},
	})
	if _, ok := s.compactor.(stubCompactor); !ok {
		t.Fatalf("compactor = %T, want stubCompactor", s.compactor)
	}
}

func TestCompactorOnly(t *testing.T) {
	s := NewSession(Options{Compactor: stubCompactor{}})
	if _, ok := s.compactor.(stubCompactor); !ok {
		t.Fatalf("compactor = %T, want stubCompactor", s.compactor)
	}
}

// Without Compactor the built-in compaction is used — the pre-noa behaviour.
func TestCompactionUsedWhenNoCompactor(t *testing.T) {
	s := NewSession(Options{Compaction: &compaction.Config{ContextWindow: 200000}})
	if s.compactor == nil {
		t.Fatal("Compaction set but session has no compactor")
	}
	if _, ok := s.compactor.(*compaction.Compactor); !ok {
		t.Fatalf("compactor = %T, want *compaction.Compactor", s.compactor)
	}
}

func TestNoContextManagerByDefault(t *testing.T) {
	s := NewSession(Options{})
	if s.compactor != nil {
		t.Fatalf("compactor = %T, want nil when neither Compactor nor Compaction is set", s.compactor)
	}
}

// A plain Compactor must not accidentally satisfy ContextView: hosts opt into
// taking over the projection by implementing the extra method.
func TestPlainCompactorIsNotContextView(t *testing.T) {
	var c harness.Compactor = stubCompactor{}
	if _, ok := c.(harness.ContextView); ok {
		t.Fatal("stubCompactor unexpectedly satisfies harness.ContextView")
	}
	var built harness.Compactor = compaction.New(compaction.Config{}, nil)
	if _, ok := built.(harness.ContextView); ok {
		t.Fatal("compaction.Compactor must not satisfy harness.ContextView — its Pre rewrites messages in place")
	}
}
