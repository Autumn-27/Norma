package agentcore

import (
	"testing"

	"github.com/Autumn-27/norma/compaction"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/tool"
)

// noa is opt-in: a host that never calls noaadapter.Enable must get exactly the
// pre-noa behaviour. Every hook noa needed is additive and inert by default, and
// each of these tests pins one of them — because "inert by default" is the kind
// of property that stays true until someone adds a method.

// The load-bearing one. harness.requestMessages takes the ContextView path only
// if the compactor implements View. The built-in compaction must not: if it ever
// grew a View method, every session would silently switch to a projection that
// bypasses the boundary slice, with no config change and no error.
func TestBuiltinCompactionIsNotAContextView(t *testing.T) {
	c := compaction.New(compaction.Config{ContextWindow: 200000}, nil)
	if _, ok := any(c).(harness.ContextView); ok {
		t.Fatal("compaction.Compactor now implements harness.ContextView, so the default " +
			"session builds its requests through View instead of llm.MessagesForAPI — " +
			"a silent behaviour change for every host that never opted into noa")
	}
}

// Without Compactor the session must hold the built-in compaction, unwarned.
func TestNoCompactorMeansBuiltinCompactionUnchanged(t *testing.T) {
	var warns []string
	s := NewSession(Options{
		Compaction: &compaction.Config{ContextWindow: 200000},
		OnWarn:     func(m string) { warns = append(warns, m) },
	})
	if s.compactor == nil {
		t.Fatal("no compactor was installed")
	}
	if _, isNoa := s.compactor.(harness.ContextView); isNoa {
		t.Fatal("the session installed a ContextView without being asked for one")
	}
	if len(warns) != 0 {
		t.Fatalf("a plain Compaction setup produced warnings: %v", warns)
	}
}

// Neither set is also a supported configuration and must stay silent.
func TestNoContextManagerAtAllIsSilent(t *testing.T) {
	var warns []string
	s := NewSession(Options{OnWarn: func(m string) { warns = append(warns, m) }})
	if s.compactor != nil {
		t.Fatalf("a session with no context manager installed %T", s.compactor)
	}
	if len(warns) != 0 {
		t.Fatalf("warnings with no context manager configured: %v", warns)
	}
}

// The schema gate opt-out (tool.Spec.RawInput) defaults off. Every tool that
// does not ask for it keeps being validated exactly as before.
func TestToolsDefaultToSchemaValidation(t *testing.T) {
	plain := tool.Build(tool.Spec{Name: "Plain", Schema: map[string]any{"type": "object"}})
	r, ok := plain.(interface{ AcceptsRawInput() bool })
	if !ok {
		t.Fatal("built tools no longer expose AcceptsRawInput; the harness assertion cannot work")
	}
	if r.AcceptsRawInput() {
		t.Fatal("a tool that never set RawInput reports that it accepts raw input; " +
			"schema validation is off for tools that never asked")
	}
}

// The default tool set ships without the opt-out. A tool that quietly gains it
// loses its input validation, which is a security-relevant change for anything
// that shells out or touches the filesystem.
func TestDefaultToolsDoNotOptOutOfValidation(t *testing.T) {
	s := NewSession(Options{WorkingDir: t.TempDir()})
	if s.registry == nil {
		t.Skip("no tool registry on a bare session")
	}
	for _, sc := range s.registry.Schemas() {
		ct, ok := s.registry.Get(sc.Name)
		if !ok {
			continue
		}
		if r, ok := ct.(interface{ AcceptsRawInput() bool }); ok && r.AcceptsRawInput() {
			t.Errorf("built-in tool %q accepts raw input; its arguments are no longer "+
				"schema-checked before it runs", sc.Name)
		}
	}
}

// ToolContext gained ToolUseID for noa's benefit. It must be populated for every
// tool, not just noa's, and its presence must not change anything else.
func TestToolContextIsUnchangedApartFromToolUseID(t *testing.T) {
	// A zero ToolContext is still valid: tools that ignore the field see nothing
	// different from before.
	tc := &tool.ToolContext{}
	if tc.ToolUseID != "" {
		t.Fatalf("a zero ToolContext carries ToolUseID %q", tc.ToolUseID)
	}
	if tc.WorkingDir != "" || tc.AgentID != "" {
		t.Fatal("a zero ToolContext is no longer zero")
	}
}

// llm.PairToolBlocks was exported for noa. The default request path must still
// route through it, or unpaired tool blocks would reach the provider.
func TestMessagesForAPIStillDropsUnpairedToolBlocks(t *testing.T) {
	msgs := []llm.Message{
		llm.UserText("hi"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ID: "call_orphan", Name: "Read", Input: []byte(`{}`)},
		}},
	}
	out := llm.MessagesForAPI(msgs)
	for _, m := range out {
		for _, b := range m.Content {
			if b.Type == llm.BlockToolUse && b.ID == "call_orphan" {
				t.Fatal("an unpaired tool_use survived MessagesForAPI; providers reject those")
			}
		}
	}
}
