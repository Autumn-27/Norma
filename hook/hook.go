// Package hook provides Go-native lifecycle hooks (FR-08): callbacks fired at
// eight points in the agent loop that can observe state, block or rewrite tool
// calls, inject context, and gate the Stop transition. A Registry implements
// harness.HookRunner, so wiring it into a Session needs no glue.
package hook

import (
	"context"
	"encoding/json"

	"github.com/Autumn-27/norma/llm"
)

// EventType enumerates the eight hook points (FR-08.2).
type EventType string

const (
	PreToolUse         EventType = "PreToolUse"
	PostToolUse        EventType = "PostToolUse"
	PostToolUseFailure EventType = "PostToolUseFailure"
	UserPromptSubmit   EventType = "UserPromptSubmit"
	SessionStart       EventType = "SessionStart"
	SessionEnd         EventType = "SessionEnd"
	Stop               EventType = "Stop"
	SubagentStart      EventType = "SubagentStart"
)

// Event is the payload passed to a hook. Which fields are set depends on Type.
type Event struct {
	Type      EventType
	ToolName  string
	Input     json.RawMessage
	Result    json.RawMessage
	IsError   bool
	Prompt    string
	AgentType string
	Messages  []llm.Message
}

// Result is a hook's response (FR-08.3).
type Result struct {
	// Decision is "block" to halt the action, "approve"/"" to continue.
	Decision string
	// Message explains a block (shown to the model).
	Message string
	// UpdatedInput optionally rewrites a tool's input (PreToolUse).
	UpdatedInput json.RawMessage
	// AdditionalContext is injected into the conversation (UserPromptSubmit, Stop).
	AdditionalContext string
	// PreventContinuation, on a Stop hook, hard-terminates the loop
	// (stop_hook_prevented) even if other logic would continue. A "block"
	// decision instead forces the loop to continue with AdditionalContext
	// injected (stop_hook_blocking).
	PreventContinuation bool
}

func (r Result) blocked() bool { return r.Decision == "block" }

// Func is a single hook callback.
type Func func(ctx context.Context, ev Event) Result

// Registry holds hooks by event type and implements harness.HookRunner.
type Registry struct {
	byType map[EventType][]Func
}

// NewRegistry builds an empty registry.
func NewRegistry() *Registry { return &Registry{byType: map[EventType][]Func{}} }

// On registers a hook for an event type.
func (r *Registry) On(t EventType, f Func) *Registry {
	r.byType[t] = append(r.byType[t], f)
	return r
}

func (r *Registry) run(ctx context.Context, t EventType, ev Event) []Result {
	ev.Type = t
	var out []Result
	for _, f := range r.byType[t] {
		out = append(out, f(ctx, ev))
	}
	return out
}

// --- harness.HookRunner implementation ---

// PreToolUse runs PreToolUse hooks. The first block halts; the last non-empty
// UpdatedInput wins.
func (r *Registry) PreToolUse(ctx context.Context, name string, input []byte) (bool, string, []byte) {
	var updated []byte
	for _, res := range r.run(ctx, PreToolUse, Event{ToolName: name, Input: input}) {
		if res.blocked() {
			return true, res.Message, nil
		}
		if len(res.UpdatedInput) > 0 {
			updated = res.UpdatedInput
		}
	}
	return false, "", updated
}

// PostToolUse runs PostToolUse hooks (and PostToolUseFailure on error).
func (r *Registry) PostToolUse(ctx context.Context, name string, input, result []byte, isErr bool) {
	ev := Event{ToolName: name, Input: input, Result: result, IsError: isErr}
	r.run(ctx, PostToolUse, ev)
	if isErr {
		r.run(ctx, PostToolUseFailure, ev)
	}
}

// Stop runs Stop hooks and reports two distinct outcomes: a hook may
// hard-terminate the loop (preventContinuation), or it
// may force the loop to continue by emitting blocking errors injected as
// messages (FR-08.4). preventContinuation wins if any hook sets it.
func (r *Registry) Stop(ctx context.Context, messages []llm.Message) (prevent bool, blockingErrors []string, message string) {
	for _, res := range r.run(ctx, Stop, Event{Messages: messages}) {
		if res.PreventContinuation {
			return true, nil, res.Message
		}
		if res.blocked() {
			msg := res.AdditionalContext
			if msg == "" {
				msg = res.Message
			}
			blockingErrors = append(blockingErrors, msg)
		}
	}
	return false, blockingErrors, ""
}

// --- fired by agentcore / subagent ---

// FireUserPromptSubmit runs UserPromptSubmit hooks and returns concatenated
// additional context to inject (empty if none).
func (r *Registry) FireUserPromptSubmit(ctx context.Context, prompt string) string {
	var ctxAdd string
	for _, res := range r.run(ctx, UserPromptSubmit, Event{Prompt: prompt}) {
		if res.AdditionalContext != "" {
			ctxAdd += res.AdditionalContext + "\n"
		}
	}
	return ctxAdd
}

// FireSessionStart / FireSessionEnd are fire-and-forget notifications.
func (r *Registry) FireSessionStart(ctx context.Context) { r.run(ctx, SessionStart, Event{}) }
func (r *Registry) FireSessionEnd(ctx context.Context)   { r.run(ctx, SessionEnd, Event{}) }

// FireSubagentStart notifies that a subagent is starting.
func (r *Registry) FireSubagentStart(ctx context.Context, agentType string) {
	r.run(ctx, SubagentStart, Event{AgentType: agentType})
}
