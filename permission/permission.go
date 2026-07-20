// Package permission is the agent's safety boundary. It models tool-call
// decisions (allow / deny / ask), six permission modes, allow/deny rule lists
// with deny-priority, and a host-provided interactive callback. The four-stage
// pipeline in Evaluate combines a tool's own CheckPermissions verdict with
// policy and mode to produce a final decision.
package permission

import (
	"context"
	"encoding/json"
	"strings"
)

// Behavior is the kind of a permission decision.
type Behavior string

const (
	Allow Behavior = "allow"
	Deny  Behavior = "deny"
	Ask   Behavior = "ask"
)

// Update is a persistence suggestion ("always allow this") surfaced to the host
// (FR-05.6).
type Update struct {
	Type string `json:"type"` // e.g. "addAllowRule"
	Rule string `json:"rule"` // e.g. "Bash(git:*)"
}

// Decision is the outcome of a permission check.
type Decision struct {
	Behavior     Behavior
	UpdatedInput json.RawMessage // optional rewritten input (e.g. stripped flag)
	Message      string          // reason, shown to model on deny / to user on ask
	Suggestions  []Update        // optional "always allow" suggestions
}

// Allowed / Denied / AskUser build common decisions.
func Allowed() Decision          { return Decision{Behavior: Allow} }
func Denied(msg string) Decision { return Decision{Behavior: Deny, Message: msg} }
func AskUser(msg string) Decision {
	return Decision{Behavior: Ask, Message: msg}
}

// Mode is a permission mode (FR-05.4).
type Mode string

const (
	ModeDefault     Mode = "default"           // ask for non-read-only unless a rule allows
	ModeAcceptEdits Mode = "acceptEdits"       // auto-allow file edits/writes
	ModeBypass      Mode = "bypassPermissions" // allow everything (except deny rules)
	ModePlan        Mode = "plan"              // read-only only
	ModeDontAsk     Mode = "dontAsk"           // never prompt; rules + read-only only
	ModeAuto        Mode = "auto"              // like dontAsk but allows tool's own allow verdict
)

// Context is passed to a tool's CheckPermissions and the interactive callback.
type Context struct {
	Mode       Mode
	WorkingDir string
	Allowed    []string
	Disallowed []string
}

// CanUseTool is the host's interactive approval callback (CLI prompt, GUI, or
// automated policy). It is consulted only when the pipeline reaches the "ask"
// stage (FR-05.1, C-02).
type CanUseTool func(ctx context.Context, toolName string, input json.RawMessage, pc Context) Decision

// editTools are auto-allowed under ModeAcceptEdits.
var editTools = map[string]bool{"Write": true, "Edit": true, "NotebookEdit": true}

// Request bundles everything Evaluate needs without importing the tool package.
type Request struct {
	ToolName     string
	Input        json.RawMessage
	IsReadOnly   bool
	ToolDecision Decision // verdict from the tool's own CheckPermissions
	Ctx          Context
	Ask          CanUseTool
}

// Evaluate runs the four-stage pipeline (FR-05.3) and returns the final
// decision. Order encodes deny-priority (FR-05.5): an explicit deny rule wins
// over every mode, including bypass.
func Evaluate(ctx context.Context, req Request) Decision {
	// Stage 1 — deny rules (absolute priority).
	if matchRule(req.Ctx.Disallowed, req.ToolName) {
		return Denied("denied: tool '" + req.ToolName + "' is disallowed by policy")
	}
	// A tool that hard-denies (e.g. failed self-validation) is also absolute.
	if req.ToolDecision.Behavior == Deny {
		return req.ToolDecision
	}

	// Stage 2 — mode short-circuits.
	switch req.Ctx.Mode {
	case ModeBypass:
		return Allowed()
	case ModePlan:
		if !req.IsReadOnly {
			return Denied("denied: plan mode permits read-only tools only")
		}
		return Allowed()
	}

	// Stage 3 — allow rules and mode-based auto-allows.
	if matchRule(req.Ctx.Allowed, req.ToolName) {
		return Allowed()
	}
	if req.Ctx.Mode == ModeAcceptEdits && editTools[req.ToolName] {
		return Allowed()
	}
	if req.IsReadOnly {
		return Allowed()
	}
	if req.ToolDecision.Behavior == Allow {
		return req.ToolDecision
	}

	// Stage 4 — interactive confirmation (only if mode permits prompting).
	switch req.Ctx.Mode {
	case ModeDontAsk:
		return Denied("denied: '" + req.ToolName + "' requires approval but mode is dontAsk")
	case ModeAuto:
		return Denied("denied: '" + req.ToolName + "' not permitted by rules in auto mode")
	}
	if req.Ask != nil {
		d := req.Ask(ctx, req.ToolName, req.Input, req.Ctx)
		if d.Behavior == "" {
			return Denied("denied by approval callback")
		}
		return d
	}
	return Denied("denied: no approval handler configured for '" + req.ToolName + "'")
}

// matchRule reports whether toolName matches any rule. A rule is either a bare
// tool name, "*" (any), or "Name(...)" whose Name part matches.
func matchRule(rules []string, toolName string) bool {
	for _, r := range rules {
		r = strings.TrimSpace(r)
		if r == "*" || r == toolName {
			return true
		}
		if i := strings.IndexByte(r, '('); i > 0 && r[:i] == toolName {
			return true
		}
	}
	return false
}
