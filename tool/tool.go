// Package tool defines the CoreTool contract and the built-in file/shell tools
// that give the agent its hands. Every tool is self-describing: a name and
// description for the model, a JSON-Schema input, behavioral flags used by the
// scheduler (read-only / concurrency-safe), a permission self-check, and a Call
// method. New tools are built through Build, which applies fail-closed defaults.
package tool

import (
	"context"
	"encoding/json"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
)

// ProgressInfo is an incremental progress update emitted during a long tool run.
type ProgressInfo struct {
	Message string
}

// ToolContext carries per-call execution state into a tool.
type ToolContext struct {
	WorkingDir string
	AgentID    string
	// ToolUseID is the id of the tool_use block this call answers. Tools that
	// need to refer back to their own call in the conversation — to anchor
	// something to it, or to recognise it later — read it here.
	ToolUseID      string
	MaxOutputChars int
	// OutputDir, when set, makes oversized tool output spill to a file there
	// (full content preserved) instead of being discarded by truncation; the
	// tool returns a head + a pointer to the file. Empty disables spilling.
	OutputDir string
	// Emit reports progress; may be nil. Safe to call from the tool goroutine.
	Emit func(ProgressInfo)
	// Tasks is the background-task manager for this session; may be nil (background
	// execution disabled). Tools use it to spawn, read, and stop background work.
	Tasks *Manager
	// Env holds extra "KEY=VALUE" entries injected into the environment of Bash
	// subprocesses (both foreground and background). Applied on top of the parent
	// process environment, so these override inherited values. Empty = inherit
	// unchanged. Used e.g. to route child-command HTTP through a proxy + trust a
	// CA, without affecting the agent's own process.
	Env []string
}

func (tc *ToolContext) progress(msg string) {
	if tc != nil && tc.Emit != nil {
		tc.Emit(ProgressInfo{Message: msg})
	}
}

// Result is the outcome of a tool call. Content is the tool_result payload fed
// back to the model.
type Result struct {
	Content []llm.ContentBlock
	IsError bool
	// Extra holds additional messages the tool injects into the conversation
	// right after its tool_result (e.g. the Skill tool emits the invoked skill's
	// instructions as an independent user message rather than burying them in the
	// tool_result). Appended in order; nil for ordinary tools.
	Extra []llm.Message
}

// Text builds a successful text result.
func Text(s string) Result { return Result{Content: []llm.ContentBlock{llm.TextBlock(s)}} }

// Errorf builds an error result.
func Errorf(s string) Result {
	return Result{Content: []llm.ContentBlock{llm.TextBlock(s)}, IsError: true}
}

// Flatten concatenates the text of a result's content blocks.
func (r Result) Flatten() string {
	var s string
	for _, b := range r.Content {
		if b.Type == llm.BlockText {
			s += b.Text
		}
	}
	return s
}

// CoreTool is a capability the agent can invoke (FR-04.1).
type CoreTool interface {
	Name() string
	Description() string
	Prompt() string
	InputSchema() map[string]any
	IsReadOnly(input json.RawMessage) bool
	IsConcurrencySafe(input json.RawMessage) bool
	CheckPermissions(ctx context.Context, input json.RawMessage, pc permission.Context) permission.Decision
	Call(ctx context.Context, input json.RawMessage, tc *ToolContext) (Result, error)
}

// Spec declares a tool. Build fills unset behavioral fields with fail-closed
// defaults (FR-04.2): not read-only, not concurrency-safe, requires approval.
type Spec struct {
	Name        string
	Description string
	Prompt      string
	Schema      map[string]any
	ReadOnly    func(json.RawMessage) bool
	Concurrent  func(json.RawMessage) bool
	Permissions func(ctx context.Context, input json.RawMessage, pc permission.Context) permission.Decision
	Run         func(ctx context.Context, input json.RawMessage, tc *ToolContext) (Result, error)

	// RawInput hands Run whatever the model produced, skipping the schema check
	// the harness otherwise applies first.
	//
	// Schema stays required and is still advertised to the model — the two are
	// separate jobs. Schema tells the model what to send; validation decides what
	// gets through. A tool that repairs malformed arguments itself needs the
	// first and is defeated by the second: a truncated or fence-wrapped call is
	// not valid JSON, so it would be rejected before Run ever saw it, and the
	// repair could never run.
	//
	// Only set this on a tool that treats its input as untrusted bytes and
	// answers every shape with a Result rather than an error.
	RawInput bool
}

// Build constructs a CoreTool from a Spec, applying fail-closed defaults.
func Build(s Spec) CoreTool { return &builtTool{spec: s} }

type builtTool struct{ spec Spec }

func (t *builtTool) Name() string                { return t.spec.Name }
func (t *builtTool) Description() string         { return t.spec.Description }
func (t *builtTool) Prompt() string              { return t.spec.Prompt }
func (t *builtTool) InputSchema() map[string]any { return t.spec.Schema }

// AcceptsRawInput reports whether this tool parses its own arguments.
//
// Deliberately not on CoreTool: that interface is public, and adding a method
// would break every host implementing it. The harness asks for it with an
// optional type assertion, the same way it discovers ContextView.
func (t *builtTool) AcceptsRawInput() bool { return t.spec.RawInput }

func (t *builtTool) IsReadOnly(in json.RawMessage) bool {
	if t.spec.ReadOnly == nil {
		return false
	}
	return t.spec.ReadOnly(in)
}

func (t *builtTool) IsConcurrencySafe(in json.RawMessage) bool {
	if t.spec.Concurrent == nil {
		return false
	}
	return t.spec.Concurrent(in)
}

func (t *builtTool) CheckPermissions(ctx context.Context, in json.RawMessage, pc permission.Context) permission.Decision {
	if t.spec.Permissions == nil {
		return permission.AskUser("approve " + t.spec.Name + "?")
	}
	return t.spec.Permissions(ctx, in, pc)
}

func (t *builtTool) Call(ctx context.Context, in json.RawMessage, tc *ToolContext) (Result, error) {
	return t.spec.Run(ctx, in, tc)
}

// readOnlyTrue / concurrentTrue are shared helpers for read tools.
func always(json.RawMessage) bool { return true }

// allowReadOnly is a permission self-check that auto-allows (used by read tools;
// the pipeline also auto-allows read-only, this makes intent explicit).
func allowReadOnly(context.Context, json.RawMessage, permission.Context) permission.Decision {
	return permission.Allowed()
}
