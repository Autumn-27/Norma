// Package agentcore is the public entry point of the SDK. It aggregates the
// provider, tools, permission policy, and hooks into a stateful Session whose
// Prompt method streams the agent loop as a range-over-func iterator. The SDK
// injects no preset system prompt: the host controls it entirely via
// Options.SystemPrompt (design decision 1).
package agentcore

import (
	"context"
	"errors"
	"iter"
	"strings"
	"time"

	"github.com/Autumn-27/norma/compaction"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/memory"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/plan"
	"github.com/Autumn-27/norma/skill"
	"github.com/Autumn-27/norma/tool"
	"github.com/Autumn-27/norma/transcript"
)

// Options configures a Session (FR-07).
type Options struct {
	// Provider is the LLM backend (any Anthropic- or OpenAI-format endpoint, or
	// a custom implementation).
	Provider llm.Provider

	// SystemPrompt is the host-controlled system prompt, in segments. The SDK
	// injects nothing of its own. AppendSystemPrompt is added after it (e.g. for
	// SDK-managed dynamic context). DynamicBoundary marks the cacheable prefix.
	SystemPrompt       []string
	AppendSystemPrompt []string
	DynamicBoundary    int

	// Tools the agent may call. Defaults to tool.DefaultTools().
	Tools []tool.CoreTool

	// DeferredTools names tools whose schemas are withheld from the model (only
	// their names surface, e.g. via tool.RenderDeferredToolsBlock placed in the
	// system prompt). They stay callable through the auto-added ExecuteExtraTool
	// after discovery via SearchExtraTools. When non-empty the SDK auto-adds those
	// two access tools. Empty = classic behavior (all tool schemas sent).
	DeferredTools []string
	// UnlockSet gates which deferred tools are currently callable via
	// ExecuteExtraTool. Nil + non-empty DeferredTools → the SDK seeds one with all
	// DeferredTools unlocked. Hosts doing skill-gated disclosure pass their own set
	// (seeded with the globally-available subset) and Add() to it when a skill loads.
	UnlockSet *tool.UnlockSet

	// Permission policy.
	CanUseTool      permission.CanUseTool
	AllowedTools    []string
	DisallowedTools []string
	PermissionMode  permission.Mode

	// Hooks is optional lifecycle instrumentation (Phase 3).
	Hooks harness.HookRunner

	// Compaction, when non-nil, enables automatic context-window management,
	// summarizing with the same Provider.
	Compaction *compaction.Config

	// Plan, when non-nil, enables plan mode (read-only exploration → approval →
	// implementation) with EnterPlanMode/ExitPlanMode tools.
	Plan *PlanOptions

	// Memory, when non-nil, enables persistent long-term memory with
	// RecallMemory/RecordMemory tools and optional auto-injection.
	Memory *MemoryOptions

	// Skills, when non-nil, exposes a Skill tool for invoking named procedures.
	Skills *skill.Registry

	// EnableWebFetch adds the network-facing WebFetch tool (permission-gated).
	EnableWebFetch bool
	// WebFetchProxy, when set, routes WebFetch requests through this http(s) proxy
	// URL (e.g. a recording/intercepting proxy). Empty = direct connection.
	WebFetchProxy string
	// WebFetchCACert is a path to a PEM CA cert WebFetch adds to the system trust
	// pool — verifies HTTPS through a MITM proxy without disabling TLS (preferred
	// over WebFetchInsecureTLS).
	WebFetchCACert string
	// WebFetchInsecureTLS skips TLS verification for WebFetch (curl -k). Blunt
	// fallback when no CA cert is available; prefer WebFetchCACert.
	WebFetchInsecureTLS bool
	// EnableWebSearch adds the network-facing web_search tool (permission-gated).
	EnableWebSearch bool
	// WebSearchBackend selects the search backend: "ddgs" (DuckDuckGo, no key),
	// "brave-free" (Brave Search API), or "tavily" (Tavily Search API). Empty
	// defaults to "ddgs".
	WebSearchBackend string
	// BraveSearchAPIKey authenticates the "brave-free" backend. Required only when
	// WebSearchBackend == "brave-free".
	BraveSearchAPIKey string
	// TavilySearchAPIKey authenticates the "tavily" backend. Required only when
	// WebSearchBackend == "tavily".
	TavilySearchAPIKey string
	// WebSearchProxy, when set, routes web_search requests through this http(s)
	// proxy URL — useful when the search endpoint is only reachable via a proxy.
	WebSearchProxy string
	// WebSearchCACert / WebSearchInsecureTLS mirror the WebFetch TLS options for
	// web_search when it runs through a MITM/recording proxy (prefer the CA cert).
	WebSearchCACert      string
	WebSearchInsecureTLS bool
	// Todos, when non-nil, adds the TodoWrite tool backed by this store, letting
	// the host observe the agent's plan/progress via Todos.List().
	Todos *tool.TodoStore
	// AskUser, when non-nil, adds the AskUserQuestion tool so the agent can pause
	// a long task to ask the user a structured multiple-choice question.
	AskUser tool.AskUserFunc

	// SystemReminderFunc, when set, is called before each Prompt and its return
	// value is included in the <system-reminder> prepended to the conversation.
	// The SDK always injects today's date into the reminder regardless of whether
	// this is set. Return "" to include only the date.
	SystemReminderFunc func() string

	MaxTurns int
	// MaxDuration bounds wall-clock run time, checked at the turn boundary (never
	// mid-stream), so a run ends cleanly rather than being aborted in flight (0 = off).
	MaxDuration time.Duration
	// Settlement, when set with a Prompt, runs a bounded wrap-up phase when a budget
	// (MaxTurns/MaxDuration) is hit — injecting the prompt and hiding DisabledTools —
	// so a budget-terminated run checkpoints its findings. See harness.Settlement.
	Settlement     *harness.Settlement
	MaxTokens      int
	Temperature    *float64
	MaxConcurrency int
	WorkingDir     string

	// ToolOutputDir, when set, makes oversized tool output spill to a file there
	// (full content preserved) and the tool return a head + pointer instead of
	// discarding the overflow (tool.Capture). MaxToolOutputChars caps the head
	// size fed to the model (0 = default 30000).
	ToolOutputDir      string
	MaxToolOutputChars int

	// BashEnv holds extra "KEY=VALUE" entries injected into the environment of
	// Bash subprocesses (foreground + background), overriding inherited values.
	// The agent's own process env is untouched. Empty = inherit unchanged. Used
	// e.g. to route child-command HTTP through a proxy and trust its CA.
	BashEnv []string

	// Transcript, when non-nil, persists the conversation to an append-only
	// JSONL log so the session can be resumed (see Session.Resume). Subagents
	// write their own sidechain files under the same store.
	Transcript *transcript.Store
	// SessionID names the transcript file. Empty generates a fresh id (read it
	// back via Session.SessionID).
	SessionID string

	// TokenBudget, when > 0, lets the agent keep working (with budget nudges)
	// after it would otherwise stop, until ~90% of the output-token budget or
	// diminishing returns.
	TokenBudget int
	// EscalateMaxTokens retries a max_tokens truncation with a raised output cap
	// before falling back to resume-recovery.
	EscalateMaxTokens bool

	// DisableBackgroundTasks, when true, prevents the SDK from injecting the
	// background-task management tools (TaskOutput, TaskStop, TaskList, Monitor)
	// into this session's registry. Use for focused single-purpose agents (e.g.
	// goal decomposers, structured-output callers) that should only see the tools
	// explicitly provided in Options.Tools. The global AGENT_CORE_DISABLE_BACKGROUND_TASKS
	// env var disables them for every session; this field is the per-session equivalent.
	DisableBackgroundTasks bool

	// Deps optionally overrides side-effect boundaries (testing/advanced use).
	Deps harness.QueryDeps
}

// PlanOptions configures plan mode.
type PlanOptions struct {
	// Approver decides whether a presented plan is accepted. Nil auto-approves.
	Approver plan.Approver
	// StartInPlanMode begins the session read-only (the agent must ExitPlanMode
	// to implement). If false, the agent enters plan mode itself via EnterPlanMode.
	StartInPlanMode bool
	// TargetMode is the permission mode after approval (default acceptEdits).
	TargetMode permission.Mode
}

// MemoryOptions configures long-term memory.
type MemoryOptions struct {
	Store *memory.Store
	// AutoInject prepends memories relevant to each prompt into the conversation.
	AutoInject bool
	// MaxInject caps auto-injected memories (default 3).
	MaxInject int
	// IncludeIndexInPrompt adds the memory index to the system prompt.
	IncludeIndexInPrompt bool
}

// Session is a long-lived conversation. Call Prompt repeatedly; state
// (messages, usage) persists across calls (FR-07.1/2). Not safe for concurrent
// Prompt calls on the same Session.
type Session struct {
	opts      Options
	registry  *tool.Registry
	compactor harness.Compactor
	planCtrl  *plan.Controller
	messages  []llm.Message
	usage     llm.Usage

	sessionID string
	store     *transcript.Store
	writer    *transcript.Writer
	tasks     *tool.Manager
	unlock    *tool.UnlockSet // deferred-tool call gate (nil when no deferred tools)
}

// NewSession builds a Session, applying defaults.
func NewSession(opts Options) *Session {
	tools := opts.Tools
	if tools == nil {
		tools = tool.DefaultTools()
	}
	if opts.PermissionMode == "" {
		opts.PermissionMode = permission.ModeDefault
	}
	s := &Session{opts: opts}
	if opts.Compaction != nil {
		s.compactor = compaction.New(*opts.Compaction, compaction.ProviderSummarizer(opts.Provider))
	}
	if opts.Plan != nil {
		start := opts.PermissionMode
		if opts.Plan.StartInPlanMode {
			start = permission.ModePlan
		}
		s.planCtrl = plan.NewController(start, opts.Plan.TargetMode)
		tools = append(tools, plan.Tools(s.planCtrl, opts.Plan.Approver)...)
	}
	if opts.Memory != nil && opts.Memory.Store != nil {
		tools = append(tools, memory.Tools(opts.Memory.Store)...)
	}
	if opts.Skills != nil {
		tools = append(tools, opts.Skills.Tool())
	}
	if opts.EnableWebFetch {
		tools = append(tools, tool.NewWebFetch(tool.WebFetchConfig{Proxy: opts.WebFetchProxy, CACert: opts.WebFetchCACert, InsecureTLS: opts.WebFetchInsecureTLS}))
	}
	if opts.EnableWebSearch {
		// A misconfigured backend (unknown name, or brave-free without a key)
		// drops the tool rather than failing the whole session — the rest of the
		// agent stays usable and the missing tool surfaces the misconfig.
		if ws, err := tool.NewWebSearch(tool.WebSearchConfig{Backend: opts.WebSearchBackend, BraveAPIKey: opts.BraveSearchAPIKey, TavilyAPIKey: opts.TavilySearchAPIKey, Proxy: opts.WebSearchProxy, CACert: opts.WebSearchCACert, InsecureTLS: opts.WebSearchInsecureTLS}); err == nil {
			tools = append(tools, ws)
		}
	}
	if opts.Todos != nil {
		tools = append(tools, opts.Todos.Tool())
	}
	if opts.AskUser != nil {
		tools = append(tools, tool.NewAskUserQuestion(opts.AskUser))
	}
	// A session id is always needed (it names the background-task temp dir, and
	// the transcript file when persistence is on).
	s.sessionID = opts.SessionID
	if s.sessionID == "" {
		s.sessionID = transcript.NewSessionID()
	}
	// Background task execution: a per-session manager plus the tools that read and
	// control it. Disabled by the global AGENT_CORE_DISABLE_BACKGROUND_TASKS env var
	// or by setting Options.DisableBackgroundTasks for this session only.
	if !tool.BackgroundTasksDisabled() && !opts.DisableBackgroundTasks {
		if mgr, err := tool.NewManager(s.sessionID); err == nil {
			s.tasks = mgr
			tools = append(tools, tool.NewTaskOutput(), tool.NewTaskStop(), tool.NewTaskList(), tool.NewMonitor())
		}
	}
	s.registry = tool.NewRegistry(tools...)
	// Deferred tools: withhold their schemas (harness filters by opts.DeferredTools)
	// but keep them callable via the two always-loaded access tools. Both close over
	// the registry (which already holds the deferred tools) and are added to it.
	if len(opts.DeferredTools) > 0 {
		s.unlock = opts.UnlockSet
		if s.unlock == nil {
			s.unlock = tool.NewUnlockSet(opts.DeferredTools...)
		}
		s.registry.Add(tool.NewSearchExtraTools(s.registry, opts.DeferredTools))
		s.registry.Add(tool.NewExecuteExtraTool(s.registry, s.unlock))
	}
	if opts.Transcript != nil {
		s.store = opts.Transcript
		s.writer = s.store.NewWriter(s.sessionID, "")
	}
	return s
}

// Close releases session resources, killing any background tasks and removing
// their temporary output directory. Safe to call multiple times. Hosts running a
// long-lived Session should defer Close; Run does so automatically.
func (s *Session) Close() error {
	if s.tasks != nil {
		s.tasks.Cleanup()
		s.tasks = nil
	}
	return nil
}

// Messages returns the conversation history.
func (s *Session) Messages() []llm.Message { return s.messages }

// Usage returns cumulative token usage.
func (s *Session) Usage() llm.Usage { return s.usage }

// Reset clears the conversation.
func (s *Session) Reset() { s.messages = nil; s.usage = llm.Usage{} }

// SessionID returns the transcript session id (empty if persistence is off).
func (s *Session) SessionID() string { return s.sessionID }

// UnlockSet returns the session's deferred-tool call gate (nil when no deferred
// tools). Hosts Add() to it to unlock skill-gated tools at runtime.
func (s *Session) UnlockSet() *tool.UnlockSet { return s.unlock }

// Resume reloads a prior conversation from the transcript store, rebuilding the
// message history and cumulative usage so the next Prompt continues it. Requires
// Options.Transcript to be set. Subsequent messages append to the same file.
func (s *Session) Resume(sessionID string) error {
	if s.store == nil {
		return errors.New("agentcore: Resume requires Options.Transcript")
	}
	recs, err := s.store.Load(s.store.MainPath(sessionID))
	if err != nil {
		return err
	}
	msgs, usage := transcript.Messages(recs)
	s.messages = cleanResumed(msgs)
	s.usage = usage
	s.sessionID = sessionID
	s.writer = s.store.ResumeWriter(sessionID, transcript.LastUUID(recs))
	return nil
}

// cleanResumed drops trailing assistant tool calls whose results were never
// persisted — a crash mid-tool would otherwise leave a dangling tool_use that
// the API rejects.
func cleanResumed(msgs []llm.Message) []llm.Message {
	for len(msgs) > 0 {
		last := msgs[len(msgs)-1]
		if last.Role == llm.RoleAssistant && len(last.ToolUses()) > 0 {
			msgs = msgs[:len(msgs)-1]
			continue
		}
		break
	}
	return msgs
}

// buildSystemReminder assembles the <system-reminder> content. Today's date is
// always included; extra is appended when non-empty.
func buildSystemReminder(extra string) string {
	date := time.Now().Format("2006-01-02")
	var b strings.Builder
	b.WriteString("<system-reminder>\n# currentDate\nToday's date is ")
	b.WriteString(date)
	b.WriteString(".")
	if extra != "" {
		b.WriteString("\n")
		b.WriteString(extra)
	}
	b.WriteString("\n</system-reminder>")
	return b.String()
}

func (s *Session) system() []string {
	out := append([]string(nil), s.opts.SystemPrompt...)
	out = append(out, s.opts.AppendSystemPrompt...)
	if s.opts.Memory != nil && s.opts.Memory.Store != nil && s.opts.Memory.IncludeIndexInPrompt {
		out = append(out, s.opts.Memory.Store.SystemSegment())
	}
	return out
}

// hookFirer is the optional host-fired subset of the hook system.
type hookFirer interface {
	FireSessionStart(context.Context)
	FireUserPromptSubmit(context.Context, string) string
}

// Prompt appends a user message and streams the agent loop to completion. The
// final event is KindResult; Session state is updated when the consumer reaches
// it (FR-07.3).
func (s *Session) Prompt(ctx context.Context, input string) iter.Seq2[harness.Event, error] {
	// Auto-inject memories relevant to this prompt (Phase 4).
	if s.opts.Memory != nil && s.opts.Memory.AutoInject && s.opts.Memory.Store != nil {
		max := s.opts.Memory.MaxInject
		if max <= 0 {
			max = 3
		}
		if mems := s.opts.Memory.Store.Relevant(input, max); len(mems) > 0 {
			input = input + "\n\n" + memory.RenderForInjection(mems, time.Now())
		}
	}

	// UserPromptSubmit hooks may inject additional context (FR-08.2).
	if hf, ok := s.opts.Hooks.(hookFirer); ok {
		if len(s.messages) == 0 {
			hf.FireSessionStart(ctx)
		}
		if extra := hf.FireUserPromptSubmit(ctx, input); extra != "" {
			input = input + "\n\n" + extra
		}
	}
	// Surface any background-task completions that landed since the previous turn
	// returned, before the new user message.
	if s.tasks != nil {
		if notes := s.tasks.DrainNotifications(); len(notes) > 0 {
			var b strings.Builder
			for i, n := range notes {
				if i > 0 {
					b.WriteString("\n")
				}
				b.WriteString(n.String())
			}
			nm := llm.UserText(b.String())
			s.messages = append(s.messages, nm)
			if s.writer != nil {
				s.writer.RecordMessage(nm, llm.Usage{})
			}
		}
	}
	userMsg := llm.UserText(input)
	s.messages = append(s.messages, userMsg)
	if s.writer != nil {
		s.writer.RecordMessage(userMsg, llm.Usage{})
		ctx = transcript.WithSessionID(ctx, s.sessionID)
	}
	var reminderExtra string
	if s.opts.SystemReminderFunc != nil {
		reminderExtra = s.opts.SystemReminderFunc()
	}
	in := harness.QueryInput{
		Provider:           s.opts.Provider,
		System:             s.system(),
		DynamicBoundary:    s.opts.DynamicBoundary,
		Messages:           s.messages,
		SystemReminder:     buildSystemReminder(reminderExtra),
		Tools:              s.registry,
		DeferredTools:      s.opts.DeferredTools,
		MaxTokens:          s.opts.MaxTokens,
		Temperature:        s.opts.Temperature,
		MaxTurns:           s.opts.MaxTurns,
		MaxDuration:        s.opts.MaxDuration,
		Settlement:         s.opts.Settlement,
		MaxConcurrency:     s.opts.MaxConcurrency,
		WorkingDir:         s.opts.WorkingDir,
		Todos:              s.opts.Todos,
		ToolOutputDir:      s.opts.ToolOutputDir,
		MaxToolOutputChars: s.opts.MaxToolOutputChars,
		Tasks:              s.tasks,
		BashEnv:            s.opts.BashEnv,
		PermissionMode:     s.opts.PermissionMode,
		Allowed:            s.opts.AllowedTools,
		Disallowed:         s.opts.DisallowedTools,
		CanUseTool:         s.opts.CanUseTool,
		Hooks:              s.opts.Hooks,
		Compactor:          s.compactor,
		TokenBudget:        s.opts.TokenBudget,
		EscalateMaxTokens:  s.opts.EscalateMaxTokens,
	}
	if s.writer != nil {
		in.Recorder = s.writer
	}
	if s.planCtrl != nil {
		in.PermissionModeFunc = s.planCtrl.Mode
	}
	inner := harness.Query(ctx, in, s.opts.Deps)
	return func(yield func(harness.Event, error) bool) {
		for ev, err := range inner {
			if ev.Kind == harness.KindResult && ev.Terminal != nil {
				s.messages = ev.Terminal.Messages
				s.usage = ev.Terminal.Usage
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

// Run is a one-shot convenience: it creates a Session, runs a single prompt to
// completion, and returns the assistant's final text (FR-07.4).
func Run(ctx context.Context, opts Options, input string) (string, error) {
	s := NewSession(opts)
	defer s.Close()
	var text string
	var rerr error
	for ev, err := range s.Prompt(ctx, input) {
		if err != nil {
			return "", err
		}
		if ev.Kind == harness.KindResult && ev.Terminal != nil {
			text = ev.Terminal.Text
			if ev.Terminal.Err != nil {
				rerr = ev.Terminal.Err
			}
		}
	}
	return text, rerr
}
