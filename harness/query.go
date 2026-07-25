package harness

import (
	"context"
	"fmt"
	"iter"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// QueryInput is a fully-resolved request to run the loop.
type QueryInput struct {
	Provider        llm.Provider
	System          []string
	DynamicBoundary int
	Messages        []llm.Message // conversation including the latest user turn
	// SystemReminder, when non-empty, is prepended as the first message on each
	// model call. It is not stored in conversation history — the host builds it
	// fresh per Prompt call so it never accumulates across turns.
	SystemReminder string
	Tools          *tool.Registry
	// DeferredTools names tools whose schemas are withheld from the model's tool
	// list (they stay in Tools and are invoked via ExecuteExtraTool). Empty = all
	// tool schemas sent as usual.
	DeferredTools []string
	MaxTokens     int
	Temperature     *float64
	// MaxTurns bounds tool-turns (0 = off). MaxDuration bounds wall-clock time,
	// checked at the turn boundary so a turn is never aborted mid-stream — unlike
	// cancelling ctx, which cuts an in-flight stream/tool (0 = off).
	MaxTurns        int
	MaxDuration     time.Duration
	MaxConcurrency  int
	WorkingDir      string
	AgentID         string
	// ToolOutputDir, when set, makes oversized tool output spill to a file there
	// (full content kept) instead of being discarded; MaxToolOutputChars sets the
	// per-tool output budget (head size; 0 = default 30000). See tool.Capture.
	ToolOutputDir      string
	MaxToolOutputChars int

	// Tasks, when non-nil, is the background-task manager shared with the tools.
	// The loop drains its completion notifications into the conversation between
	// tool turns so the model learns when background work finishes.
	Tasks *tool.Manager

	// BashEnv holds extra "KEY=VALUE" entries injected into Bash subprocess
	// environments (foreground + background). Empty = inherit unchanged.
	BashEnv []string

	// Permission policy.
	PermissionMode     permission.Mode
	PermissionModeFunc func() permission.Mode
	Allowed            []string
	Disallowed         []string
	CanUseTool         permission.CanUseTool

	// Hooks is optional lifecycle instrumentation. Nil disables it.
	Hooks HookRunner
	// Compactor is optional context-window management. Nil disables it.
	Compactor Compactor
	// Recorder, when non-nil, is notified as the conversation grows so a host can
	// persist a transcript. It fires for genuinely new messages (births) in
	// order, and for compaction boundaries; it is never called for compaction's
	// ephemeral working-set edits (collapse/snip), so the recorded transcript is
	// the raw, full-fidelity history.
	Recorder Recorder

	// TokenBudget, when > 0, enables budget-driven continuation: after a turn
	// with no tool calls, the agent is nudged to keep working until ~90% of the
	// budget (output tokens), stopping early on diminishing returns.
	TokenBudget int
	// EscalateMaxTokens, when true, first retries a max_tokens truncation with a
	// raised output cap (64k) before falling back to resume-recovery.
	EscalateMaxTokens bool

	// Settlement, when non-nil with a Prompt, runs a bounded wrap-up phase the
	// moment a budget (MaxTurns/MaxDuration) is hit: it injects Prompt as a user
	// turn and hides DisabledTools, so a budget-terminated run can checkpoint its
	// findings instead of ending abruptly. The run still finishes with the budget's
	// reason (ReasonMaxTurns/ReasonTimeout), never ReasonCompleted. Nil = off.
	Settlement *Settlement
}

// Settlement configures the wrap-up phase that runs when a budget is hit.
type Settlement struct {
	Prompt        string   // wrap-up instruction injected as a user turn (empty = settlement off)
	MaxTurns      int      // turn budget for the wrap-up phase itself (0 = default 2)
	DisabledTools []string // tool names hidden from the model during wrap-up (e.g. ["Bash"])
	// PromptByReason optionally overrides Prompt based on WHY the run settled
	// (keyed by the terminal reason, e.g. ReasonTimeout / ReasonMaxTurns). When the
	// settling reason has a non-empty entry here it is used; otherwise Prompt is used.
	// Lets a caller inject a different wrap-up depending on whether a run was cut by
	// its time budget vs its step budget, without knowing which at build time.
	PromptByReason map[TerminalReason]string
}

// promptFor returns the wrap-up prompt to inject for a settling reason: the
// PromptByReason override when present and non-empty, else the base Prompt.
func (s *Settlement) promptFor(reason TerminalReason) string {
	if s.PromptByReason != nil {
		if p := s.PromptByReason[reason]; p != "" {
			return p
		}
	}
	return s.Prompt
}

// Compactor is the harness's view of the compaction system (implemented in
// package compaction). Compaction is non-destructive; the harness sends only
// llm.MessagesForAPI(messages) to the model.
type Compactor interface {
	Pre(ctx context.Context, msgs []llm.Message) []llm.Message
	Reactive(ctx context.Context, msgs []llm.Message) ([]llm.Message, bool)
	IsOverflow(err error) bool
}

// Recorder receives conversation events for durable transcript persistence
// (implemented in package transcript). RecordMessage fires once per new message
// in birth order; usage carries the token delta attributed to that turn
// (assistant messages only; zero otherwise). RecordBoundary fires when
// compaction inserts a boundary, for observability — boundaries are not part of
// the raw reconstructable history.
type Recorder interface {
	RecordMessage(m llm.Message, usage llm.Usage)
	RecordBoundary(meta llm.BoundaryMeta)
}

// HookRunner is the harness's view of the hook system (implemented in package
// hook). Stop returns two distinct outcomes: preventContinuation (hard
// terminate) and/or blockingErrors (inject + continue).
type HookRunner interface {
	PreToolUse(ctx context.Context, name string, input []byte) (block bool, message string, updated []byte)
	PostToolUse(ctx context.Context, name string, input, result []byte, isErr bool)
	Stop(ctx context.Context, messages []llm.Message) (prevent bool, blockingErrors []string, message string)
}

// QueryDeps are injectable side-effect boundaries for testing (FR-01.6).
type QueryDeps struct {
	CallModel func(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error]
	NewID     func() string
}

func (d *QueryDeps) withDefaults(in QueryInput) {
	if d.CallModel == nil {
		d.CallModel = in.Provider.Stream
	}
	if d.NewID == nil {
		var n int64
		d.NewID = func() string { return "id_" + strconv.FormatInt(atomic.AddInt64(&n, 1), 10) }
	}
}

// Query runs the agent loop and returns its event stream. The final event is
// always KindResult carrying a Terminal (unless the consumer stops early).
func Query(ctx context.Context, in QueryInput, deps QueryDeps) iter.Seq2[Event, error] {
	deps.withDefaults(in)
	conc := in.MaxConcurrency
	if conc <= 0 {
		conc = 10
	}
	return func(yield func(Event, error) bool) {
		l := &loop{ctx: ctx, in: in, deps: deps, yield: yield, sem: make(chan struct{}, conc)}
		l.messages = append([]llm.Message(nil), in.Messages...)
		l.run()
	}
}

// budgetTracker tracks token-budget continuation state.
type budgetTracker struct {
	continuationCount    int
	lastGlobalTurnTokens int
	lastDeltaTokens      int
}

type loop struct {
	ctx   context.Context
	in    QueryInput
	deps  QueryDeps
	yield func(Event, error) bool
	sem   chan struct{} // bounds concurrent tool execution

	messages      []llm.Message
	usage         llm.Usage
	turnCount     int
	boundaryCount int // boundary markers already seen, so Recorder fires once each
	startTime     time.Time // loop start, for MaxDuration (wall-clock budget)

	// settlement (wrap-up) phase state, entered when a budget is hit
	settling     bool
	settleReason TerminalReason // the budget reason to finish with after wrap-up
	settleTurn   int

	// recovery / continuation state
	maxOutputTokensOverride      int
	maxOutputTokensRecoveryCount int
	hasAttemptedReactiveCompact  bool
	stopHookActive               bool
	budget                       budgetTracker
	lastContinue                 ContinueReason
}

const (
	budgetCompletionThresholdNum = 9 // 0.9
	budgetCompletionThresholdDen = 10
	budgetDiminishingThreshold   = 500
	escalatedMaxTokens           = 64000
	maxOutputTokensRecoveryLimit = 3
	// maxLoopIterations is a defensive backstop; turnCount counts only tool-turns,
	// so this only guards against a pathological recovery/continuation loop.
	maxLoopIterations = 10000
)

func (l *loop) emit(ev Event) bool { return l.yield(ev, nil) }

// record forwards a newly appended message to the transcript Recorder, if any.
func (l *loop) record(m llm.Message, u llm.Usage) {
	if l.in.Recorder != nil {
		l.in.Recorder.RecordMessage(m, u)
	}
}

// injectTaskNotifications drains completion notifications from the background-task
// manager and appends them to the conversation as a user message, so the model is
// told when background work finishes. Returns true if any were injected.
func (l *loop) injectTaskNotifications() bool {
	if l.in.Tasks == nil {
		return false
	}
	notes := l.in.Tasks.DrainNotifications()
	if len(notes) == 0 {
		return false
	}
	var b strings.Builder
	for i, n := range notes {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(n.String())
	}
	msg := llm.UserText(b.String())
	l.messages = append(l.messages, msg)
	l.record(msg, llm.Usage{})
	return true
}

// recordNewBoundary persists a boundary that compaction just inserted (detected
// by a rise in the boundary-marker count). It writes the post-boundary working
// set — the boundary marker, the summary, and the kept recent tail — as durable
// messages, so a COLD-START Resume restores the compacted view via MessagesForAPI
// instead of replaying raw history and re-summarizing. The tail is re-recorded
// after the boundary (it also exists earlier as births); only the post-boundary
// copy is sent to the model. RecordBoundary keeps the observability-only marker
// (a Type:"boundary" record, skipped during reconstruction).
func (l *loop) recordNewBoundary() {
	if l.in.Recorder == nil {
		return
	}
	c := countBoundaries(l.messages)
	if c <= l.boundaryCount {
		return
	}
	if i := llm.LastBoundaryIndex(l.messages); i >= 0 {
		if meta, ok := llm.ParseBoundaryMeta(l.messages[i]); ok {
			l.in.Recorder.RecordBoundary(meta)
		}
		for _, m := range l.messages[i:] { // boundary marker + summary + kept tail
			l.in.Recorder.RecordMessage(m, llm.Usage{})
		}
	}
	l.boundaryCount = c
}

func usageDelta(before, after llm.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:      after.InputTokens - before.InputTokens,
		OutputTokens:     after.OutputTokens - before.OutputTokens,
		CacheReadTokens:  after.CacheReadTokens - before.CacheReadTokens,
		CacheWriteTokens: after.CacheWriteTokens - before.CacheWriteTokens,
	}
}

func countBoundaries(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		if llm.IsBoundaryMarker(m) {
			n++
		}
	}
	return n
}

func (l *loop) finish(reason TerminalReason, err error, text string) {
	l.yield(Event{Kind: KindResult, Terminal: &Terminal{
		Reason:   reason,
		Err:      err,
		Usage:    l.usage,
		Turns:    l.turnCount,
		Messages: l.messages,
		Text:     text,
	}}, nil)
}

func (l *loop) maxTokens() int {
	if l.maxOutputTokensOverride > 0 {
		return l.maxOutputTokensOverride
	}
	if l.in.MaxTokens > 0 {
		return l.in.MaxTokens
	}
	return 8192
}

func (l *loop) run() {
	var schemas []llm.ToolSchema
	if l.in.Tools != nil {
		schemas = l.in.Tools.Schemas()
		// Deferred tools stay in the registry (callable via ExecuteExtraTool) but
		// their schemas are withheld from the model to save context tokens.
		if len(l.in.DeferredTools) > 0 {
			schemas = filterToolSchemas(schemas, l.in.DeferredTools)
		}
	}
	l.turnCount = 1 // the first model call is turn 1
	l.boundaryCount = countBoundaries(l.messages)
	l.startTime = time.Now() // wall-clock baseline for MaxDuration
	iterations := 0
	for {
		if l.ctx.Err() != nil {
			l.finish(ReasonAbortedStreaming, l.ctx.Err(), "")
			return
		}
		// Defensive backstop against a pathological recovery/continuation loop.
		// turnCount itself only counts tool-turns (below); every continuation path
		// is individually bounded, so this guard should never fire in practice.
		iterations++
		if iterations > maxLoopIterations {
			l.finish(ReasonMaxTurns, nil, "")
			return
		}

		// Non-destructive proactive compaction (snip/micro/collapse/auto).
		if l.in.Compactor != nil {
			l.messages = l.in.Compactor.Pre(l.ctx, l.messages)
			l.recordNewBoundary()
		}

		usageBefore := l.usage
		asst, exec, stopReason, streamErr, consumerStopped, aborted := l.streamAndExecute(schemas)
		if consumerStopped {
			return
		}
		if streamErr != nil {
			if l.ctx.Err() != nil {
				l.finish(ReasonAbortedStreaming, l.ctx.Err(), "")
				return
			}
			// Reactive recovery from prompt-too-long (collapse → compact → snip).
			if l.in.Compactor != nil && l.in.Compactor.IsOverflow(streamErr) && !l.hasAttemptedReactiveCompact {
				l.hasAttemptedReactiveCompact = true
				if msgs, ok := l.in.Compactor.Reactive(l.ctx, l.messages); ok {
					l.messages = msgs
					l.recordNewBoundary()
					l.lastContinue = ContinueReactiveCompactRetry
					continue
				}
				l.finish(ReasonPromptTooLong, streamErr, "")
				return
			}
			l.finish(ReasonModelError, streamErr, "")
			return
		}
		l.hasAttemptedReactiveCompact = false
		l.messages = append(l.messages, asst)
		l.record(asst, usageDelta(usageBefore, l.usage))
		// surface cumulative usage after each model turn so hosts can show live token
		// counts during a run (not just at the terminal event).
		if u := l.usage; u != (llm.Usage{}) {
			if !l.emit(Event{Kind: KindUsage, Usage: &u}) {
				return
			}
		}
		toolUses := asst.ToolUses()

		// max_tokens truncation with no tool calls → escalate / resume-recovery.
		// These continuations are retries of the same turn (they do not advance
		// turnCount).
		if stopReason == "max_tokens" && len(toolUses) == 0 {
			if l.handleMaxOutputTokens() {
				continue
			}
			// recovery exhausted: surface the truncated turn as completion.
		} else {
			l.maxOutputTokensOverride = 0
		}

		if len(toolUses) == 0 {
			// During the wrap-up phase, a natural stop ends the run with the ORIGINAL
			// budget reason (not "completed") — the run was budget-terminated; the
			// settlement is just a checkpoint. Skip the keep-working nudges below.
			if l.settling {
				l.finish(l.settleReason, nil, asst.Text())
				return
			}
			// Stop hooks: hard terminate, or force continue with injected errors.
			// stopHookActive prevents an always-blocking hook from looping forever
			// without an intervening tool-turn.
			if l.in.Hooks != nil && !l.stopHookActive {
				prevent, blocking, msg := l.in.Hooks.Stop(l.ctx, l.messages)
				if prevent {
					l.finish(ReasonStopHookPrevented, nil, msg)
					return
				}
				if len(blocking) > 0 {
					for _, b := range blocking {
						bm := llm.UserText(b)
						l.messages = append(l.messages, bm)
						l.record(bm, llm.Usage{})
					}
					l.stopHookActive = true
					l.lastContinue = ContinueStopHookBlocking
					continue
				}
			}
			l.stopHookActive = false
			// Surface any background-task completions before ending the turn; if any
			// landed, take another turn so the model can react to them.
			if l.injectTaskNotifications() {
				l.lastContinue = ContinueTaskNotification
				continue
			}
			// Token budget: nudge to keep working until ~90% / diminishing returns.
			if l.in.TokenBudget > 0 && l.checkBudgetContinue() {
				l.lastContinue = ContinueTokenBudget
				continue
			}
			l.finish(ReasonCompleted, nil, asst.Text())
			return
		}

		// Tools were executed during streaming; record their results.
		results := exec.results()
		resultMsg := llm.Message{Role: llm.RoleUser, Content: results}
		l.messages = append(l.messages, resultMsg)
		l.record(resultMsg, llm.Usage{})
		// Surface any background-task completions that landed during this turn so
		// the next model round sees them alongside the tool results.
		l.injectTaskNotifications()
		if aborted || l.ctx.Err() != nil {
			l.finish(ReasonAbortedTools, l.ctx.Err(), "")
			return
		}

		// next_turn: only tool-turns advance turnCount.
		l.stopHookActive = false // a real turn happened; stop hooks may fire fresh
		if l.settling {
			// wrap-up phase: bounded by its own small turn budget. When spent, finish
			// with the original budget reason (not "completed").
			l.settleTurn++
			if l.settleTurn >= l.settleMaxTurns() {
				l.finish(l.settleReason, nil, "")
				return
			}
			l.lastContinue = ContinueNextTurn
			continue
		}
		// Primary budgets are checked HERE, at the turn boundary — never mid-stream —
		// so a turn is not aborted in flight (that would surface as an abort/error).
		// On hit, optionally run the wrap-up phase before finishing.
		nextTurn := l.turnCount + 1
		hitTurns := l.in.MaxTurns > 0 && nextTurn > l.in.MaxTurns
		hitTime := l.in.MaxDuration > 0 && time.Since(l.startTime) >= l.in.MaxDuration
		if hitTurns || hitTime {
			reason := ReasonMaxTurns
			if hitTime && !hitTurns {
				reason = ReasonTimeout
			}
			if l.beginSettlement(reason, &schemas) {
				l.lastContinue = ContinueNextTurn
				continue
			}
			l.finish(reason, nil, "")
			return
		}
		l.turnCount = nextTurn
		l.lastContinue = ContinueNextTurn
	}
}

// settleMaxTurns is the wrap-up phase's own turn budget (default 2).
func (l *loop) settleMaxTurns() int {
	if l.in.Settlement != nil && l.in.Settlement.MaxTurns > 0 {
		return l.in.Settlement.MaxTurns
	}
	return 2
}

// beginSettlement enters the wrap-up phase when configured: injects the caller's
// prompt as a user turn and hides DisabledTools, so a budget-terminated run can
// checkpoint its findings. Returns false when no settlement is configured (caller
// then finishes immediately).
func (l *loop) beginSettlement(reason TerminalReason, schemas *[]llm.ToolSchema) bool {
	s := l.in.Settlement
	if s == nil {
		return false
	}
	prompt := s.promptFor(reason)
	if prompt == "" {
		return false
	}
	msg := llm.UserText(prompt)
	l.messages = append(l.messages, msg)
	l.record(msg, llm.Usage{})
	l.settling = true
	l.settleReason = reason
	l.settleTurn = 0
	if len(s.DisabledTools) > 0 {
		*schemas = filterToolSchemas(*schemas, s.DisabledTools)
	}
	return true
}

// filterToolSchemas returns schemas with the named tools removed.
func filterToolSchemas(schemas []llm.ToolSchema, disabled []string) []llm.ToolSchema {
	block := make(map[string]bool, len(disabled))
	for _, d := range disabled {
		block[d] = true
	}
	out := make([]llm.ToolSchema, 0, len(schemas))
	for _, s := range schemas {
		if !block[s.Name] {
			out = append(out, s)
		}
	}
	return out
}

// handleMaxOutputTokens implements the escalate→recover(3×) flow. Returns true
// if it injected a recovery and the loop should continue.
func (l *loop) handleMaxOutputTokens() bool {
	if l.in.EscalateMaxTokens && l.maxOutputTokensOverride == 0 {
		l.maxOutputTokensOverride = escalatedMaxTokens
		l.lastContinue = ContinueMaxOutputTokensEscalate
		return true
	}
	if l.maxOutputTokensRecoveryCount < maxOutputTokensRecoveryLimit {
		nudge := llm.UserText(
			"Output token limit hit. Resume directly — no apology, no recap of what you already wrote. Continue exactly where you left off.")
		l.messages = append(l.messages, nudge)
		l.record(nudge, llm.Usage{})
		l.maxOutputTokensRecoveryCount++
		l.maxOutputTokensOverride = 0
		l.lastContinue = ContinueMaxOutputTokensRecovery
		return true
	}
	return false
}

// checkBudgetContinue continues with a nudge while under 90% of budget and not
// in diminishing returns; otherwise stops.
func (l *loop) checkBudgetContinue() bool {
	budget := l.in.TokenBudget
	if budget <= 0 || l.in.AgentID != "" { // subagents have no budget
		return false
	}
	turnTokens := l.usage.OutputTokens
	delta := turnTokens - l.budget.lastGlobalTurnTokens
	diminishing := l.budget.continuationCount >= 3 && delta < budgetDiminishingThreshold && l.budget.lastDeltaTokens < budgetDiminishingThreshold
	if !diminishing && turnTokens < budget*budgetCompletionThresholdNum/budgetCompletionThresholdDen {
		l.budget.continuationCount++
		l.budget.lastDeltaTokens = delta
		l.budget.lastGlobalTurnTokens = turnTokens
		pct := 0
		if budget > 0 {
			pct = turnTokens * 100 / budget
		}
		nudge := llm.UserText(fmt.Sprintf(
			"You have used %d%% of your token budget for this task (%d/%d output tokens). Keep going and finish the task; wrap up before the budget is exhausted.",
			pct, turnTokens, budget))
		l.messages = append(l.messages, nudge)
		l.record(nudge, llm.Usage{})
		return true
	}
	return false
}

// streamAndExecute runs one model completion, surfacing text/thinking, and feeds
// tool_use blocks to the streaming executor as they finish — so tools begin
// running while the rest of the message is still arriving. It returns the
// assembled assistant message, the executor, the stop reason, and flags.
func (l *loop) streamAndExecute(schemas []llm.ToolSchema) (asst llm.Message, exec *streamExec, stopReason string, err error, consumerStopped, aborted bool) {
	msgs := llm.MessagesForAPI(l.messages)
	if l.in.SystemReminder != "" {
		msgs = append([]llm.Message{llm.UserText(l.in.SystemReminder)}, msgs...)
	}
	req := llm.CompletionRequest{
		System:          l.in.System,
		DynamicBoundary: l.in.DynamicBoundary,
		Messages:        msgs,
		Tools:           schemas,
		MaxTokens:       l.maxTokens(),
		Temperature:     l.in.Temperature,
	}
	exec = newStreamExec(l)
	acc := llm.NewAccumulator()

	var building bool
	var curID, curName string
	var curInput strings.Builder
	finalizeTool := func() bool { // false = consumer stopped
		if !building {
			return true
		}
		building = false
		raw := strings.TrimSpace(curInput.String())
		curInput.Reset()
		if raw == "" || !jsonValid(raw) {
			raw = "{}"
		}
		block := llm.ContentBlock{Type: llm.BlockToolUse, ID: curID, Name: curName, Input: []byte(raw)}
		if !l.emit(Event{Kind: KindToolUse, ToolUse: &block}) {
			return false
		}
		exec.add(block)
		return exec.drain()
	}

	for ev, e := range l.deps.CallModel(l.ctx, req) {
		if e != nil {
			err = e
			break
		}
		if ev.Type == llm.SEToolInputJSON {
			if building {
				curInput.WriteString(ev.Text)
			}
			acc.Add(ev)
			continue
		}
		if !finalizeTool() {
			return asst, exec, "", nil, true, false
		}
		switch ev.Type {
		case llm.SETextDelta:
			if ev.Text != "" && !l.emit(Event{Kind: KindText, Text: ev.Text}) {
				return asst, exec, "", nil, true, false
			}
		case llm.SEThinkingDelta:
			if ev.Text != "" && !l.emit(Event{Kind: KindThinking, Text: ev.Text}) {
				return asst, exec, "", nil, true, false
			}
		case llm.SEToolUseStart:
			curID, curName, building = ev.ToolID, ev.ToolName, true
		}
		acc.Add(ev)
	}
	if !finalizeTool() {
		return asst, exec, "", nil, true, false
	}
	l.usage.Add(acc.Usage)
	asst = acc.Message()
	stopReason = acc.StopReason
	if err != nil {
		return asst, exec, stopReason, err, false, false
	}
	// Drain tools to completion (or synthesize on abort).
	cs, ab := exec.finish()
	if cs {
		return asst, exec, stopReason, nil, true, false
	}
	return asst, exec, stopReason, nil, false, ab
}

func jsonValid(s string) bool {
	// minimal check used only to guard tool input assembly
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	depth := 0
	inStr := false
	esc := false
	for _, r := range s {
		if inStr {
			if esc {
				esc = false
			} else if r == '\\' {
				esc = true
			} else if r == '"' {
				inStr = false
			}
			continue
		}
		switch r {
		case '"':
			inStr = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
		}
	}
	return depth == 0 && !inStr
}
