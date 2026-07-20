// Package harness implements the agent's core loop as a state machine:
// stream a completion, recover from max-output-tokens / prompt-too-long,
// surface text/thinking/tool events, execute tool calls through a streaming
// executor (FIFO + exclusive scheduling), gate the Stop transition with hooks
// and a token budget, and repeat. It is exposed as a range-over-func iterator
// of Events and threads context cancellation throughout.
package harness

import "github.com/Autumn-27/norma/llm"

// EventKind classifies a loop Event.
type EventKind string

const (
	KindText       EventKind = "text"        // assistant text delta
	KindThinking   EventKind = "thinking"    // reasoning delta
	KindToolUse    EventKind = "tool_use"    // a tool is about to run
	KindToolResult EventKind = "tool_result" // a tool finished
	KindProgress   EventKind = "progress"    // mid-tool progress
	KindUsage      EventKind = "usage"       // cumulative token usage so far (per model turn)
	KindResult     EventKind = "result"      // terminal event (loop finished)
)

// Event is one item in the loop's output stream.
type Event struct {
	Kind       EventKind
	Text       string            // KindText / KindThinking / KindProgress
	ToolUse    *llm.ContentBlock // KindToolUse
	ToolResult *llm.ContentBlock // KindToolResult
	Terminal   *Terminal         // KindResult
	Usage      *llm.Usage        // KindUsage: cumulative usage after the latest model turn
}

// TerminalReason explains why the loop stopped.
type TerminalReason string

const (
	ReasonCompleted         TerminalReason = "completed"           // model ended its turn
	ReasonBlockingLimit     TerminalReason = "blocking_limit"      // hard context limit hit pre-request
	ReasonImageError        TerminalReason = "image_error"         // unsupported media (multimodal)
	ReasonModelError        TerminalReason = "model_error"         // provider/API failure
	ReasonAbortedStreaming  TerminalReason = "aborted_streaming"   // canceled during streaming
	ReasonAbortedTools      TerminalReason = "aborted_tools"       // canceled during tool execution
	ReasonPromptTooLong     TerminalReason = "prompt_too_long"     // overflow recovery exhausted
	ReasonStopHookPrevented TerminalReason = "stop_hook_prevented" // a Stop hook blocked continuation
	ReasonHookStopped       TerminalReason = "hook_stopped"        // a tool/hook stopped continuation
	ReasonMaxTurns          TerminalReason = "max_turns"           // hit MaxTurns
	ReasonTimeout           TerminalReason = "timeout"             // hit a wall-clock run budget (host-imposed)
)

// ContinueReason names why the loop took another turn. These drive control
// flow; they are recorded on the loop for telemetry.
type ContinueReason string

const (
	ContinueNextTurn                ContinueReason = "next_turn"
	ContinueReactiveCompactRetry    ContinueReason = "reactive_compact_retry"
	ContinueMaxOutputTokensEscalate ContinueReason = "max_output_tokens_escalate"
	ContinueMaxOutputTokensRecovery ContinueReason = "max_output_tokens_recovery"
	ContinueStopHookBlocking        ContinueReason = "stop_hook_blocking"
	ContinueTokenBudget             ContinueReason = "token_budget_continuation"
	ContinueTaskNotification        ContinueReason = "task_notification"
)

// Terminal is the final state of a Query, delivered as the KindResult event.
type Terminal struct {
	Reason   TerminalReason
	Err      error
	Usage    llm.Usage
	Turns    int
	Messages []llm.Message // full conversation snapshot (non-destructive: retains history)
	Text     string        // the assistant's final text, for convenience
}
