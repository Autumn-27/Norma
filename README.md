# Norma

A general-purpose **Agent Harness SDK in Go**. Norma is the engine that turns a language model into an autonomous agent: a streaming tool-use loop that lets a model read, run, and reason its way through a task while you keep control of the prompt, the tools, and the safety boundary.

You point it at any model (via the **Anthropic** *or* **OpenAI** wire format), drive it with a **host-controlled system prompt**, and extend it with your own tools, permission policy, hooks, and subagents. It ships as a focused, near-dependency-free library — the loop, scheduler, and safety boundary are provider- and scenario-agnostic: they don't know whether they're driving a coding agent, a security auditor, or a customer-support bot.

**What you get out of the box:** dual-format streaming, a fail-closed tool system with the standard file/shell toolset, a four-stage permission pipeline, automatic context-window management, lifecycle hooks, subagents, MCP, plan mode, long-term memory, a skill system, background tasks, and multi-agent coordination — all behind one stateful `Session`.

Module path: `github.com/Autumn-27/norma` · Go 1.23+ (uses `iter`).

## Packages

| Package | Responsibility |
|---|---|
| `llm` | Vendor-neutral message model (`Message`/`ContentBlock`/`StreamEvent`) + `Provider` interface. `NewProvider(Config{Format})` builds an **Anthropic** or **OpenAI** adapter; both stream and support tool calls, normalized to one event type. |
| `tool` | `CoreTool` contract, fail-closed `Build` factory, `Registry`, a lightweight JSON-Schema validator, and the core tools: **Read, Write, Edit, MultiEdit, LS, Glob, Grep, Bash** (default set) plus opt-in **WebFetch** / **WebSearch** (network, permission-gated), **TodoWrite** (stateful planning), **AskUserQuestion** (structured mid-task questions), background-task tools (**TaskOutput/TaskStop/TaskList/Monitor**), and **deferred tools** — schemas withheld from the model to save context, discovered on demand via `SearchExtraTools` and invoked through `ExecuteExtraTool`. |
| `permission` | The safety boundary: `allow`/`deny`/`ask` decisions, six `PermissionMode`s, allow/deny rules with deny-priority, and the four-stage `Evaluate` pipeline. |
| `harness` | The loop: `Query` streams `Event`s (`iter.Seq2`), schedules tools (parallel for read-only, serial for mutating), pairs every `tool_use` with a `tool_result` (synthetic on deny/abort), enforces termination reasons, and threads cancellation. `QueryDeps` injects side effects for testing. |
| `compaction` | Context-window management: Snip / MicroCompact / Context Collapse / AutoCompact, a circuit breaker, reactive recovery from prompt-too-long, and non-destructive `compact_boundary` summaries. Invoked skills are re-injected verbatim after each boundary so their instructions survive summarization. |
| `hook` | Eight lifecycle hooks (PreToolUse, PostToolUse, Stop, …) as Go callbacks that can block, rewrite input, inject context, or gate the Stop transition. |
| `subagent` | An `Agent` tool that delegates to isolated child agents reusing the same loop, with a recursion-depth guard. |
| `mcp` | Model Context Protocol integration: a stdio JSON-RPC client and in-process servers, exposing tools as `mcp__<server>__<tool>`. |
| `plan` | Plan mode: read-only exploration → `ExitPlanMode` presents a plan for approval → the permission mode flips mid-loop so the agent can implement. |
| `memory` | Persistent long-term memory: markdown-with-frontmatter files + a `MEMORY.md` index, relevance selection, staleness warnings, and `RecallMemory`/`RecordMemory` tools. |
| `skill` | Named, reusable instruction packages loaded from `SKILL.md` files (YAML frontmatter — `name`/`description`/`license`/`compatibility`/`mcps` — plus a Markdown body). A `Skill` tool injects a chosen skill's procedure on demand as standing guidance that survives compaction, and can unlock a skill's declared MCP tools on invocation. |
| `coordinator` | Coordinator mode: a top-level agent that decomposes work and delegates to isolated `worker` subagents — LLM-driven (Agent tool + system segment) or programmatic (`RunParallel` fan-out). |
| `agentcore` | The public entry point: a stateful `Session` whose `Prompt` streams the loop, plus a one-shot `Run`. Aggregates everything above, and optionally persists an append-only JSONL transcript so a session can be resumed (`Session.Resume`). |

## Quick start

```go
prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, Model: "your-model-name"})
// or:  llm.Config{Format: llm.FormatOpenAI, BaseURL: "https://api.deepseek.com/v1", Model: "deepseek-chat"}

sess := agentcore.NewSession(agentcore.Options{
    Provider:       prov,
    SystemPrompt:   []string{"You are a helpful coding agent."}, // the SDK injects NO preset prompt
    PermissionMode: permission.ModeDefault,
    CanUseTool:     myApprovalCallback, // gates mutating tools
})

for ev, err := range sess.Prompt(ctx, "Add a /health endpoint and run the tests.") {
    if err != nil { /* ... */ }
    if ev.Kind == harness.KindText { fmt.Print(ev.Text) }
}
```

## CLI demo

```bash
go build -o agentcore ./cmd/agentcore

ANTHROPIC_API_KEY=... ./agentcore -provider anthropic -model your-model-name
OPENAI_API_KEY=...    ./agentcore -provider openai    -model gpt-4o
./agentcore -provider openai -model gpt-4o -yes -p "summarize the go files here"
```

Mutating tools (Write/Edit/Bash) prompt for approval unless `-yes` is set. Supply your own system prompt with `-system FILE`.

## Design highlights

- **Dual wire formats, one model.** Both adapters translate to/from a single Anthropic-style content-block model, so the loop and tools never change when you switch providers. Reasoning models' `reasoning_content` surfaces as thinking events.
- **Host owns the prompt.** The SDK ships no preset system prompt — `Options.SystemPrompt` is yours. Coding / security-audit prompts live in `examples/`, not the core.
- **Fail-closed tools.** New tools default to not-read-only, not-concurrency-safe, approval-required. Read-only tools opt in and become eligible for parallel execution.
- **`iter.Seq2` streaming.** `Prompt` and `Query` are range-over-func iterators; cancellation is via `context.Context`.

## Examples

- `examples/custom-llm` — three provider configs + a custom non-coding prompt + a custom tool.
- `examples/coding-agent` — full coding agent with interactive permissions, a PreToolUse hook, and compaction.
- `examples/security-audit` — read-only auditor that delegates to a subagent.
- `examples/planner-memory` — plan mode (explore → approve → implement) plus persistent memory.

## Tests

```bash
go test ./...
```

Coverage: both provider adapters over `httptest` SSE; the loop end-to-end with a scripted provider (tool execution, message threading, permission denial, max-turns, parallel reads); real tool execution in a temp dir; the permission pipeline matrix; compaction (auto-summary, circuit breaker, micro-compact); hooks (block/rewrite/Stop-gating); subagent dispatch + depth guard; and the MCP client over in-memory pipes.

## Scope

Implements the core loop, dual providers, tools, the full permission pipeline, compaction, hooks, subagents, and MCP — plus plan mode, long-term memory, the skill system, background tasks, deferred tools, transcript persistence with resume, and coordinator multi-agent orchestration.
