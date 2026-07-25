# Norma

A general-purpose **Agent Harness SDK in Go**, extracted from the core of a production LLM coding agent. It gives you the agent's engine — the streaming tool-use loop — as a reusable, dependency-free library you can point at any model (via the **Anthropic** *or* **OpenAI** wire format), drive with a **host-controlled system prompt**, and extend with your own tools, permission policy, hooks, and subagents.

The SDK is provider- and scenario-agnostic: the loop, scheduler, and safety boundary don't know whether they're driving a coding agent, a security auditor, or a customer-support bot.

Module path: `github.com/Autumn-27/norma` · Go 1.23+ (uses `iter`).

## Packages

| Package | Responsibility |
|---|---|
| `llm` | Vendor-neutral message model (`Message`/`ContentBlock`/`StreamEvent`) + `Provider` interface. `NewProvider(Config{Format})` builds an **Anthropic** or **OpenAI** adapter; both stream and support tool calls, normalized to one event type. |
| `tool` | `CoreTool` contract, fail-closed `Build` factory, `Registry`, a lightweight JSON-Schema validator, and the core tools: **Read, Write, Edit, MultiEdit, LS, Glob, Grep, Bash** (default set) plus opt-in **WebFetch** (network, permission-gated), **TodoWrite** (stateful planning), and **AskUserQuestion** (structured mid-task questions via a host callback). |
| `permission` | The safety boundary: `allow`/`deny`/`ask` decisions, six `PermissionMode`s, allow/deny rules with deny-priority, and the four-stage `Evaluate` pipeline. |
| `harness` | The loop: `Query` streams `Event`s (`iter.Seq2`), schedules tools (parallel for read-only, serial for mutating), pairs every `tool_use` with a `tool_result` (synthetic on deny/abort), enforces termination reasons, and threads cancellation. `QueryDeps` injects side effects for testing. |
| `compaction` | Context-window management: Snip / MicroCompact / AutoCompact, a circuit breaker, reactive recovery from prompt-too-long, and `compact_boundary` summaries. |
| `hook` | Eight lifecycle hooks (PreToolUse, PostToolUse, Stop, …) as Go callbacks that can block, rewrite input, inject context, or gate the Stop transition. |
| `subagent` | An `Agent` tool that delegates to isolated child agents reusing the same loop, with a recursion-depth guard. |
| `mcp` | Model Context Protocol integration: a stdio JSON-RPC client and in-process servers, exposing tools as `mcp__<server>__<tool>`. |
| `plan` | Plan mode: read-only exploration → `ExitPlanMode` presents a plan for approval → the permission mode flips mid-loop so the agent can implement. |
| `memory` | Persistent long-term memory: markdown-with-frontmatter files + a `MEMORY.md` index, relevance selection, staleness warnings, and `RecallMemory`/`RecordMemory` tools. |
| `skill` | Named, reusable instruction packages (`name`/`description`/`whenToUse` + body) loaded from `SKILL.md` files; a `Skill` tool injects a chosen skill's procedure on demand. |
| `coordinator` | Coordinator mode: a top-level agent that decomposes work and delegates to isolated `worker` subagents — LLM-driven (Agent tool + system segment) or programmatic (`RunParallel` fan-out). |
| `agentcore` | The public entry point: a stateful `Session` whose `Prompt` streams the loop, plus a one-shot `Run`. Aggregates everything above. |

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

Implements the core loop, dual providers, tools, the full permission pipeline, compaction, hooks, subagents, and MCP — plus plan mode, memory, Context Collapse (L3 structural compaction), the skill system, and coordinator multi-agent orchestration.
