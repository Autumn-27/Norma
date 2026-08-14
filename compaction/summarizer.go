package compaction

import (
	"context"
	"regexp"
	"strings"

	"github.com/Autumn-27/norma/llm"
)

// Summary generation constants (== claude-code).
const (
	compactMaxOutputTokens = 20000                                                   // COMPACT_MAX_OUTPUT_TOKENS
	maxPTLRetries          = 3                                                       // MAX_PTL_RETRIES
	maxStreamingRetries    = 2                                                       // MAX_COMPACT_STREAMING_RETRIES
	ptlRetryMarker         = "[earlier conversation truncated for compaction retry]" // PTL_RETRY_MARKER
)

// summarySystemPrompt is the summarizer's system prompt (== claude-code
// streamCompactSummary fallback path). Thinking is disabled and no tools are
// offered; the no-tools guardrails live in the user prompt (§7).
const summarySystemPrompt = "You are a helpful AI assistant tasked with summarizing conversations."

// ProviderSummarizer builds a Summarizer backed by an llm.Provider. It replays
// the real conversation messages (not a rendered text transcript) followed by the
// summary-request user message, with thinking disabled and output capped at
// min(20000, maxOutputTokens). It retries transient stream failures
// (MAX_COMPACT_STREAMING_RETRIES) and, if the summary request itself overflows,
// drops the oldest API-round groups and retries (MAX_PTL_RETRIES).
func ProviderSummarizer(p llm.Provider, maxOutputTokens int) Summarizer {
	maxOut := compactMaxOutputTokens
	if maxOutputTokens > 0 && maxOutputTokens < maxOut {
		maxOut = maxOutputTokens
	}
	return func(ctx context.Context, msgs []llm.Message) (string, error) {
		work := pairForReplay(msgs)
		var lastErr error
		for attempt := 0; attempt <= maxPTLRetries; attempt++ {
			raw, err := streamSummary(ctx, p, maxOut, work)
			if err == nil {
				return formatCompactSummary(raw), nil
			}
			if !IsOverflow(err) {
				return "", err
			}
			lastErr = err
			truncated, ok := truncateHeadForPTLRetry(work)
			if !ok {
				break
			}
			work = truncated
		}
		return "", lastErr
	}
}

// streamSummary issues one summary request (head + summary-request message),
// retrying transient (non-overflow) stream failures. Overflow is returned
// immediately so the caller can truncate and retry at a coarser granularity.
func streamSummary(ctx context.Context, p llm.Provider, maxOut int, head []llm.Message) (string, error) {
	reqMsgs := make([]llm.Message, 0, len(head)+1)
	reqMsgs = append(reqMsgs, head...)
	reqMsgs = append(reqMsgs, llm.UserText(compactPrompt))
	req := llm.CompletionRequest{
		System:    []string{summarySystemPrompt},
		Messages:  reqMsgs,
		MaxTokens: maxOut,
		Thinking:  "disabled",
	}
	var lastErr error
	for range maxStreamingRetries {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		acc := llm.NewAccumulator()
		var streamErr error
		for ev, err := range p.Stream(ctx, req) {
			if err != nil {
				streamErr = err
				break
			}
			acc.Add(ev)
		}
		if streamErr == nil {
			return acc.Message().Text(), nil
		}
		if IsOverflow(streamErr) {
			return "", streamErr // PTL — let the caller truncate, don't burn streaming retries
		}
		lastErr = streamErr // transient — retry
	}
	return "", lastErr
}

// pairForReplay drops dangling tool_use / tool_result blocks so the replayed head
// is a valid request (the working-set boundary may split a tool_use from its
// result). Mirrors claude-code's ensureToolResultPairing intent (drop, not
// synthesize — summarization only needs the surrounding text). Messages left
// empty are removed.
func pairForReplay(msgs []llm.Message) []llm.Message {
	useIDs := map[string]bool{}
	resIDs := map[string]bool{}
	for _, m := range msgs {
		for _, b := range m.Content {
			switch b.Type {
			case llm.BlockToolUse:
				useIDs[b.ID] = true
			case llm.BlockToolResult:
				resIDs[b.ToolUseID] = true
			}
		}
	}
	out := make([]llm.Message, 0, len(msgs))
	for _, m := range msgs {
		nc := make([]llm.ContentBlock, 0, len(m.Content))
		for _, b := range m.Content {
			switch b.Type {
			case llm.BlockToolUse:
				if !resIDs[b.ID] {
					continue
				}
			case llm.BlockToolResult:
				if !useIDs[b.ToolUseID] {
					continue
				}
			}
			nc = append(nc, b)
		}
		if len(nc) > 0 {
			out = append(out, llm.Message{Role: m.Role, Content: nc})
		}
	}
	return out
}

// truncateHeadForPTLRetry drops the oldest API-round groups when the summary
// request itself overflows (== claude-code truncateHeadForPTLRetry, 20% fallback
// branch — Norma cannot read a precise token gap from the provider error). It
// strips a leading retry marker, groups by API round, drops floor(20%) of the
// oldest groups (≥1, keeping ≥1), and re-prepends the marker if the result would
// start with an assistant message (the API requires a user-first message).
func truncateHeadForPTLRetry(msgs []llm.Message) ([]llm.Message, bool) {
	input := msgs
	if len(input) > 0 && isPTLMarker(input[0]) {
		input = input[1:]
	}
	groups := groupByAPIRound(input)
	if len(groups) < 2 {
		return nil, false
	}
	dropCount := len(groups) / 5 // floor(len * 0.2)
	dropCount = max(dropCount, 1)
	dropCount = min(dropCount, len(groups)-1)
	var sliced []llm.Message
	for _, g := range groups[dropCount:] {
		sliced = append(sliced, g...)
	}
	if len(sliced) == 0 {
		return nil, false
	}
	if sliced[0].Role == llm.RoleAssistant {
		sliced = append([]llm.Message{llm.UserText(ptlRetryMarker)}, sliced...)
	}
	return sliced, true
}

func isPTLMarker(m llm.Message) bool {
	return m.Role == llm.RoleUser && len(m.Content) == 1 &&
		m.Content[0].Type == llm.BlockText && m.Content[0].Text == ptlRetryMarker
}

// groupByAPIRound partitions messages into one group per API round-trip: a new
// group begins at each assistant message. Mirrors claude-code
// groupMessagesByApiRound (Norma has one assistant message per round, so role
// boundaries == message-id boundaries). Group 0 holds any leading preamble.
func groupByAPIRound(msgs []llm.Message) [][]llm.Message {
	var groups [][]llm.Message
	var cur []llm.Message
	for _, m := range msgs {
		if m.Role == llm.RoleAssistant && len(cur) > 0 {
			groups = append(groups, cur)
			cur = nil
		}
		cur = append(cur, m)
	}
	if len(cur) > 0 {
		groups = append(groups, cur)
	}
	return groups
}

var (
	analysisRe = regexp.MustCompile(`(?s)<analysis>.*?</analysis>`)
	summaryRe  = regexp.MustCompile(`(?s)<summary>(.*?)</summary>`)
	blankRe    = regexp.MustCompile(`\n{2,}`)
)

// formatCompactSummary post-processes the model's raw output (== claude-code
// formatCompactSummary): strip the <analysis> scratchpad, unwrap <summary> into a
// "Summary:" header, and collapse blank-line runs.
func formatCompactSummary(raw string) string {
	s := analysisRe.ReplaceAllString(raw, "")
	if m := summaryRe.FindStringSubmatch(s); m != nil {
		s = "Summary:\n" + strings.TrimSpace(m[1])
	}
	s = blankRe.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

// compactPrompt is the full summary-request user message:
// NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT + NO_TOOLS_TRAILER (§7).
var compactPrompt = noToolsPreamble + baseCompactPrompt + noToolsTrailer

// noToolsPreamble == claude-code NO_TOOLS_PREAMBLE (prompt.ts:19-26).
const noToolsPreamble = `CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

`

// noToolsTrailer == claude-code NO_TOOLS_TRAILER (prompt.ts:269-272).
const noToolsTrailer = `

REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task.`

// baseCompactPrompt == claude-code BASE_COMPACT_PROMPT (prompt.ts:61-143), verbatim
// (including the DETAILED_ANALYSIS_INSTRUCTION_BASE it interpolates and the source
// trailing spaces on the "All user messages:" and "...in your response. " lines).
const baseCompactPrompt = `Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>`
