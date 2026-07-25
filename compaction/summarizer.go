package compaction

import (
	"context"
	"strings"

	"github.com/Autumn-27/norma/llm"
)

// ProviderSummarizer builds a Summarizer backed by an llm.Provider. It renders
// the messages as a transcript and asks the model for a concise digest, so it
// is robust to dangling tool_use/tool_result pairing.
func ProviderSummarizer(p llm.Provider) Summarizer {
	return func(ctx context.Context, msgs []llm.Message) (string, error) {
		var b strings.Builder
		for _, m := range msgs {
			b.WriteString(string(m.Role))
			b.WriteString(": ")
			b.WriteString(transcriptLine(m))
			b.WriteByte('\n')
		}
		req := llm.CompletionRequest{
			System:    []string{summarySystemPrompt},
			Messages:  []llm.Message{llm.UserText(summaryRequestPrompt + "\n\n<conversation>\n" + b.String() + "\n</conversation>")},
			MaxTokens: 8192,
		}
		acc := llm.NewAccumulator()
		for ev, err := range p.Stream(ctx, req) {
			if err != nil {
				return "", err
			}
			acc.Add(ev)
		}
		return acc.Message().Text(), nil
	}
}

// summarySystemPrompt / summaryRequestPrompt build the compaction summary
// request: a single-turn, no-tools call that yields a 9-section digest
// preserving everything needed to continue the work.
const summarySystemPrompt = `You are summarizing a conversation to compact it into context. Do NOT call any tools. Produce only the structured summary requested — be thorough and specific, preserving every detail needed to continue the work without the original transcript.`

const summaryRequestPrompt = `Summarize the conversation above into the following structured sections. Preserve concrete details: file paths, identifiers, commands, error text, and decisions.

1. Primary Request and Intent — what the user is ultimately trying to achieve.
2. Key Technical Concepts — technologies, frameworks, patterns in play.
3. Files and Code — files read or modified, with the salient code/changes and why.
4. Errors and Fixes — errors encountered and how they were (or weren't) resolved.
5. Problem Solving — approaches tried, what worked, what was ruled out.
6. All User Messages — a list of the user's explicit instructions, in order.
7. Pending Tasks — what remains to be done.
8. Current Work — exactly what was being worked on at the moment of summarization.
9. Next Step — the single most logical next action.
10. Active Skills — any skill invoked in the conversation that is still in effect, named, with the key guidelines it imposed (so its procedure is not lost).`

func transcriptLine(m llm.Message) string {
	var parts []string
	for _, b := range m.Content {
		switch b.Type {
		case llm.BlockText, llm.BlockThinking:
			parts = append(parts, b.Text)
		case llm.BlockToolUse:
			parts = append(parts, "[called "+b.Name+" "+string(b.Input)+"]")
		case llm.BlockToolResult:
			parts = append(parts, "[tool result: "+truncateStr(flatten(b.Content), 500)+"]")
		}
	}
	return strings.Join(parts, " ")
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
