package noaadapter

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/agentcore"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

// pressureSession builds a session whose view is large enough to sit in the
// pressure band with a real backlog.
func pressureSession(t *testing.T) (*Session, []llm.Message) {
	t.Helper()
	opts := baseOptions()
	cfg := noa.DefaultConfig(20000) // a small window makes the fixture manageable
	sess, err := EnableWithSession(&opts, Options{
		ArchiveBaseDir: t.TempDir(), SessionID: "s", Config: &cfg,
	})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}
	body := strings.Repeat("compressible tool output. ", 400)
	msgs := []llm.Message{userText("the task")}
	for i := range 6 {
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ID: "t" + string(rune('a'+i)), Name: "Bash", Input: json.RawMessage(`{}`)},
		}})
		msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: "t" + string(rune('a'+i)),
			Content: []llm.ContentBlock{llm.TextBlock(body)},
		}}})
	}
	tail := strings.Repeat("recent. ", 300)
	for range 6 {
		msgs = append(msgs, userText(tail))
	}
	return sess, msgs
}

func viewHasNudge(view []llm.Message) bool {
	for _, m := range view {
		if strings.Contains(m.Text(), "HOW TO COMPRESS") {
			return true
		}
	}
	return false
}

func TestNudgeIsInjectedUnderPressure(t *testing.T) {
	sess, msgs := pressureSession(t)
	view := sess.View(msgs)
	if !viewHasNudge(view) {
		t.Fatal("no nudge injected despite pressure and a large backlog")
	}
	last := view[len(view)-1]
	if last.Role != llm.RoleUser {
		t.Fatalf("the nudge landed as %s, want a user message", last.Role)
	}
}

// The nudge exists only in the request. It is never written back, so the host's
// history stays exactly as it was.
func TestNudgeNeverEntersHistory(t *testing.T) {
	sess, msgs := pressureSession(t)
	before := len(msgs)
	view := sess.View(msgs)
	if !viewHasNudge(view) {
		t.Skip("no nudge this turn; the injection path is covered elsewhere")
	}
	if len(msgs) != before {
		t.Fatalf("history grew from %d to %d — the nudge must not be written back", before, len(msgs))
	}
	for _, m := range msgs {
		if strings.Contains(m.Text(), "HOW TO COMPRESS") {
			t.Fatal("a nudge reached the stored history")
		}
	}
}

// Three consecutive ignored nudges must silence nudging: replaying a
// multi-thousand-token nudge into an overflowing context makes the problem it
// describes worse.
func TestNudgeSuppressedAfterRepeatedIgnores(t *testing.T) {
	sess, msgs := pressureSession(t)

	nudged := 0
	for range sess.Config().MaxCompressAttempts {
		if viewHasNudge(sess.View(msgs)) {
			nudged++
		}
		// The model answers with plain text — no Compress call at all.
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextBlock("working on it")}})
	}
	if nudged == 0 {
		t.Fatal("no nudge fired at all; the fixture is not under pressure")
	}
	if viewHasNudge(sess.View(msgs)) {
		t.Fatalf("still nudging after %d ignored prompts", nudged)
	}
}

// Suppression must not be permanent: once the context has grown by a cadence
// step the situation genuinely differs from the one the model declined.
func TestNudgeSuppressionLiftsAfterGrowth(t *testing.T) {
	sess, msgs := pressureSession(t)
	for range sess.Config().MaxCompressAttempts + 1 {
		sess.View(msgs)
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextBlock("working on it")}})
	}
	if viewHasNudge(sess.View(msgs)) {
		t.Fatal("expected suppression before testing its release")
	}

	// The context grows by well over a cadence step.
	grow := strings.Repeat("more content. ", 4000)
	for range 3 {
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextBlock(grow)}})
	}
	if !viewHasNudge(sess.View(msgs)) {
		t.Fatal("suppression did not lift after the context grew a full cadence step")
	}
}

// A new user message is a new situation; the ladder resets.
func TestNudgeSuppressionResetsOnUserTurn(t *testing.T) {
	sess, msgs := pressureSession(t)
	for range sess.Config().MaxCompressAttempts + 1 {
		sess.View(msgs)
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextBlock("working on it")}})
	}
	if viewHasNudge(sess.View(msgs)) {
		t.Fatal("expected suppression before testing the reset")
	}
	msgs = append(msgs, userText("actually, do this instead"))
	if !viewHasNudge(sess.View(msgs)) {
		t.Fatal("a new user turn must reset the failure ladder")
	}
}

// Norma packs tool results into user-role messages, so the role alone cannot
// distinguish a person speaking from a tool answering.
func TestIsRealUserTurn(t *testing.T) {
	if !isRealUserTurn(userText("hello")) {
		t.Error("a plain user message must read as a real turn")
	}
	if isRealUserTurn(toolResult("t1", "output")) {
		t.Error("a packaged tool result must NOT read as a real user turn")
	}
	if isRealUserTurn(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("x")}}) {
		t.Error("an assistant message is not a user turn")
	}
	if isRealUserTurn(llm.Message{Role: llm.RoleUser}) {
		t.Error("an empty message is not a turn")
	}
}

func TestLastAssistantCalledCompress(t *testing.T) {
	compressCall := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		{Type: llm.BlockToolUse, ID: "t1", Name: noa.CompressToolName, Input: json.RawMessage(`{}`)},
	}}
	other := llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		{Type: llm.BlockToolUse, ID: "t1", Name: "Read", Input: json.RawMessage(`{}`)},
	}}
	if !lastAssistantCalledCompress([]llm.Message{userText("q"), compressCall}) {
		t.Error("a Compress call was not detected")
	}
	if lastAssistantCalledCompress([]llm.Message{userText("q"), other}) {
		t.Error("a Read call was mistaken for a Compress call")
	}
	// Only the MOST RECENT assistant message counts: an older compression says
	// nothing about whether this nudge was acted on.
	if lastAssistantCalledCompress([]llm.Message{compressCall, other}) {
		t.Error("an older Compress call must not mask the latest response")
	}
}

// A successful compression clears the ladder outright.
func TestSuccessfulCompressionResetsAttempts(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)

	// Two failures.
	for range 2 {
		_, _ = runCompress(sess, json.RawMessage(`{"content":[{"startId":"m09999","endId":"m09998","summary":"`+
			strings.Repeat("s", 100)+`"}]}`), &tool.ToolContext{ToolUseID: "x"})
	}
	if sess.attempts != 2 {
		t.Fatalf("attempts = %d, want 2", sess.attempts)
	}

	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00001", "endId": "m00008", "summary": longSummary}},
	})
	res, _ := runCompress(sess, args, &tool.ToolContext{ToolUseID: "ok"})
	if noa.PanelBlockCount(res.Content[0].Text) == 0 {
		t.Skipf("the fixture produced no compressible block: %s", res.Content[0].Text)
	}
	if sess.attempts != 0 {
		t.Fatalf("attempts = %d after a successful compression, want 0", sess.attempts)
	}
}

// Malformed arguments count against the same ladder: from the context's point
// of view nothing was reclaimed either way.
func TestParseFailureCountsAsAttempt(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)
	before := sess.attempts
	res, _ := runCompress(sess, json.RawMessage(`not json`), &tool.ToolContext{ToolUseID: "x"})
	if !res.IsError {
		t.Fatal("malformed arguments must be reported as an error")
	}
	if sess.attempts != before+1 {
		t.Fatalf("attempts = %d, want %d", sess.attempts, before+1)
	}
}

// "No ranges provided" is neither success nor failure — a valid question with a
// valid answer must not distort the ladder.
func TestNoRangesIsNeutral(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)
	before := sess.attempts
	res, _ := runCompress(sess, json.RawMessage(`{"content":[]}`), &tool.ToolContext{ToolUseID: "x"})
	if res.IsError {
		t.Fatalf("an empty range list must not be an error: %s", res.Content[0].Text)
	}
	if res.Content[0].Text != noa.NoRangesMessage {
		t.Fatalf("result = %q, want %q", res.Content[0].Text, noa.NoRangesMessage)
	}
	if sess.attempts != before {
		t.Fatalf("attempts = %d, want it unchanged at %d", sess.attempts, before)
	}
}

func TestEnableAppendsResidentPromptOnly(t *testing.T) {
	opts := agentcore.Options{SystemPrompt: []string{"host"}}
	if err := Enable(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "s"}); err != nil {
		t.Fatalf("Enable: %v", err)
	}
	prompt := strings.Join(opts.AppendSystemPrompt, "\n")
	for _, want := range []string{"NOA TAGS", "COMPRESSION SUMMARIES IN CONTEXT", "THE ARCHIVE"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("resident prompt is missing %q", want)
		}
	}
	// Everything that teaches a one-off action travels with the nudge instead;
	// duplicating it here would double the cost for no benefit.
	for _, unwanted := range []string{"HOW TO COMPRESS", "WHEN TO COMPRESS", "TIER 2 COMPRESSION"} {
		if strings.Contains(prompt, unwanted) {
			t.Errorf("resident prompt carries %q, which belongs in the nudge", unwanted)
		}
	}
}
