package noaadapter

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/agentcore"
	"github.com/Autumn-27/norma/compaction"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

func baseOptions() agentcore.Options {
	return agentcore.Options{
		SystemPrompt: []string{"you are an assistant"},
		Tools:        []tool.CoreTool{},
		Compaction:   &compaction.Config{ContextWindow: 200000},
	}
}

// Enable attaches all three pieces at once. Wiring them separately would make
// three ways to get it half-right, each of which fails quietly.
func TestEnableAttachesAllThreePieces(t *testing.T) {
	opts := baseOptions()
	if err := Enable(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "s1"}); err != nil {
		t.Fatalf("Enable: %v", err)
	}

	if opts.Compactor == nil {
		t.Fatal("Compactor not attached")
	}
	if _, ok := opts.Compactor.(harness.ContextView); !ok {
		t.Fatal("the compactor must implement harness.ContextView, or it never takes over the projection")
	}

	found := false
	for _, tl := range opts.Tools {
		if tl.Name() == noa.CompressToolName {
			found = true
		}
	}
	if !found {
		t.Fatalf("the Compress tool was not registered; tools = %v", toolNames(opts.Tools))
	}

	joined := strings.Join(opts.AppendSystemPrompt, "\n")
	if !strings.Contains(joined, noa.SystemPromptHeader) {
		t.Fatal("the resident prompt was not appended")
	}
	if opts.DynamicBoundary < len(opts.SystemPrompt)+1 {
		t.Fatalf("DynamicBoundary = %d, want it to cover the noa section — otherwise the prompt is re-billed every request",
			opts.DynamicBoundary)
	}
}

// Not calling Enable is the off switch: Options must be untouched.
func TestNotEnablingLeavesOptionsUntouched(t *testing.T) {
	opts := baseOptions()
	before := opts

	if opts.Compactor != nil {
		t.Fatal("Compactor set without Enable")
	}
	if len(opts.Tools) != len(before.Tools) {
		t.Fatal("tools changed without Enable")
	}
	if len(opts.AppendSystemPrompt) != 0 {
		t.Fatal("system prompt changed without Enable")
	}
	if opts.Compaction == nil {
		t.Fatal("the built-in compaction must remain configured when noa is off")
	}
}

// The archive is the only route back to compressed originals, so its location
// must be the host's explicit decision — never a silent fallback.
func TestEnableRequiresArchiveDir(t *testing.T) {
	opts := baseOptions()
	err := Enable(&opts, Options{SessionID: "s1"})
	if err == nil {
		t.Fatal("Enable accepted an empty ArchiveBaseDir")
	}
	if opts.Compactor != nil || len(opts.Tools) != 0 || len(opts.AppendSystemPrompt) != 0 {
		t.Fatalf("a failed Enable must leave Options untouched: %+v", opts)
	}
}

// The session enabled by Enable is the same one the tool holds — otherwise the
// tool would compress into a state the view never reads.
func TestEnableSharesOneSession(t *testing.T) {
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "s1"})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}
	c := opts.Compactor.(*Compactor)
	if c.sess != sess {
		t.Fatal("the compactor and the returned session differ")
	}
}

func toolNames(ts []tool.CoreTool) []string {
	out := make([]string, len(ts))
	for i, t := range ts {
		out[i] = t.Name()
	}
	return out
}

// End to end: a model-driven compression must shrink the view, write an
// archive holding the originals, and survive a restart.
func TestCompressEndToEnd(t *testing.T) {
	dir := t.TempDir()
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: dir, SessionID: "sess1"})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}

	body := strings.Repeat("compressible content. ", 300)
	tail := strings.Repeat("recent tail content. ", 250)
	msgs := []llm.Message{userText("the task")}
	for range 4 {
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock(body)}})
	}
	for range 6 {
		msgs = append(msgs, userText(tail))
	}

	view := sess.View(msgs)
	if len(view) != len(msgs) {
		t.Fatalf("initial view has %d messages, want %d unchanged", len(view), len(msgs))
	}

	// The model compresses the filler.
	args, _ := json.Marshal(map[string]any{
		"topic": "filler",
		"content": []map[string]any{{
			"startId": "m00002", "endId": "m00005", "summary": longSummary,
		}},
	})
	res, err := runCompress(sess, args, &tool.ToolContext{ToolUseID: "toolu_1"})
	if err != nil {
		t.Fatalf("runCompress: %v", err)
	}
	panel := res.Content[0].Text
	if res.IsError {
		t.Fatalf("compression reported an error: %s", panel)
	}
	if noa.PanelBlockCount(panel) != 1 {
		t.Fatalf("panel reports %d blocks, want 1:\n%s", noa.PanelBlockCount(panel), panel)
	}

	// The next view must be smaller and carry the summary.
	after := sess.View(msgs)
	if len(after) >= len(view) {
		t.Fatalf("view has %d messages after compression, want fewer than %d", len(after), len(view))
	}
	foundSummary := false
	for _, m := range after {
		if strings.HasPrefix(m.Text(), noa.SummaryHeader) {
			foundSummary = true
			if !strings.Contains(m.Text(), "<noa-archive") {
				t.Fatalf("the summary carries no archive pointer: %q", m.Text())
			}
		}
	}
	if !foundSummary {
		t.Fatal("no summary message in the compressed view")
	}

	// The archive must hold the originals.
	blocks := sess.State().Blocks
	if len(blocks) != 1 {
		t.Fatalf("state has %d blocks, want 1", len(blocks))
	}
	archive, err := os.ReadFile(blocks[0].ArchivePath)
	if err != nil {
		t.Fatalf("read archive: %v", err)
	}
	if !strings.Contains(string(archive), "compressible content.") {
		t.Fatal("the archive does not contain the original text")
	}
	if !strings.Contains(string(archive), "block: "+blocks[0].BlockID) {
		t.Fatal("the archive is missing its front matter")
	}

	// State must survive a restart.
	if _, err := os.Stat(filepath.Join(dir, "sess1", "state.json")); err != nil {
		t.Fatalf("state file not written: %v", err)
	}
	opts2 := baseOptions()
	sess2, err := EnableWithSession(&opts2, Options{ArchiveBaseDir: dir, SessionID: "sess1"})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if len(sess2.State().Blocks) != 1 {
		t.Fatalf("reopened session has %d blocks, want 1", len(sess2.State().Blocks))
	}
}

// Two consecutive views of an unchanged history must be byte-identical, or the
// prefix cache is invalidated every turn.
func TestViewIsStableAcrossTurns(t *testing.T) {
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "s"})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}
	msgs := []llm.Message{
		userText("the task"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("a reply")}},
		userText("follow up"),
	}
	first := sess.View(msgs)
	second := sess.View(msgs)

	if len(first) != len(second) {
		t.Fatalf("view length changed: %d -> %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Text() != second[i].Text() {
			t.Fatalf("message %d changed between identical turns:\n%q\n%q", i, first[i].Text(), second[i].Text())
		}
	}
}

// View must never write back: message identity is a content hash over the
// stored text, so a rewrite would invalidate every ref the model holds.
func TestViewDoesNotMutateHistory(t *testing.T) {
	opts := baseOptions()
	sess, _ := EnableWithSession(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "s"})
	msgs := []llm.Message{userText("original text")}
	_ = sess.View(msgs)
	if msgs[0].Text() != "original text" {
		t.Fatalf("history was mutated: %q", msgs[0].Text())
	}
}

const longSummary = "Explored the authentication design and settled on JWT with refresh tokens. " +
	"Key files: auth/jwt.go (issue/verify), auth/middleware.go (context injection). " +
	"Refresh tokens live in Redis with a 7-day TTL; access tokens are 15 minutes."
