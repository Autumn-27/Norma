package noaadapter

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

// compressedSession returns a session with one real block, plus the history it
// was built from and the options needed to reopen it.
func compressedSession(t *testing.T) (Options, []llm.Message, *Session) {
	t.Helper()
	dir := t.TempDir()
	o := Options{ArchiveBaseDir: dir, SessionID: "sess"}
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, o)
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

	sess.View(msgs)
	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00002", "endId": "m00005", "summary": longSummary}},
	})
	res, err := runCompress(sess, args, &tool.ToolContext{ToolUseID: "toolu_1"})
	if err != nil || noa.PanelBlockCount(res.Content[0].Text) != 1 {
		t.Fatalf("setup compression failed: %v / %s", err, res.Content[0].Text)
	}
	return o, msgs, sess
}

// With noa off, the full history would be sent and overflow at once; applying
// the state keeps it the size it was compressed to.
func TestMaterializeAppliesCompression(t *testing.T) {
	o, msgs, _ := compressedSession(t)

	out, err := Materialize(msgs, o)
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if len(out) >= len(msgs) {
		t.Fatalf("materialized to %d messages, want fewer than the %d original", len(out), len(msgs))
	}
	found := false
	for _, m := range out {
		if strings.HasPrefix(m.Text(), noa.SummaryHeader) {
			found = true
			// The archive pointer must survive: following a compression back to
			// its originals is a plain file read and does not need noa running.
			if !strings.Contains(m.Text(), "<noa-archive") {
				t.Fatalf("the summary lost its archive pointer: %q", m.Text())
			}
		}
	}
	if !found {
		t.Fatal("no summary in the materialized output")
	}
}

// Ref tags address a scheme that is no longer running; leaving them would
// invite the model to use ids nothing will resolve.
func TestMaterializeEmitsNoRefTags(t *testing.T) {
	o, msgs, _ := compressedSession(t)
	out, err := Materialize(msgs, o)
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	for _, m := range out {
		if strings.Contains(m.Text(), "<noa-ref") {
			t.Fatalf("a ref tag survived materialization: %q", m.Text())
		}
	}
}

func TestMaterializeIsIdempotent(t *testing.T) {
	o, msgs, _ := compressedSession(t)
	first, err := Materialize(msgs, o)
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	second, err := Materialize(msgs, o)
	if err != nil {
		t.Fatalf("Materialize (second): %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("lengths differ between runs: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Text() != second[i].Text() {
			t.Fatalf("message %d differs between runs:\n%q\n%q", i, first[i].Text(), second[i].Text())
		}
	}
}

// Safe to call unconditionally: with nothing compressed there is nothing to do.
func TestMaterializeWithoutBlocksIsPassthrough(t *testing.T) {
	o := Options{ArchiveBaseDir: t.TempDir(), SessionID: "fresh"}
	msgs := []llm.Message{userText("hello"), userText("world")}
	out, err := Materialize(msgs, o)
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	if len(out) != len(msgs) {
		t.Fatalf("got %d messages, want the input returned unchanged", len(out))
	}
}

// The transcript and the block ledger must be untouched, so switching noa back
// on costs nothing.
func TestMaterializeLeavesStateAndHistoryAlone(t *testing.T) {
	o, msgs, sess := compressedSession(t)
	before := sess.State()
	stateFile := filepath.Join(o.ArchiveBaseDir, o.SessionID, "state.json")
	raw1, err := os.ReadFile(stateFile)
	if err != nil {
		t.Fatalf("read state: %v", err)
	}
	originalFirst := msgs[0].Text()

	if _, err := Materialize(msgs, o); err != nil {
		t.Fatalf("Materialize: %v", err)
	}

	raw2, _ := os.ReadFile(stateFile)
	if string(raw1) != string(raw2) {
		t.Fatal("Materialize rewrote state.json; it must be a pure function")
	}
	if msgs[0].Text() != originalFirst {
		t.Fatal("Materialize mutated the input history")
	}

	// Re-enabling must find the ledger intact.
	opts := baseOptions()
	again, err := EnableWithSession(&opts, o)
	if err != nil {
		t.Fatalf("re-enable: %v", err)
	}
	if len(again.State().Blocks) != len(before.Blocks) {
		t.Fatalf("blocks after re-enable = %d, want %d", len(again.State().Blocks), len(before.Blocks))
	}
}

func TestMaterializeRequiresArchiveDir(t *testing.T) {
	if _, err := Materialize(nil, Options{SessionID: "s"}); err == nil {
		t.Fatal("Materialize accepted an empty ArchiveBaseDir")
	}
}

// Losing state.json costs a replay, not the data: the summaries live in the
// recorded call arguments and come back byte-identical.
func TestRebuildFromHistoryRecoversBlocks(t *testing.T) {
	o, msgs, sess := compressedSession(t)
	original := sess.State()

	// Simulate the conversation as the transcript holds it: the compression
	// call and its result are real messages.
	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00002", "endId": "m00005", "summary": longSummary}},
	})
	history := append([]llm.Message(nil), msgs...)
	history = append(history,
		llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: llm.BlockToolUse, ID: "toolu_1", Name: noa.CompressToolName, Input: args,
		}}},
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: "toolu_1",
			Content: []llm.ContentBlock{llm.TextBlock("▣ noa | 12K → 3K tokens (~9K reclaimed)\n  b1(T1)=m00002–m00005 → /a/b1.md")},
		}}},
	)

	// Wipe the state file and rebuild.
	if err := os.Remove(filepath.Join(o.ArchiveBaseDir, o.SessionID, "state.json")); err != nil {
		t.Fatalf("remove state: %v", err)
	}
	fresh := Options{ArchiveBaseDir: o.ArchiveBaseDir, SessionID: o.SessionID}
	rebuilt, err := RebuildFromHistory(history, fresh)
	if err != nil {
		t.Fatalf("RebuildFromHistory: %v", err)
	}
	blocks := rebuilt.State().Blocks
	if len(blocks) != 1 {
		t.Fatalf("rebuilt %d blocks, want 1", len(blocks))
	}
	if blocks[0].Summary != original.Blocks[0].Summary {
		t.Fatal("the rebuilt summary differs — replay must return the byte-identical text, not a regeneration")
	}
	if !rebuilt.store.Exists() {
		t.Fatal("the rebuilt state was not persisted")
	}
}

// A call that failed then would fail identically now; replaying it is pointless.
func TestRebuildSkipsFailedCalls(t *testing.T) {
	dir := t.TempDir()
	o := Options{ArchiveBaseDir: dir, SessionID: "s"}
	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00001", "endId": "m00002", "summary": longSummary}},
	})
	history := []llm.Message{
		userText("the task"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: llm.BlockToolUse, ID: "failed", Name: noa.CompressToolName, Input: args,
		}}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: "failed",
			Content: []llm.ContentBlock{llm.TextBlock("▣ noa | 0 blocks created\n  error: Summary too short")},
		}}},
	}
	sess, err := RebuildFromHistory(history, o)
	if err != nil {
		t.Fatalf("RebuildFromHistory: %v", err)
	}
	if len(sess.State().Blocks) != 0 {
		t.Fatalf("rebuilt %d blocks from a failed call, want 0", len(sess.State().Blocks))
	}
}

// Existing archives are historical fact; a re-render would be a different file
// describing the same moment.
func TestReuseArchiverAdoptsExistingFiles(t *testing.T) {
	root := t.TempDir()
	a := NewReuseArchiver(root)
	abs, rel, err := a.Write(1, "b1", "m00001", "m00009", []byte("original content"))
	if err != nil {
		t.Fatalf("first write: %v", err)
	}
	abs2, rel2, err := a.Write(1, "b1", "m00001", "m00009", []byte("DIFFERENT content"))
	if err != nil {
		t.Fatalf("second write: %v", err)
	}
	if abs != abs2 || rel != rel2 {
		t.Fatalf("paths differ: %s/%s vs %s/%s", abs, rel, abs2, rel2)
	}
	raw, _ := os.ReadFile(abs)
	if string(raw) != "original content" {
		t.Fatalf("the existing archive was overwritten: %q", raw)
	}
	// Remove is a no-op: a replay must not delete archives it did not create.
	if err := a.Remove(abs); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := os.Stat(abs); err != nil {
		t.Fatal("a replay deleted a pre-existing archive")
	}
}

func TestFileArchiverWritesAtomically(t *testing.T) {
	root := t.TempDir()
	a := NewFileArchiver(root)
	abs, rel, err := a.Write(2, "b5", "m00012", "m00160", []byte("archive body"))
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if !strings.HasSuffix(rel, ".md") || !strings.HasPrefix(rel, "tier2/") {
		t.Fatalf("rel = %q, want tier2/<name>.md", rel)
	}
	raw, err := os.ReadFile(abs)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(raw) != "archive body" {
		t.Fatalf("content = %q", raw)
	}
	// No temp files may be left behind.
	entries, _ := os.ReadDir(filepath.Join(root, "tier2"))
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".noa-") {
			t.Fatalf("a temp file survived: %s", e.Name())
		}
	}
	if err := a.Remove(abs); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := os.Stat(abs); !os.IsNotExist(err) {
		t.Fatal("Remove did not delete the archive")
	}
	// Removing a missing archive is fine: rollback runs over files that may
	// never have landed.
	if err := a.Remove(abs); err != nil {
		t.Fatalf("Remove of a missing file = %v, want nil", err)
	}
}
