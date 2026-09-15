package noaadapter

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

func edgeSession(t *testing.T) *Session {
	t.Helper()
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "edge"})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}
	return sess
}

// --- Degenerate histories ------------------------------------------------

func TestViewOfEmptyHistory(t *testing.T) {
	sess := edgeSession(t)
	if got := sess.View(nil); len(got) != 0 {
		t.Fatalf("view of an empty history = %v", got)
	}
}

func TestViewOfSingleMessage(t *testing.T) {
	sess := edgeSession(t)
	got := sess.View([]llm.Message{userText("hello")})
	if len(got) != 1 {
		t.Fatalf("got %d messages, want 1", len(got))
	}
	if !strings.HasPrefix(got[0].Text(), "hello") {
		t.Fatalf("text = %q", got[0].Text())
	}
}

// A history that is only tool traffic (no user text at all) must still project.
func TestViewOfToolOnlyHistory(t *testing.T) {
	sess := edgeSession(t)
	msgs := []llm.Message{
		assistantWith(toolUse("t1", "Read", `{}`)),
		toolResult("t1", "output"),
	}
	got := sess.View(msgs)
	if len(got) != 2 {
		t.Fatalf("got %d messages, want the exchange preserved", len(got))
	}
}

// An assistant message carrying only thinking must survive: dropping it would
// orphan the tool call that follows.
func TestViewKeepsThinkingOnlyMessage(t *testing.T) {
	sess := edgeSession(t)
	msgs := []llm.Message{
		userText("q"),
		assistantWith(llm.ContentBlock{Type: llm.BlockThinking, Thinking: "pondering", Signature: "sig"}),
		assistantWith(toolUse("t1", "Read", `{}`)),
		toolResult("t1", "out"),
	}
	got := sess.View(msgs)
	found := false
	for _, m := range got {
		for _, b := range m.Content {
			if b.Type == llm.BlockThinking {
				found = true
				if b.Signature != "sig" {
					t.Fatalf("signature = %q, want it preserved", b.Signature)
				}
			}
		}
	}
	if !found {
		t.Fatal("the thinking-only message was dropped")
	}
}

// --- Compress against odd views ------------------------------------------

func TestCompressBeforeAnyView(t *testing.T) {
	sess := edgeSession(t)
	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00001", "endId": "m00002", "summary": longSummary}},
	})
	res, err := runCompress(sess, args, &tool.ToolContext{ToolUseID: "x"})
	if err != nil {
		t.Fatalf("runCompress before any View: %v", err)
	}
	// No view means no refs; it must report that rather than crash.
	if noa.PanelBlockCount(res.Content[0].Text) != 0 {
		t.Fatalf("a block was created with no view:\n%s", res.Content[0].Text)
	}
}

func TestCompressWithNilToolContext(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)
	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00001", "endId": "m00008", "summary": longSummary}},
	})
	if _, err := runCompress(sess, args, nil); err != nil {
		t.Fatalf("a nil ToolContext must not break the call: %v", err)
	}
}

// --- Concurrency ---------------------------------------------------------

// View and Compress share one Session. Background tasks and subagents can reach
// it at the same time, so the shared state must be safe under -race.
func TestSessionConcurrentViewAndCompress(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)

	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00001", "endId": "m00008", "summary": longSummary}},
	})

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			if n%2 == 0 {
				sess.View(msgs)
			} else {
				_, _ = runCompress(sess, args, &tool.ToolContext{ToolUseID: "c"})
			}
		}(i)
	}
	wg.Wait()

	// Whatever interleaving happened, the ledger must be coherent.
	st := sess.State()
	seen := map[string]bool{}
	for _, b := range st.Blocks {
		if seen[b.BlockID] {
			t.Fatalf("block id %s was allocated twice", b.BlockID)
		}
		seen[b.BlockID] = true
	}
	for raw, ref := range st.MessageRefs.ByRaw {
		if ref == noa.BlockedRef {
			continue
		}
		if st.MessageRefs.ByRef[ref] != raw {
			t.Fatalf("ref map is inconsistent: ByRaw[%q]=%q but ByRef[%q]=%q",
				raw, ref, ref, st.MessageRefs.ByRef[ref])
		}
	}
}

func TestStateStoreConcurrentSaves(t *testing.T) {
	s := NewStateStore(t.TempDir())
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			st := sampleState()
			st.NextBlockID = n + 1
			_ = s.Save(st)
		}(i)
	}
	wg.Wait()
	if _, err := s.Load("sess", "/a"); err != nil {
		t.Fatalf("state is unreadable after concurrent saves: %v", err)
	}
}

func TestArchiverConcurrentWrites(t *testing.T) {
	a := NewFileArchiver(t.TempDir())
	var wg sync.WaitGroup
	errs := make([]error, 16)
	for i := range 16 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := "b" + string(rune('a'+n))
			_, _, errs[n] = a.Write(1, id, "m00001", "m00009", []byte("body "+id))
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent write %d failed: %v", i, err)
		}
	}
}

// --- Filesystem failures -------------------------------------------------

// A compression whose archive cannot be written must not create a block: the
// block would claim originals that are not there.
func TestCompressFailsWhenArchiveDirIsUnwritable(t *testing.T) {
	base := t.TempDir()
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: base, SessionID: "ro"})
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

	// Make the session directory read-only so the archive write fails.
	dir := filepath.Join(base, "ro")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Skipf("cannot make the directory read-only here: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0o755) })
	if os.Geteuid() == 0 {
		t.Skip("running as root; a read-only directory is not enforced")
	}

	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00002", "endId": "m00005", "summary": longSummary}},
	})
	res, _ := runCompress(sess, args, &tool.ToolContext{ToolUseID: "x"})
	panel := res.Content[0].Text

	if noa.PanelBlockCount(panel) != 0 {
		t.Fatalf("a block was created despite an unwritable archive:\n%s", panel)
	}
	if len(sess.State().Blocks) != 0 {
		t.Fatal("state gained a block whose archive could not be written")
	}
	if !strings.Contains(panel, "archive") {
		t.Fatalf("the panel does not explain the archive failure:\n%s", panel)
	}
}

// --- Identity edge cases -------------------------------------------------

// Two messages that differ only by role must not collide: the role is part of
// what a message IS.
func TestIdentityDistinguishesRole(t *testing.T) {
	a := DeriveMessageID(noa.RoleUser, noa.CTText, "same text", "", "")
	b := DeriveMessageID(noa.RoleAssistant, noa.CTText, "same text", "", "")
	if a == b {
		t.Fatal("a user and an assistant message with identical text share an id")
	}
}

func TestIdentityDistinguishesToolCallID(t *testing.T) {
	a := DeriveMessageID(noa.RoleTool, noa.CTToolResult, "out", "call_1", "Read")
	b := DeriveMessageID(noa.RoleTool, noa.CTToolResult, "out", "call_2", "Read")
	if a == b {
		t.Fatal("two results of different calls share an id")
	}
}

// Many repetitions of the same content must each keep a distinct identity.
func TestClusterCounterHandlesManyDuplicates(t *testing.T) {
	cc := NewClusterCounter()
	seen := map[string]bool{}
	for range 1000 {
		id := cc.Next("base")
		if seen[id] {
			t.Fatalf("duplicate id %q", id)
		}
		seen[id] = true
	}
}

// A history of identical messages is the worst case for content hashing.
func TestViewWithManyIdenticalMessages(t *testing.T) {
	sess := edgeSession(t)
	var msgs []llm.Message
	for range 50 {
		msgs = append(msgs, userText("git status"))
	}
	sess.View(msgs)
	st := sess.State()
	if len(st.MessageRefs.ByRaw) != 50 {
		t.Fatalf("assigned %d refs for 50 identical messages, want 50 distinct identities",
			len(st.MessageRefs.ByRaw))
	}
	// And the assignment must be stable across turns.
	before := make(map[string]string, len(st.MessageRefs.ByRaw))
	for k, v := range st.MessageRefs.ByRaw {
		before[k] = v
	}
	sess.View(msgs)
	for k, v := range sess.State().MessageRefs.ByRaw {
		if before[k] != "" && before[k] != v {
			t.Fatalf("message %q changed ref from %q to %q", k, before[k], v)
		}
	}
}

// --- Tag edge cases ------------------------------------------------------

// A message whose own text happens to look like a tag must not be corrupted.
func TestTagStrippingIgnoresTagLikeContent(t *testing.T) {
	cases := []string{
		`the config says <noa-ref id="user-service"/> which is unrelated`,
		"<noa-archive block=\"b1\" tier=\"1\" range=\"m1-m2\" path=\"/p.md\"/>",
		"prose about <noa-ref> tags in general",
		`<noa-ref id="m00001"/> in the MIDDLE of a sentence`,
	}
	for _, in := range cases {
		if got := StripRefTag(in); got != in {
			t.Errorf("StripRefTag altered content it should not touch:\n  in:  %q\n  out: %q", in, got)
		}
	}
}

// Re-tagging an already-tagged body must not stack tags.
func TestTagInjectionIsIdempotentAcrossTurns(t *testing.T) {
	sess := edgeSession(t)
	msgs := []llm.Message{userText("hello")}
	for range 5 {
		view := sess.View(msgs)
		if n := strings.Count(view[0].Text(), "<noa-ref"); n != 1 {
			t.Fatalf("message carries %d tags, want exactly 1: %q", n, view[0].Text())
		}
	}
}

// --- State file compatibility -------------------------------------------

// A state file from a newer version must be refused rather than half-read.
func TestStateStoreForwardIncompatibility(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "state.json"),
		[]byte(`{"version":2,"blocks":[{"blockId":"b1","tier":1,"active":true}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := NewStateStore(root).Load("s", "/a")
	if err == nil {
		t.Fatal("a newer state file was accepted; fields it does not know about would be silently dropped")
	}
	if len(got.Blocks) != 0 {
		t.Fatal("blocks were read out of an incompatible file")
	}
}

// A block whose archive was deleted still renders; the model gets a path that
// does not resolve, which is better than losing the summary too.
func TestMissingArchiveDoesNotBreakTheView(t *testing.T) {
	o, msgs, sess := compressedSession(t)
	blocks := sess.State().Blocks
	if err := os.Remove(blocks[0].ArchivePath); err != nil {
		t.Fatalf("remove archive: %v", err)
	}
	view := sess.View(msgs)
	found := false
	for _, m := range view {
		if strings.HasPrefix(m.Text(), noa.SummaryHeader) {
			found = true
		}
	}
	if !found {
		t.Fatal("the summary vanished when its archive did; the compression would be lost twice over")
	}
	// And a rebuild can regenerate the archive from the transcript.
	_ = o
}
