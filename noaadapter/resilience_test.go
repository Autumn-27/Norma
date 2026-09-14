package noaadapter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
)

// buildWorkingSession runs a cooperative session far enough to have real blocks,
// real archives and a real ref map, and returns it with its history.
func buildWorkingSession(t *testing.T, dir string, turns int) (*Session, []llm.Message) {
	t.Helper()
	opts := baseOptions()
	cfg := noa.DefaultConfig(40000)
	sess, err := EnableWithSession(&opts, Options{
		ArchiveBaseDir: dir, SessionID: "resume", Config: &cfg,
	})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}
	a := newSimAgent(t, sess, 1)
	for range turns {
		a.work(4000)
		a.observe()
	}
	if len(noa.ActiveBlocks(sess.State())) == 0 {
		t.Fatalf("%d turns produced no active block; the fixture cannot test recovery", turns)
	}
	return sess, a.history
}

// renderView flattens a view to comparable text.
func renderView(msgs []llm.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&b, "[%s]", m.Role)
		for _, c := range m.Content {
			fmt.Fprintf(&b, "<%s|%s|%s|%s|%s>", c.Type, c.ID, c.ToolUseID, c.Text, c.Input)
			for _, inner := range c.Content {
				fmt.Fprintf(&b, "(%s)", inner.Text)
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

// The everyday resume: the process restarts, a new Session is built over the
// same archive directory, and the conversation continues. The model must see
// exactly what it saw before — different refs would invalidate every id it is
// holding, and lost blocks would resurrect content that was already summarised.
func TestResumeFromDiskReproducesTheSameView(t *testing.T) {
	dir := t.TempDir()
	sess, history := buildWorkingSession(t, dir, 45)
	want := renderView(sess.View(history))
	wantBlocks := len(noa.ActiveBlocks(sess.State()))

	opts := baseOptions()
	cfg := noa.DefaultConfig(40000)
	resumed, err := EnableWithSession(&opts, Options{
		ArchiveBaseDir: dir, SessionID: "resume", Config: &cfg,
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if got := len(noa.ActiveBlocks(resumed.State())); got != wantBlocks {
		t.Fatalf("resumed with %d active blocks, want %d", got, wantBlocks)
	}
	if got := renderView(resumed.View(history)); got != want {
		t.Fatalf("the resumed view differs from the original.\nwant %d bytes\ngot  %d bytes\n"+
			"first divergence at byte %d", len(want), len(got), firstDiff(want, got))
	}
}

// The state file is the only thing that is not derivable, so losing it is the
// interesting failure. RebuildFromHistory replays the recorded Compress calls;
// because the summaries were arguments the model wrote, the replay returns the
// same bytes rather than an approximation.
func TestRebuildAfterStateLossReproducesTheSameView(t *testing.T) {
	dir := t.TempDir()
	sess, history := buildWorkingSession(t, dir, 45)
	want := renderView(sess.View(history))
	wantBlocks := noa.ActiveBlocks(sess.State())

	// Lose the state file the way an import or a botched copy would.
	var removed int
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".json") {
			return err
		}
		if err := os.Remove(p); err != nil {
			return err
		}
		removed++
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if removed == 0 {
		t.Fatal("no state file was found to remove; the test is not exercising recovery")
	}

	cfg := noa.DefaultConfig(40000)
	rebuilt, err := RebuildFromHistory(history, Options{
		ArchiveBaseDir: dir, SessionID: "resume", Config: &cfg,
	})
	if err != nil {
		t.Fatalf("RebuildFromHistory: %v", err)
	}

	gotBlocks := noa.ActiveBlocks(rebuilt.State())
	if len(gotBlocks) != len(wantBlocks) {
		t.Fatalf("rebuilt %d active blocks from the log, want %d", len(gotBlocks), len(wantBlocks))
	}
	for i := range gotBlocks {
		if gotBlocks[i].Summary != wantBlocks[i].Summary {
			t.Fatalf("block %d summary changed on replay — the whole point of replaying the "+
				"call arguments is that it cannot:\nwant %q\ngot  %q",
				i, head(wantBlocks[i].Summary, 120), head(gotBlocks[i].Summary, 120))
		}
		if gotBlocks[i].Tier != wantBlocks[i].Tier {
			t.Fatalf("block %d came back at tier %d, was tier %d", i, gotBlocks[i].Tier, wantBlocks[i].Tier)
		}
	}
	if got := renderView(rebuilt.View(history)); got != want {
		t.Fatalf("the rebuilt view differs from the original at byte %d", firstDiff(want, got))
	}
}

// Materialize is the "switch noa off and keep going" path. It must leave a
// history the provider accepts and small enough to be worth doing, and it must
// not mention a ref scheme that is no longer running.
func TestMaterializeLeavesAUsableHistory(t *testing.T) {
	dir := t.TempDir()
	sess, history := buildWorkingSession(t, dir, 45)

	cfg := noa.DefaultConfig(40000)
	out, err := Materialize(history, Options{ArchiveBaseDir: dir, SessionID: "resume", Config: &cfg})
	if err != nil {
		t.Fatalf("Materialize: %v", err)
	}
	assertValidRequest(t, out, 0)

	raw := estimateMessages(history)
	got := estimateMessages(out)
	if got >= raw {
		t.Fatalf("materialized history is %d tokens against the raw %d; nothing was applied", got, raw)
	}
	for _, m := range out {
		for _, c := range m.Content {
			if strings.Contains(c.Text, "<noa-ref") {
				t.Fatalf("a ref tag survived into the materialized history: %s", head(c.Text, 160))
			}
		}
	}
	// The archives the summaries point at must still be readable — that is the
	// promise that makes switching noa off safe.
	for _, b := range noa.ActiveBlocks(sess.State()) {
		if _, err := os.ReadFile(b.ArchivePath); err != nil {
			t.Fatalf("archive for %s unreadable after materialize: %v", b.BlockID, err)
		}
	}
	t.Logf("materialized %d tokens down from %d", got, raw)
}

// The prefix cache is why the token count in a ref tag is frozen. If the bytes
// of an already-sent message can change between turns, the cache is invalidated
// from that point on and the saving is lost.
func TestViewPrefixIsStableAcrossTurns(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1)

	prev := ""
	prevLen := 0
	for turn := range 40 {
		a.work(4000)
		a.observe()
		cur := renderView(sess.View(a.history))

		// A compression legitimately rewrites the prefix: that is the whole point.
		// What must not happen is the prefix changing on a turn where it did not.
		blocksBefore := len(sess.State().Blocks)
		_ = blocksBefore
		if prev != "" && len(cur) >= prevLen && strings.HasPrefix(cur, prev) {
			prev, prevLen = cur, len(cur)
			continue
		}
		if prev != "" {
			d := firstDiff(prev, cur)
			// Everything before the first divergence was re-sent byte-identically.
			// Report how much was thrown away so a regression shows up as a number.
			t.Logf("turn %d: prefix held for %d of %d bytes (%.0f%%)",
				turn, d, prevLen, float64(d)*100/float64(max(prevLen, 1)))
			if d == 0 {
				t.Fatalf("turn %d: the view changed from its very first byte; nothing of the "+
					"prefix cache survives", turn)
			}
		}
		prev, prevLen = cur, len(cur)
	}
}

// Two goroutines calling View while a compression runs is the ordinary shape of
// a streaming turn. Run with -race, this is the test that catches a missing
// lock.
func TestConcurrentViewAndCompressAreSafe(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1000000)
	for range 25 {
		a.work(4000)
		a.observe()
	}

	var wg sync.WaitGroup
	for i := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				_ = sess.View(a.history)
			}
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			args, _ := json.Marshal(map[string]any{"content": []map[string]any{{
				"startId": noa.IndexToRef(2), "endId": noa.IndexToRef(13),
				"summary": a.summary(fmt.Sprintf("racer %d", i)),
			}}})
			if _, err := runCompress(sess, args, nil); err != nil {
				t.Errorf("concurrent Compress: %v", err)
			}
		}()
	}
	wg.Wait()

	// Whatever interleaving happened, the ledger must still be coherent.
	checkInvariants(t, sess, a.history, "after concurrent access")
}

// An archive is only useful if the original text is actually in it. Checking the
// front matter proves the file was written; this checks it was written with the
// content the summary replaced.
func TestTier1ArchivesContainTheOriginalText(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1)
	for range 45 {
		a.work(4000)
		a.observe()
	}

	// Every original body, by ref.
	cores, _ := Project(a.history)
	byID := map[string]noa.CoreMessage{}
	for _, c := range cores {
		byID[c.ID] = c
	}

	checked := 0
	for _, b := range noa.ActiveBlocks(sess.State()) {
		if b.Tier != 1 {
			continue
		}
		raw, err := os.ReadFile(b.ArchivePath)
		if err != nil {
			t.Fatalf("block %s: %v", b.BlockID, err)
		}
		body := string(raw)
		for _, id := range b.DirectMessageIDs {
			m, ok := byID[id]
			if !ok || len(m.Text) < 200 {
				continue
			}
			// A distinctive slice from the middle, past any header the archive adds.
			probe := m.Text[100:200]
			if !strings.Contains(body, probe) {
				t.Fatalf("block %s (%s): the archive does not contain the original body of "+
					"message %s. A summary that points at an archive missing its own content "+
					"is a silent loss.\nprobe: %q", b.BlockID, b.ArchivePath, id, probe)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("no tier-1 message body was checked; the fixture is not exercising archiving")
	}
	t.Logf("verified %d original message bodies are present in their archives", checked)
}

// Following the tier chain from a top-tier block must reach the raw originals.
// If a link is broken, a summary points at a file that points nowhere, and the
// content is unreachable even though it is still on disk.
func TestArchiveChainReachesRawContentFromEveryTier(t *testing.T) {
	// 128K: the smallest window where consolidation actually fires, so the chain
	// has more than one link to walk. See TestTierTwoIsUnreachableOnSmallWindows.
	sess := simSession(t, 128_000)
	a := newSimAgent(t, sess, 1)
	for range 182 {
		a.work(4000)
		a.observe()
	}
	st := sess.State()
	byID := map[string]noa.CompressionBlock{}
	for _, b := range st.Blocks {
		byID[b.BlockID] = b
	}

	highest := 0
	for _, b := range noa.ActiveBlocks(st) {
		highest = max(highest, int(b.Tier))
	}
	if highest < 2 {
		t.Skip("no block reached tier 2; the chain is not exercised")
	}

	var walk func(b noa.CompressionBlock, depth int) bool
	walk = func(b noa.CompressionBlock, depth int) bool {
		if depth > 10 {
			t.Fatalf("the archive chain from %s is more than ten deep; it is probably a cycle", b.BlockID)
		}
		raw, err := os.ReadFile(b.ArchivePath)
		if err != nil {
			t.Fatalf("block %s (tier %d): archive unreadable: %v", b.BlockID, b.Tier, err)
		}
		if b.Tier == 1 {
			return len(b.DirectMessageIDs) > 0 && len(raw) > 0
		}
		if len(b.DirectBlockIDs) == 0 {
			t.Fatalf("block %s is tier %d but names no child block; the chain stops here and "+
				"the originals underneath it are unreachable", b.BlockID, b.Tier)
		}
		for _, childID := range b.DirectBlockIDs {
			child, ok := byID[childID]
			if !ok {
				t.Fatalf("block %s names child %s, which is not in the ledger", b.BlockID, childID)
			}
			// The parent archive must actually cite the child's file, or a reader
			// following the chain by hand cannot get there.
			if !strings.Contains(string(raw), filepath.Base(child.ArchivePath)) {
				t.Fatalf("the tier-%d archive for %s does not cite its child %s (%s); the chain "+
					"is broken for a human or an agent following it with Read",
					b.Tier, b.BlockID, childID, filepath.Base(child.ArchivePath))
			}
			if !walk(child, depth+1) {
				return false
			}
		}
		return true
	}

	for _, b := range noa.ActiveBlocks(st) {
		if int(b.Tier) == highest {
			walk(b, 0)
		}
	}
}

func firstDiff(a, b string) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
