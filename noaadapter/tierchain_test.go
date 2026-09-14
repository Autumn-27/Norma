package noaadapter

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

// compress runs one Compress call and returns the panel.
func compress(t *testing.T, sess *Session, callID string, ranges ...map[string]any) string {
	t.Helper()
	args, _ := json.Marshal(map[string]any{"content": ranges})
	res, err := runCompress(sess, args, &tool.ToolContext{ToolUseID: callID})
	if err != nil {
		t.Fatalf("runCompress: %v", err)
	}
	return res.Content[0].Text
}

// A tier-2 archive must be a step in a chain, not a dead end: it holds the
// absorbed summaries, each pointing at its own archive, so a reader can walk
// down to the original messages one level at a time.
func TestTierChainIsTraceableToOriginals(t *testing.T) {
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "chain"})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}

	unique := "UNIQUE-ORIGINAL-MARKER"
	body := strings.Repeat("filler content. ", 300)
	tail := strings.Repeat("recent tail. ", 400)

	msgs := []llm.Message{userText("the task")}
	for i := range 6 {
		text := body
		if i == 0 {
			text = unique + " " + body
		}
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextBlock(text)}})
	}
	for range 6 {
		msgs = append(msgs, userText(tail))
	}

	// Two tier-1 compressions.
	sess.View(msgs)
	p1 := compress(t, sess, "c1", map[string]any{
		"startId": "m00002", "endId": "m00004", "summary": longSummary, "topic": "first half",
	})
	if noa.PanelBlockCount(p1) != 1 {
		t.Fatalf("first compression produced no block:\n%s", p1)
	}
	sess.View(msgs)
	p2 := compress(t, sess, "c2", map[string]any{
		"startId": "m00005", "endId": "m00007", "summary": longSummary, "topic": "second half",
	})
	if noa.PanelBlockCount(p2) != 1 {
		t.Fatalf("second compression produced no block:\n%s", p2)
	}

	blocks := sess.State().Blocks
	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2 tier-1 blocks", len(blocks))
	}

	// Consolidate them into a tier-2 block.
	sess.View(msgs)
	p3 := compress(t, sess, "c3", map[string]any{
		"startId": blocks[0].BlockID, "endId": blocks[1].BlockID,
		"summary": longSummary, "topic": "consolidated",
	})
	if noa.PanelBlockCount(p3) != 1 {
		t.Fatalf("consolidation produced no block:\n%s", p3)
	}

	var t2 *noa.CompressionBlock
	for i, b := range sess.State().Blocks {
		if b.Tier == 2 && b.Active {
			t2 = &sess.State().Blocks[i]
		}
	}
	if t2 == nil {
		t.Fatalf("no active tier-2 block after consolidation: %+v", sess.State().Blocks)
	}
	if len(t2.DirectBlockIDs) != 2 {
		t.Fatalf("the tier-2 block absorbed %d blocks, want 2", len(t2.DirectBlockIDs))
	}
	for _, id := range t2.DirectBlockIDs {
		if b := noa.FindBlock(ptr(sess.State()), id); b == nil || b.Active {
			t.Fatalf("absorbed block %s is still active", id)
		}
	}

	// Walk the chain: tier-2 archive -> tier-1 archive -> the original text.
	t2body, err := os.ReadFile(t2.ArchivePath)
	if err != nil {
		t.Fatalf("read tier-2 archive: %v", err)
	}
	if strings.Contains(string(t2body), unique) {
		t.Fatal("the tier-2 archive holds raw message text; it should hold the absorbed summaries and point onward")
	}

	var childPaths []string
	for _, line := range strings.Split(string(t2body), "\n") {
		if strings.HasPrefix(line, "> 绝对路径：`") {
			p := strings.TrimSuffix(strings.TrimPrefix(line, "> 绝对路径：`"), "`")
			childPaths = append(childPaths, p)
		}
	}
	if len(childPaths) != 2 {
		t.Fatalf("the tier-2 archive names %d child archives, want 2:\n%s", len(childPaths), t2body)
	}

	foundOriginal := false
	for _, p := range childPaths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("read child archive %s: %v", p, err)
		}
		if strings.Contains(string(raw), unique) {
			foundOriginal = true
		}
	}
	if !foundOriginal {
		t.Fatal("the original text is not reachable by following the archive chain")
	}
}

// Mixing tiers in one range absorbs only the lowest; the higher-tier block
// stays active and the panel explains why.
func TestMixedTierCompressionLeavesHigherTierActive(t *testing.T) {
	opts := baseOptions()
	sess, err := EnableWithSession(&opts, Options{ArchiveBaseDir: t.TempDir(), SessionID: "mixed"})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}

	body := strings.Repeat("filler content. ", 300)
	tail := strings.Repeat("recent tail. ", 400)
	msgs := []llm.Message{userText("the task")}
	for range 9 {
		msgs = append(msgs, llm.Message{Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{llm.TextBlock(body)}})
	}
	for range 6 {
		msgs = append(msgs, userText(tail))
	}

	// Three tier-1 blocks.
	for i, span := range [][2]string{{"m00002", "m00003"}, {"m00004", "m00005"}, {"m00006", "m00007"}} {
		sess.View(msgs)
		p := compress(t, sess, "c"+string(rune('1'+i)), map[string]any{
			"startId": span[0], "endId": span[1], "summary": longSummary,
		})
		if noa.PanelBlockCount(p) != 1 {
			t.Fatalf("compression %d produced no block:\n%s", i, p)
		}
	}
	ids := blockIDs(sess.State(), 1)
	if len(ids) != 3 {
		t.Fatalf("got %v, want three tier-1 blocks", ids)
	}

	// Promote the middle one to tier 2 on its own.
	sess.View(msgs)
	p := compress(t, sess, "promote", map[string]any{
		"startId": ids[1], "endId": ids[1], "summary": longSummary,
	})
	if noa.PanelBlockCount(p) != 1 {
		t.Fatalf("promotion produced no block:\n%s", p)
	}
	t2ids := blockIDs(sess.State(), 2)
	if len(t2ids) != 1 {
		t.Fatalf("got %v tier-2 blocks, want exactly one", t2ids)
	}

	// Now span everything: the tier-2 block sits inside the range but must not
	// be absorbed.
	remaining := blockIDs(sess.State(), 1)
	if len(remaining) != 2 {
		t.Fatalf("got %v remaining tier-1 blocks, want 2", remaining)
	}
	sess.View(msgs)
	panel := compress(t, sess, "mix", map[string]any{
		"startId": remaining[0], "endId": remaining[len(remaining)-1],
		"summary": longSummary, "topic": "mixed span",
	})
	if noa.PanelBlockCount(panel) != 1 {
		t.Fatalf("the mixed-tier compression produced no block:\n%s", panel)
	}
	if b := noa.FindBlock(ptr(sess.State()), t2ids[0]); b == nil || !b.Active {
		t.Fatalf("%s is a tier above the target and must stay active", t2ids[0])
	}
	if !strings.Contains(panel, "was inside the requested range but was not consumed") {
		t.Fatalf("the panel must explain why the higher-tier block survived:\n%s", panel)
	}
}

func blockIDs(st noa.CompressionState, tier noa.Tier) []string {
	var out []string
	for _, b := range st.Blocks {
		if b.Active && b.Tier == tier {
			out = append(out, b.BlockID)
		}
	}
	return out
}

func ptr(st noa.CompressionState) *noa.CompressionState { return &st }
