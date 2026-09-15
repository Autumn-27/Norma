package noaadapter

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

// The scripted simAgent models a model that either cooperates perfectly or does
// nothing at all. Real models do neither: they hallucinate refs, invert ranges,
// reach into the protected tail, re-compress what they already compressed, and
// occasionally emit something that is not a range at all.
//
// chaosAgent is that model. It is seeded, so a failure reproduces exactly.
//
// The point is not that any single call succeeds — most are meant to be
// rejected. The point is that after any sequence of them the GLOBAL invariants
// still hold, because those are what the rest of the system depends on.
type chaosAgent struct {
	*simAgent
	rng *rand.Rand
	// competence is how many of the twelve behaviour slots are the cooperative
	// one. 4 is a model that is wrong most of the time; 9 is a good model having
	// an ordinary bad day.
	competence int

	accepted int
	rejected int
	kinds    map[string]int
}

func newChaosAgent(t *testing.T, sess *Session, seed uint64) *chaosAgent {
	return &chaosAgent{
		simAgent:   newSimAgent(t, sess, 1),
		rng:        rand.New(rand.NewPCG(seed, seed^0x9e3779b9)),
		competence: 4,
		kinds:      map[string]int{},
	}
}

// step runs one turn: work, look at the view, then do something — usually
// something wrong.
func (c *chaosAgent) step() []llm.Message {
	c.t.Helper()
	c.work(1000 + c.rng.IntN(6000))

	view := c.sess.View(c.history)
	c.viewTokens = append(c.viewTokens, estimateMessages(view))
	nudge := findNudge(view)
	if nudge != "" {
		c.nudgesSeen++
	}

	kind, entries := c.invent(nudge)
	c.kinds[kind]++
	if entries == nil {
		return view
	}

	c.callSeq++
	callID := fmt.Sprintf("compress_%d", c.callSeq)
	args, _ := json.Marshal(map[string]any{"content": entries})
	res, err := runCompress(c.sess, args, &tool.ToolContext{ToolUseID: callID})
	if err != nil {
		c.t.Fatalf("turn %d (%s): Compress returned a Go error, which the harness cannot turn "+
			"into a tool result: %v", c.turn, kind, err)
	}
	if len(res.Content) == 0 || res.Content[0].Text == "" {
		c.t.Fatalf("turn %d (%s): Compress produced an empty result; the provider rejects those", c.turn, kind)
	}
	panel := res.Content[0].Text
	if n := noa.PanelBlockCount(panel); n > 0 {
		c.accepted += n
	} else {
		c.rejected++
	}

	c.history = append(c.history,
		llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: llm.BlockToolUse, ID: callID, Name: noa.CompressToolName, Input: args,
		}}},
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: callID,
			Content: []llm.ContentBlock{llm.TextBlock(panel)},
		}}},
	)
	return view
}

// invent picks a way to behave. Roughly a third of calls are legitimate, so the
// session makes real progress and the invariants are checked against a state
// that actually has blocks in it.
func (c *chaosAgent) invent(nudge string) (string, []map[string]any) {
	// rangeLineRE is anchored at the start of a line, so it has to be applied per
	// line — the same way simAgent.act reads the table.
	var rows [][]string
	for _, line := range strings.Split(nudge, "\n") {
		if m := rangeLineRE.FindStringSubmatch(line); m != nil {
			rows = append(rows, m)
		}
	}
	blocks := blockLineRE.FindAllStringSubmatch(nudge, -1)
	targets := tierTargets(nudge)

	roll := c.rng.IntN(12)
	if roll < c.competence {
		// Cooperate, when there is something to cooperate with.
		if len(rows) == 0 {
			return "nothing offered", nil
		}
		var out []map[string]any
		for _, m := range rows {
			out = append(out, map[string]any{
				"startId": m[1], "endId": m[2], "summary": c.summary(m[1] + "–" + m[2]),
			})
		}
		return "cooperative", out
	}

	switch roll {
	case 4:
		// Consolidate blocks, whether or not asked to. A tier nudge names its
		// targets explicitly; otherwise fall back to the block map.
		if len(targets) >= 2 {
			return "consolidate", []map[string]any{{
				"startId": targets[0], "endId": targets[len(targets)-1],
				"summary": c.summary("consolidation"),
			}}
		}
		if len(blocks) < 2 {
			return "no blocks to consolidate", nil
		}
		return "consolidate", []map[string]any{{
			"startId": blocks[0][1], "endId": blocks[len(blocks)-1][1],
			"summary": c.summary("consolidation"),
		}}

	case 5:
		// Refs that were never handed out.
		return "hallucinated refs", []map[string]any{{
			"startId": noa.IndexToRef(90000 + c.rng.IntN(9000)),
			"endId":   noa.IndexToRef(99000 + c.rng.IntN(900)),
			"summary": c.summary("invented"),
		}}

	case 6:
		// End before start.
		if len(rows) == 0 {
			return "nothing to invert", nil
		}
		m := rows[c.rng.IntN(len(rows))]
		return "inverted range", []map[string]any{{
			"startId": m[2], "endId": m[1], "summary": c.summary("backwards"),
		}}

	case 7:
		// Reach into the protected tail — the most recent messages.
		last := len(c.sess.State().MessageRefs.ByRef)
		if last < 4 {
			return "history too short", nil
		}
		return "into the protected tail", []map[string]any{{
			"startId": noa.IndexToRef(last - 2), "endId": noa.IndexToRef(last),
			"summary": c.summary("recent"),
		}}

	case 8:
		// Two overlapping ranges in one call.
		if len(rows) == 0 {
			return "nothing to overlap", nil
		}
		m := rows[0]
		return "overlapping ranges", []map[string]any{
			{"startId": m[1], "endId": m[2], "summary": c.summary("first")},
			{"startId": m[1], "endId": m[2], "summary": c.summary("second")},
		}

	case 9:
		// Re-compress something already consumed.
		if len(blocks) == 0 {
			return "nothing consumed yet", nil
		}
		b := blocks[c.rng.IntN(len(blocks))]
		return "re-compress consumed", []map[string]any{{
			"startId": b[3], "endId": b[4], "summary": c.summary("again"),
		}}

	case 10:
		// A summary that says nothing, and one that says far too much.
		if len(rows) == 0 {
			return "nothing to summarise", nil
		}
		m := rows[0]
		if c.rng.IntN(2) == 0 {
			return "empty summary", []map[string]any{{"startId": m[1], "endId": m[2], "summary": ""}}
		}
		return "enormous summary", []map[string]any{{
			"startId": m[1], "endId": m[2], "summary": strings.Repeat("verbose. ", 6000),
		}}

	default:
		return "ignored the nudge", nil
	}
}

// checkInvariants is the whole point of the chaos run. Every one of these is
// something another part of the system assumes without checking.
func checkInvariants(t *testing.T, sess *Session, history []llm.Message, when string) {
	t.Helper()
	st := sess.State()

	// 1. Block ids are never reused, and never exceed the allocator.
	seenBlock := map[string]bool{}
	for _, b := range st.Blocks {
		if seenBlock[b.BlockID] {
			t.Fatalf("%s: block id %s was issued twice", when, b.BlockID)
		}
		seenBlock[b.BlockID] = true
	}

	// 2. Active blocks partition what they cover: no message may be claimed by
	//    two live summaries, or the model would see it described twice and the
	//    token accounting would double-count.
	owner := map[string]string{}
	for _, b := range noa.ActiveBlocks(st) {
		for _, id := range b.EffectiveMessageIDs {
			if prev, dup := owner[id]; dup {
				t.Fatalf("%s: message %s is covered by both %s and %s", when, id, prev, b.BlockID)
			}
			owner[id] = b.BlockID
		}
	}

	// 3. Nothing is lost. Every original message must still be accounted for in
	//    one of the ways the design allows. Anything else has fallen out of the
	//    conversation entirely, which no amount of archiving can undo.
	//
	//    Identity is a content hash, so a message the valve truncated no longer
	//    hashes to its original id. It is still present, and its tool_use id is
	//    what survives the rewrite — so that is what it is matched on.
	cores, _ := Project(history)
	viewed, _ := Project(sess.View(history))
	inView := map[string]bool{}
	viewCallIDs := map[string]bool{}
	for _, c := range viewed {
		inView[c.ID] = true
		if c.ToolCallID != "" {
			viewCallIDs[c.ToolCallID] = true
		}
	}
	for _, c := range cores {
		switch {
		case inView[c.ID]:
		case owner[c.ID] != "":
		case c.ToolCallID != "" && viewCallIDs[c.ToolCallID]:
			// Present, but rewritten by emergency-truncate.
		case c.ToolName == noa.CompressToolName:
			// Compress bookkeeping. Calls whose block is live are stubbed, the last
			// KeepLastOrphaned failures are kept so the model can see its mistake,
			// and older failures are dropped as noise. All three are by design.
		default:
			t.Fatalf("%s: message %s (%s, tool %q) is neither in the view nor covered by "+
				"any block — it is simply gone", when, c.ID, c.ContentType, c.ToolName)
		}
	}

	// 4. Every active block's archive is on disk and names the block.
	for _, b := range noa.ActiveBlocks(st) {
		raw, err := os.ReadFile(b.ArchivePath)
		if err != nil {
			t.Fatalf("%s: block %s archive unreadable: %v", when, b.BlockID, err)
		}
		if !strings.Contains(string(raw), b.BlockID) {
			t.Fatalf("%s: block %s archive does not name the block", when, b.BlockID)
		}
	}

	// 5. The request the provider would receive is well-formed.
	assertValidRequest(t, sess.View(history), 0)
}

// The headline chaos test: a model behaving badly in every way the design
// anticipates, over a long session, must leave the invariants intact.
func TestChaosAgentPreservesInvariants(t *testing.T) {
	for _, seed := range []uint64{1, 7, 42, 1337, 90210} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			sess := simSession(t, 40000)
			c := newChaosAgent(t, sess, seed)
			for i := range 70 {
				c.step()
				if i%7 == 0 {
					checkInvariants(t, sess, c.history, fmt.Sprintf("turn %d", c.turn))
				}
			}
			checkInvariants(t, sess, c.history, "end")
			if c.accepted == 0 {
				t.Fatalf("70 chaotic turns produced no accepted compression at all; "+
					"the agent is not exercising the success path (kinds: %v)", c.kinds)
			}
			if c.rejected == 0 {
				t.Fatalf("no call was rejected; the agent is not exercising the failure path (kinds: %v)", c.kinds)
			}
			t.Logf("accepted=%d rejected=%d nudges=%d kinds=%v", c.accepted, c.rejected, c.nudgesSeen, c.kinds)
		})
	}
}

// A malformed call must never take the session down with it. Whatever the model
// sends, the tool answers with a tool result — never a Go error, which the
// harness would surface as a failed turn rather than something the model can
// correct.
func TestChaosAgentNeverProducesAGoError(t *testing.T) {
	sess := simSession(t, 40000)
	c := newChaosAgent(t, sess, 99)
	for range 40 {
		c.step() // step() fails the test if runCompress ever returns an error
	}
	if c.accepted+c.rejected == 0 {
		t.Fatalf("40 turns produced no Compress call at all; the fixture is degenerate (kinds: %v)", c.kinds)
	}
}

// A good model having an ordinary bad day — mostly right, occasionally wrong —
// is the realistic case, and it must still be held inside the window.
func TestMostlyCompetentModelStaysBounded(t *testing.T) {
	window := 40000
	for _, seed := range []uint64{3, 11, 2024} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			sess := simSession(t, window)
			c := newChaosAgent(t, sess, seed)
			c.competence = 9
			for range 90 {
				c.step()
			}
			tail := c.viewTokens[len(c.viewTokens)-15:]
			peak := 0
			for _, n := range tail {
				peak = max(peak, n)
			}
			if peak > window {
				t.Fatalf("the last fifteen turns peaked at %d over a %d window "+
					"(accepted=%d rejected=%d nudges=%d); occasional mistakes must not break "+
					"compression.\nThe mechanism: a couple of ignored or failed nudges trip the "+
					"attempt ladder, and suppression then lifts only after NudgeGrowthFloor "+
					"(%d tokens) of further growth — more than half this window. Nudging goes "+
					"quiet at ~90%% usage and does not resume before overflow. See "+
					"noa.TestSuppressionReleaseFitsInsideTheWindow.",
					peak, window, c.accepted, c.rejected, c.nudgesSeen,
					noa.NudgeGrowthFloor(sess.Config()))
			}
			t.Logf("tail peak %d of %d (accepted=%d rejected=%d nudges=%d)",
				peak, window, c.accepted, c.rejected, c.nudgesSeen)
		})
	}
}

// A model that tries on every nudge and gets it wrong every time is silenced as
// fast as one that never tries at all — the ladder counts a failed attempt and
// an ignored nudge the same way.
//
// That is deliberate (both reclaim nothing, and both loop just as expensively),
// but the consequence is worth pinning: an EAGER model gets no more chances than
// a silent one, so a model whose only problem is picking bad refs never gets the
// repeated guidance that would let it correct course, and the context runs past
// the window exactly as if it had never tried.
//
// If this test starts failing because the eager model gets materially more
// nudges, the ladder's policy has changed and the reasoning above needs revising
// rather than the numbers.
func TestFailedAttemptsAreSilencedLikeIgnoredOnes(t *testing.T) {
	window := 40000
	run := func(eager bool) (nudges, final int) {
		sess := simSession(t, window)
		a := newSimAgent(t, sess, 1000000)
		for i := range 60 {
			a.work(4000)
			view := sess.View(a.history)
			a.viewTokens = append(a.viewTokens, estimateMessages(view))
			if findNudge(view) == "" {
				continue
			}
			nudges++
			if !eager {
				continue
			}
			// Always willing, always wrong: refs that were never handed out.
			id := fmt.Sprintf("c_%d", i)
			args, _ := json.Marshal(map[string]any{"content": []map[string]any{{
				"startId": noa.IndexToRef(90001), "endId": noa.IndexToRef(90009),
				"summary": a.summary("unresolvable"),
			}}})
			res, err := runCompress(sess, args, &tool.ToolContext{ToolUseID: id})
			if err != nil {
				t.Fatalf("runCompress: %v", err)
			}
			a.history = append(a.history,
				llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
					Type: llm.BlockToolUse, ID: id, Name: noa.CompressToolName, Input: args}}},
				llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
					Type: llm.BlockToolResult, ToolUseID: id,
					Content: []llm.ContentBlock{llm.TextBlock(res.Content[0].Text)}}}})
		}
		return nudges, a.viewTokens[len(a.viewTokens)-1]
	}

	eagerNudges, eagerFinal := run(true)
	silentNudges, silentFinal := run(false)

	if eagerNudges > silentNudges*2 {
		t.Fatalf("the eager model got %d nudges against the silent model's %d; the ladder "+
			"now distinguishes them and this test's premise is stale", eagerNudges, silentNudges)
	}
	if eagerFinal <= window {
		t.Fatalf("the eager-but-wrong model ended at %d, inside the %d window; "+
			"something now rescues it and the documented limitation no longer holds",
			eagerFinal, window)
	}
	t.Logf("eager-but-wrong: %d nudges, ended at %d tokens; never tried: %d nudges, ended at %d "+
		"(window %d) — trying and failing buys nothing over staying silent",
		eagerNudges, eagerFinal, silentNudges, silentFinal, window)
}
