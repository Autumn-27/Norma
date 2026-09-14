package noaadapter

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

// simAgent drives a scripted conversation through noa the way a real agent
// would: it works, it watches the view, and when the context manager asks it to
// compress it reads the range table and acts on it.
//
// Nothing here reaches inside noa. The simulation goes through View, reads the
// rendered nudge as text, and calls the Compress tool with ordinary JSON — so a
// failure means the system genuinely does not work end to end, not that a test
// helper drifted from the implementation.
type simAgent struct {
	t        *testing.T
	sess     *Session
	history  []llm.Message
	turn     int
	callSeq  int
	obeyRate int // compress on 1 of every N nudges; 1 = always

	// Telemetry.
	viewTokens   []int
	nudgesSeen   int
	compressions int
	failures     int
}

func newSimAgent(t *testing.T, sess *Session, obeyRate int) *simAgent {
	t.Helper()
	return &simAgent{t: t, sess: sess, obeyRate: max(obeyRate, 1),
		history: []llm.Message{userText("Refactor the authentication subsystem and keep it tested.")}}
}

// work appends one turn of plausible agent activity: a thought, a tool call,
// and its output.
func (a *simAgent) work(outputChars int) {
	a.turn++
	callID := fmt.Sprintf("call_%d", a.turn)
	a.history = append(a.history,
		llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.BlockThinking, Thinking: fmt.Sprintf("Turn %d: checking the next file.", a.turn),
				Signature: fmt.Sprintf("sig-%d", a.turn)},
			llm.TextBlock(fmt.Sprintf("Reading file %d.", a.turn)),
			{Type: llm.BlockToolUse, ID: callID, Name: "Read",
				Input: json.RawMessage(fmt.Sprintf(`{"file_path":"/src/mod%d.go"}`, a.turn))},
		}},
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: callID,
			Content: []llm.ContentBlock{llm.TextBlock(
				fmt.Sprintf("// mod%d.go\n", a.turn) + strings.Repeat("source line of code. ", outputChars/21))},
		}}},
	)
}

// rangeLineRE matches the compressible rows of a rendered nudge.
var rangeLineRE = regexp.MustCompile(`^\s{2}(m\d{5})–(m\d{5})\s+\d+ msgs\s+\S+ \[tool`)

// blockLineRE matches the block map entries of a rendered nudge.
var blockLineRE = regexp.MustCompile(`(b\d+)\(T(\d+)\)=(m\d{5})–(m\d{5})`)

// observe renders the view and reacts to whatever the context manager says.
func (a *simAgent) observe() []llm.Message {
	a.t.Helper()
	view := a.sess.View(a.history)
	a.viewTokens = append(a.viewTokens, estimateMessages(view))

	nudge := findNudge(view)
	if nudge == "" {
		return view
	}
	a.nudgesSeen++
	if a.nudgesSeen%a.obeyRate != 0 {
		// Ignoring a nudge is a normal outcome and must not wedge the system.
		return view
	}
	a.act(nudge)
	return view
}

// act picks a target the way a cooperative model would: tier consolidation when
// asked for it, otherwise the ranges the table offered.
func (a *simAgent) act(nudge string) {
	a.t.Helper()
	var entries []map[string]any

	if strings.Contains(nudge, "DISTILLATION TRIGGER") || strings.Contains(nudge, "CONDENSATION TRIGGER") {
		blocks := blockLineRE.FindAllStringSubmatch(nudge, -1)
		if len(blocks) >= 2 {
			entries = append(entries, map[string]any{
				"startId": blocks[0][1], "endId": blocks[len(blocks)-1][1],
				"summary": a.summary("consolidated blocks"), "topic": "consolidation",
			})
		}
	}
	if len(entries) == 0 {
		for _, line := range strings.Split(nudge, "\n") {
			if m := rangeLineRE.FindStringSubmatch(line); m != nil {
				entries = append(entries, map[string]any{
					"startId": m[1], "endId": m[2],
					"summary": a.summary(m[1] + "–" + m[2]), "topic": "work batch",
				})
			}
		}
	}
	if len(entries) == 0 {
		return
	}

	a.callSeq++
	callID := fmt.Sprintf("compress_%d", a.callSeq)
	args, _ := json.Marshal(map[string]any{"content": entries})

	res, err := runCompress(a.sess, args, &tool.ToolContext{ToolUseID: callID})
	if err != nil {
		a.t.Fatalf("turn %d: Compress errored: %v", a.turn, err)
	}
	panel := res.Content[0].Text
	if n := noa.PanelBlockCount(panel); n > 0 {
		a.compressions += n
	} else {
		a.failures++
	}

	// A real agent's call and its result become part of the history.
	a.history = append(a.history,
		llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{
			Type: llm.BlockToolUse, ID: callID, Name: noa.CompressToolName, Input: args,
		}}},
		llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type: llm.BlockToolResult, ToolUseID: callID,
			Content: []llm.ContentBlock{llm.TextBlock(panel)},
		}}},
	)
}

// summary produces something that satisfies the length rules and reads like
// what the guidance asks for.
func (a *simAgent) summary(label string) string {
	return fmt.Sprintf("Summary of %s: read and reviewed several modules under /src. "+
		"Key files: auth/jwt.go:42 (token issue/verify), auth/middleware.go:18 (context injection). "+
		"Decision: keep JWT with refresh tokens because session cookies cannot cross the mobile client. "+
		"Open: the clock-skew leeway is still hard-coded at 60s.", label)
}

func findNudge(view []llm.Message) string {
	for _, m := range view {
		if strings.Contains(m.Text(), "HOW TO COMPRESS") {
			return m.Text()
		}
	}
	return ""
}

func simSession(t *testing.T, window int) *Session {
	t.Helper()
	opts := baseOptions()
	cfg := noa.DefaultConfig(window)
	sess, err := EnableWithSession(&opts, Options{
		ArchiveBaseDir: t.TempDir(), SessionID: "sim", Config: &cfg,
	})
	if err != nil {
		t.Fatalf("EnableWithSession: %v", err)
	}
	return sess
}

// The point of the whole system: a conversation far longer than the window runs
// to completion without the context growing without bound.
func TestLongSessionStaysBounded(t *testing.T) {
	window := 40000
	sess := simSession(t, window)
	a := newSimAgent(t, sess, 1)

	for range 60 {
		a.work(4000)
		a.observe()
	}

	if a.compressions == 0 {
		t.Fatal("60 turns of growth produced no compression at all")
	}
	peak := 0
	for _, n := range a.viewTokens {
		peak = max(peak, n)
	}
	if peak > window {
		t.Fatalf("view peaked at %d tokens, past the %d window", peak, window)
	}

	// The tail is what matters: early turns are allowed to grow freely.
	tail := a.viewTokens[len(a.viewTokens)-10:]
	tailPeak := 0
	for _, n := range tail {
		tailPeak = max(tailPeak, n)
	}
	if tailPeak > window*3/4 {
		t.Fatalf("the last ten turns peaked at %d tokens (>75%% of the %d window); "+
			"compression is not keeping up", tailPeak, window)
	}
	t.Logf("turns=%d nudges=%d compressions=%d failures=%d peak=%d tailPeak=%d blocks=%d",
		a.turn, a.nudgesSeen, a.compressions, a.failures, peak, tailPeak, len(sess.State().Blocks))
}

// Blocks accumulating is only a problem if their summaries start to dominate
// the context — that is the invariant worth holding, not a block count.
//
// Under pressure the tier with the most to reclaim wins, and while raw content
// keeps piling up that is tier 1. So a busy session legitimately carries many
// tier-1 blocks; what must not happen is the summaries themselves becoming the
// thing filling the window.
func TestLongSessionSummariesDoNotDominate(t *testing.T) {
	window := 40000
	sess := simSession(t, window)
	a := newSimAgent(t, sess, 1)
	for range 80 {
		a.work(4000)
		a.observe()
	}
	active := noa.ActiveBlocks(sess.State())
	if len(active) == 0 {
		t.Fatal("no active blocks after 80 turns")
	}
	summaryTokens := 0
	tiers := map[noa.Tier]int{}
	for _, b := range active {
		summaryTokens += noa.DefaultCountTokens(b.Summary)
		tiers[b.Tier]++
	}
	if summaryTokens > window/4 {
		t.Fatalf("summaries occupy %d of the %d window (>25%%) across %d blocks — "+
			"consolidation is not keeping up", summaryTokens, window, len(active))
	}
	t.Logf("active blocks by tier: %v (total %d, %d summary tokens = %.1f%% of window)",
		tiers, len(active), summaryTokens, float64(summaryTokens)*100/float64(window))
}

// Compression has to actually reclaim space, not merely rearrange it.
func TestCompressionReclaimsTokens(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1)
	for range 40 {
		a.work(4000)
		a.observe()
	}
	st := sess.State()
	var compressed, summaryCost int
	for _, b := range st.Blocks {
		compressed += b.CompressedTokens
		summaryCost += noa.DefaultCountTokens(b.Summary)
	}
	if compressed == 0 {
		t.Fatal("nothing was compressed")
	}
	if summaryCost >= compressed {
		t.Fatalf("summaries cost %d tokens to replace %d — compression reclaimed nothing", summaryCost, compressed)
	}
	ratio := float64(compressed) / float64(max(summaryCost, 1))
	if ratio < 2 {
		t.Fatalf("compression ratio %.1fx is too low to be worth a turn", ratio)
	}
	t.Logf("reclaimed %d tokens into %d (%.1fx) across %d blocks", compressed, summaryCost, ratio, len(st.Blocks))
}

// Every compressed range must be recoverable from disk, however deep the tier
// chain goes.
func TestLongSessionArchivesRemainComplete(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1)
	for range 50 {
		a.work(4000)
		a.observe()
	}
	st := sess.State()
	if len(st.Blocks) == 0 {
		t.Fatal("no blocks to check")
	}
	for _, b := range st.Blocks {
		raw, err := os.ReadFile(b.ArchivePath)
		if err != nil {
			t.Fatalf("block %s (tier %d): archive unreadable: %v", b.BlockID, b.Tier, err)
		}
		body := string(raw)
		if !strings.Contains(body, "block: "+b.BlockID) {
			t.Fatalf("block %s: archive front matter does not identify it", b.BlockID)
		}
		if b.Tier == 1 {
			// A tier-1 archive holds raw messages.
			if !strings.Contains(body, "· tool_result ·") && !strings.Contains(body, "· assistant") {
				t.Fatalf("block %s: tier-1 archive holds no original messages:\n%s", b.BlockID, head(body, 400))
			}
		} else {
			// A higher tier holds the summaries it absorbed, each pointing on.
			if !strings.Contains(body, "绝对路径") {
				t.Fatalf("block %s: tier-%d archive names no child archive:\n%s", b.BlockID, b.Tier, head(body, 400))
			}
		}
	}
}

// A model that only sometimes complies must not wedge the system, and the
// failure ladder must not silence nudging for good.
func TestLongSessionWithPartiallyCompliantModel(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 3) // acts on one nudge in three

	for range 80 {
		a.work(4000)
		a.observe()
	}
	if a.compressions == 0 {
		t.Fatal("a partially compliant model produced no compression at all")
	}
	// The mechanical valve must have held the line where the model did not.
	peak := 0
	for _, n := range a.viewTokens {
		peak = max(peak, n)
	}
	if peak > 40000 {
		t.Fatalf("view peaked at %d, past the window — the mechanical valve did not hold", peak)
	}
	t.Logf("partial compliance: nudges=%d compressions=%d peak=%d", a.nudgesSeen, a.compressions, peak)
}

// A model that never compresses is the worst case the design has to survive.
func TestLongSessionWithUncooperativeModel(t *testing.T) {
	window := 40000
	sess := simSession(t, window)
	a := newSimAgent(t, sess, 1000000) // never acts

	for range 60 {
		a.work(4000)
		a.observe()
	}
	if a.compressions != 0 {
		t.Fatal("the uncooperative model compressed something; the fixture is wrong")
	}
	peak := 0
	for _, n := range a.viewTokens {
		peak = max(peak, n)
	}
	// What IS guaranteed: nudging stops rather than replaying a multi-thousand
	// token prompt into a context that is already too big.
	if a.nudgesSeen > 25 {
		t.Fatalf("%d nudges were sent to a model that never acted; suppression is not working", a.nudgesSeen)
	}
	// What is NOT guaranteed is that the context stays under the window — see
	// TestTruncateCannotHelpWithManyMediumMessages. The designed outcome there
	// is a clean ReasonPromptTooLong via Reactive, not a silent overflow.
	t.Logf("uncooperative: nudges=%d peak=%d window=%d (overflow is expected here; "+
		"the valve only bites on individually large messages)", a.nudgesSeen, peak, window)
}

// What the mechanical valve actually guarantees: when tool results are large
// enough to cut, it removes most of their weight from every request.
//
// It is a stopgap, not a substitute for compression. Truncation leaves a floor
// of roughly one thousand tokens per message (the head and tail it keeps), so
// enough messages will still exceed any window — see
// TestTruncateFloorAccumulatesWithoutCompression. What it buys is time, and a
// request that stays valid while the model has the chance to act.
func TestTruncateRemovesMostOfLargeOutputs(t *testing.T) {
	window := 40000
	sess := simSession(t, window)
	a := newSimAgent(t, sess, 1000000) // never compresses

	for range 8 {
		a.work(40000) // each tool result is far above the truncation floor
		a.observe()
	}
	cores, _ := Project(a.history)
	raw := estimateCoreTokens(cores)
	sent := a.viewTokens[len(a.viewTokens)-1]

	if sent >= raw/2 {
		t.Fatalf("the request carries %d of %d raw tokens; truncation removed almost nothing", sent, raw)
	}
	if sess.lastTruncatedCount == 0 {
		t.Fatal("no tool result was truncated despite every one being far above the floor")
	}
	t.Logf("raw=%d sent=%d (%.0f%% removed) across %d truncations",
		raw, sent, (1-float64(sent)/float64(raw))*100, sess.lastTruncatedCount)
}

// A second documented limitation, pinned so it is known rather than discovered.
//
// Truncation keeps a head and a tail of each message, which floors it at roughly
// a thousand tokens. N truncated messages therefore cannot go below about N
// thousand tokens, and a long enough uncooperative session exceeds any window no
// matter how aggressively the valve fires.
//
// This is why the valve is explicitly a last resort rather than a strategy: only
// compression removes messages from the request entirely.
func TestTruncateFloorAccumulatesWithoutCompression(t *testing.T) {
	window := 40000
	sess := simSession(t, window)
	a := newSimAgent(t, sess, 1000000)

	for range 60 {
		a.work(40000)
		a.observe()
	}
	last := a.viewTokens[len(a.viewTokens)-1]
	if last <= window {
		t.Skip("the fixture stayed under the window; the floor was not reached")
	}
	// Even overflowing, truncation must still have done its work — the failure
	// mode is an accumulated floor, not an inactive valve.
	if sess.lastTruncatedCount == 0 {
		t.Fatal("the valve stopped firing entirely; that is a different bug")
	}
	c := &Compactor{sess: sess}
	if _, ok := c.Reactive(t.Context(), a.history); ok {
		t.Fatal("Reactive claimed it shrank a view it cannot shrink; that produces a 413 retry loop")
	}
	t.Logf("accumulated floor: %d tokens over a %d window after 60 uncompressed turns "+
		"(%d truncations); Reactive correctly refused", last, window, sess.lastTruncatedCount)
}

// A documented limitation, pinned so it is a known property rather than a
// surprise.
//
// emergency-truncate cuts individual tool results. It cannot help when the
// context is made of many MEDIUM messages: each is below the minimum worth
// cutting, and cutting one to head+tail would make it longer than it started.
// The constants (1000-token floor, 2000+2000 kept) are sized for a large window,
// where anything worth truncating is much bigger than this.
//
// The designed outcome in that case is not silent overflow: Reactive rebuilds
// the view, sees it did not shrink, and returns false so the run ends with a
// clean ReasonPromptTooLong instead of a 413 retry loop.
func TestTruncateCannotHelpWithManyMediumMessages(t *testing.T) {
	window := 40000
	sess := simSession(t, window)
	a := newSimAgent(t, sess, 1000000)

	for range 40 {
		a.work(4000) // just under both truncation criteria
		a.observe()
	}
	peak := 0
	for _, n := range a.viewTokens {
		peak = max(peak, n)
	}
	if peak <= window {
		t.Skip("the fixture did not reach the window; the limitation is not exercised")
	}
	// Reactive is what turns this into a clean failure rather than a retry loop.
	c := &Compactor{sess: sess}
	if _, ok := c.Reactive(t.Context(), a.history); ok {
		t.Fatal("Reactive claimed it shrank a view it cannot shrink; that produces a 413 retry loop")
	}
	t.Logf("documented limitation: peak=%d window=%d, Reactive correctly refused", peak, window)
}

// Whatever happens, the request must stay well-formed.
func TestLongSessionViewStaysValid(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 2)

	for range 50 {
		a.work(4000)
		view := a.observe()
		assertValidRequest(t, view, a.turn)
	}
}

// assertValidRequest checks the invariants a provider would reject.
func assertValidRequest(t *testing.T, view []llm.Message, turn int) {
	t.Helper()
	useIDs := map[string]bool{}
	resIDs := map[string]bool{}
	for _, m := range view {
		if len(m.Content) == 0 {
			t.Fatalf("turn %d: an empty message reached the request", turn)
		}
		for _, b := range m.Content {
			switch b.Type {
			case llm.BlockToolUse:
				useIDs[b.ID] = true
			case llm.BlockToolResult:
				resIDs[b.ToolUseID] = true
			case llm.BlockThinking:
				if b.Signature == "" {
					t.Fatalf("turn %d: a thinking block lost its signature; Anthropic rejects that on replay", turn)
				}
			}
		}
		if m.Role == llm.RoleAssistant && m.Text() == "" && len(m.ToolUses()) == 0 {
			hasThinking := false
			for _, b := range m.Content {
				if b.Type == llm.BlockThinking {
					hasThinking = true
				}
			}
			if !hasThinking {
				t.Fatalf("turn %d: an empty assistant message reached the request", turn)
			}
		}
	}
	for id := range resIDs {
		if !useIDs[id] {
			t.Fatalf("turn %d: tool_result %s has no matching tool_use", turn, id)
		}
	}
	for id := range useIDs {
		if !resIDs[id] {
			t.Fatalf("turn %d: tool_use %s has no matching tool_result", turn, id)
		}
	}
}

// The task definition must survive any amount of compression.
func TestLongSessionPreservesFirstUserMessage(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1)
	for range 60 {
		a.work(4000)
		a.observe()
	}
	view := sess.View(a.history)
	found := false
	for _, m := range view {
		if strings.Contains(m.Text(), "Refactor the authentication subsystem") {
			found = true
		}
	}
	if !found {
		t.Fatal("the first user message was compressed away; the model no longer knows what it is doing")
	}
}

// Refs are handed out once. Nothing in a long session may renumber them.
func TestLongSessionRefsNeverChange(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1)

	seen := map[string]string{}
	for range 50 {
		a.work(4000)
		a.observe()
		for raw, ref := range sess.State().MessageRefs.ByRaw {
			if prev, ok := seen[raw]; ok && prev != ref {
				t.Fatalf("message %s changed ref from %s to %s at turn %d", raw, prev, ref, a.turn)
			}
			seen[raw] = ref
		}
	}
	if len(seen) < 20 {
		t.Fatalf("only %d refs were assigned across 50 turns; the fixture is not exercising much", len(seen))
	}
}

func head(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
