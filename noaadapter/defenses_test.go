package noaadapter

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/tool"
)

func deadRanges(refs ...[2]string) []noa.CompressRange {
	out := make([]noa.CompressRange, 0, len(refs))
	for _, r := range refs {
		out = append(out, noa.CompressRange{StartRef: r[0], EndRef: r[1], Summary: longSummary})
	}
	return out
}

// The refusal only fires on the SECOND sighting: the first failure carries a
// corrective message the model deserves a chance to act on.
func TestDeadRangeRefusesOnRepeat(t *testing.T) {
	d := NewDeadRangeTracker()
	st := noa.CreateInitialState("s", "/a")
	rs := deadRanges([2]string{"m09000", "m09001"})

	if msg := d.Check(rs, st); msg != "" {
		t.Fatalf("refused on first sight: %q", msg)
	}
	d.Record(rs)
	if msg := d.Check(rs, st); msg != "" {
		t.Fatalf("refused after one failure: %q", msg)
	}
	d.Record(rs)
	msg := d.Check(rs, st)
	if msg == "" {
		t.Fatalf("not refused after %d failures", noa.DeadRepeatReject)
	}
	if !strings.Contains(msg, "Do not resubmit") {
		t.Fatalf("refusal = %q, want an explicit instruction to stop", msg)
	}
}

// A dead end the model cannot act on is just another wasted turn; naming the
// live blocks turns it into a next step.
func TestDeadRangeNamesLiveBlocks(t *testing.T) {
	d := NewDeadRangeTracker()
	st := noa.CreateInitialState("s", "/a")
	st.Blocks = []noa.CompressionBlock{
		{BlockID: "b3", Tier: 2, StartRef: "m00012", EndRef: "m00160", Active: true},
	}
	rs := deadRanges([2]string{"m09000", "m09001"})
	d.Record(rs)
	d.Record(rs)

	msg := d.Check(rs, st)
	if !strings.Contains(msg, "b3(T2)=m00012–m00160") {
		t.Fatalf("refusal = %q, want it to name the live block", msg)
	}
}

// A retry that reshuffles the ranges is still the same request.
func TestDeadRangeSignatureIsOrderIndependent(t *testing.T) {
	d := NewDeadRangeTracker()
	st := noa.CreateInitialState("s", "/a")
	a := deadRanges([2]string{"m1", "m2"}, [2]string{"m3", "m4"})
	b := deadRanges([2]string{"m3", "m4"}, [2]string{"m1", "m2"})

	d.Record(a)
	d.Record(b)
	if d.Check(a, st) == "" {
		t.Fatal("a reordered resubmission was treated as a different request")
	}
}

// A successful compression changes the view, so what was dead may not be.
func TestDeadRangeResetAfterSuccess(t *testing.T) {
	d := NewDeadRangeTracker()
	st := noa.CreateInitialState("s", "/a")
	rs := deadRanges([2]string{"m1", "m2"})
	d.Record(rs)
	d.Record(rs)
	if d.Check(rs, st) == "" {
		t.Fatal("expected a refusal before the reset")
	}
	d.Reset()
	if msg := d.Check(rs, st); msg != "" {
		t.Fatalf("still refusing after a successful compression: %q", msg)
	}
}

// End to end: the third identical call is refused without reaching the engine.
func TestDeadRangeBreaksLoopEndToEnd(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)

	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m09000", "endId": "m09001", "summary": longSummary}},
	})
	var last string
	for range 3 {
		res, _ := runCompress(sess, args, &tool.ToolContext{ToolUseID: "x"})
		last = res.Content[0].Text
	}
	if !strings.Contains(last, "Do not resubmit") {
		t.Fatalf("the third identical call was not refused:\n%s", last)
	}
}

// The provider's figure was measured before the compression, so it describes a
// context that no longer exists. Trusting it would pin the pressure ladder at
// the pre-compression size and re-raise an emergency the model just resolved.
func TestFloorStaleIgnoresPreCompressionAnchor(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)

	sess.mu.Lock()
	sess.providerTokens = 999999
	sess.providerTokensAt = 0
	sess.lastCompressAt = 1 // a compression happened after that figure was taken
	cores, _ := Project(msgs)
	got := sess.resolveTokenCount(cores)
	est := estimateCoreTokens(cores)
	sess.mu.Unlock()

	if got != est {
		t.Fatalf("token count = %d, want the local estimate %d — the provider anchor predates the compression", got, est)
	}
}

func TestProviderTokensUsedWhenCurrent(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)

	sess.mu.Lock()
	cores, _ := Project(msgs)
	est := estimateCoreTokens(cores)
	sess.providerTokens = est + 100 // within the drift tolerance
	sess.providerTokensAt = 0
	sess.lastCompressAt = 0
	got := sess.resolveTokenCount(cores)
	sess.mu.Unlock()

	if got != est+100 {
		t.Fatalf("token count = %d, want the provider figure %d", got, est+100)
	}
}

// A figure far from the estimate describes a different array than the one being
// built, so the estimate of the SENT view is the honest number.
func TestProviderTokensRejectedOnLargeDrift(t *testing.T) {
	sess, msgs := pressureSession(t)
	sess.View(msgs)

	sess.mu.Lock()
	cores, _ := Project(msgs)
	est := estimateCoreTokens(cores)
	sess.providerTokens = est * 10
	sess.providerTokensAt = 0
	sess.lastCompressAt = 0
	got := sess.resolveTokenCount(cores)
	sess.mu.Unlock()

	if got != est {
		t.Fatalf("token count = %d, want the local estimate %d on large drift", got, est)
	}
}

func TestDeriveChildStateKeepsLedgerResetsPacing(t *testing.T) {
	parent := sampleState()
	child := DeriveChildState(parent, "child", "/archive/child")

	if len(child.Blocks) != len(parent.Blocks) {
		t.Fatalf("child has %d blocks, want the parent's %d", len(child.Blocks), len(parent.Blocks))
	}
	if child.NextBlockID != parent.NextBlockID {
		t.Fatalf("NextBlockID = %d, want it inherited — otherwise the child allocates ids that collide",
			child.NextBlockID)
	}
	if len(child.MessageRefs.ByRaw) != len(parent.MessageRefs.ByRaw) {
		t.Fatal("the ref map was not inherited; the child could not address inherited context")
	}
	if child.Nudge.LastNudgeShownTokens != 0 || len(child.Nudge.LastShownByTier) != 0 {
		t.Fatalf("nudge pacing = %+v, want it reset for the child's own conversation", child.Nudge)
	}
	if child.Stats != (noa.Stats{}) {
		t.Fatalf("stats = %+v, want them reset so the child reports its own work", child.Stats)
	}
	// The inherited archive must stay findable from the child.
	if child.Blocks[0].ArchivePath != parent.Blocks[0].ArchivePath {
		t.Fatalf("archive path = %q, want the parent's absolute path kept", child.Blocks[0].ArchivePath)
	}
	if child.ArchiveRoot != "/archive/child" {
		t.Fatalf("ArchiveRoot = %q, want the child's own directory for NEW archives", child.ArchiveRoot)
	}
}

// Deep-copied, so a child mutating its ledger cannot corrupt the parent's.
func TestDeriveChildStateIsDeepCopy(t *testing.T) {
	parent := sampleState()
	child := DeriveChildState(parent, "child", "/archive/child")
	child.Blocks[0].Active = false
	child.Blocks[0].EffectiveMessageIDs[0] = "mutated"
	child.MessageRefs.ByRaw["h_abc"] = "m99999"

	if !parent.Blocks[0].Active {
		t.Error("the parent's block was deactivated by the child")
	}
	if parent.Blocks[0].EffectiveMessageIDs[0] == "mutated" {
		t.Error("the coverage slice is shared with the child")
	}
	if parent.MessageRefs.ByRaw["h_abc"] != "m00001" {
		t.Error("the ref map is shared with the child")
	}
}

func TestInheritStateWalksParentChain(t *testing.T) {
	base := t.TempDir()
	// The grandparent has the ledger; the parent is empty.
	gpRoot := filepath.Join(base, "grandparent")
	if err := NewStateStore(gpRoot).Save(sampleState()); err != nil {
		t.Fatalf("seed grandparent: %v", err)
	}
	pRoot := filepath.Join(base, "parent")
	if err := NewStateStore(pRoot).Save(noa.CreateInitialState("parent", pRoot)); err != nil {
		t.Fatalf("seed parent: %v", err)
	}

	got, from, ok := InheritState(InheritOptions{
		ArchiveBaseDir: base, SessionID: "child",
		ParentIDs: []string{"parent", "grandparent"},
	})
	if !ok {
		t.Fatal("nothing inherited despite a grandparent with blocks")
	}
	if from != "grandparent" {
		t.Fatalf("inherited from %q, want the nearest ancestor that has blocks", from)
	}
	if len(got.Blocks) != 1 {
		t.Fatalf("inherited %d blocks, want 1", len(got.Blocks))
	}
	if got.SessionID != "child" {
		t.Fatalf("SessionID = %q, want the child's own", got.SessionID)
	}
}

func TestInheritStateStopsAtDepthLimit(t *testing.T) {
	base := t.TempDir()
	// Only the tenth ancestor has anything, past the walk limit.
	deep := filepath.Join(base, "a9")
	if err := NewStateStore(deep).Save(sampleState()); err != nil {
		t.Fatalf("seed: %v", err)
	}
	var parents []string
	for i := range 9 {
		parents = append(parents, "a"+string(rune('0'+i)))
	}
	parents = append(parents, "a9")

	_, _, ok := InheritState(InheritOptions{
		ArchiveBaseDir: base, SessionID: "child", ParentIDs: parents,
	})
	if ok {
		t.Fatalf("the walk exceeded the %d-ancestor limit", maxInheritDepth)
	}
}

func TestInheritStateNoParents(t *testing.T) {
	got, from, ok := InheritState(InheritOptions{ArchiveBaseDir: t.TempDir(), SessionID: "solo"})
	if ok || from != "" {
		t.Fatalf("inherited %q from nowhere", from)
	}
	if len(got.Blocks) != 0 || got.SessionID != "solo" {
		t.Fatalf("fresh state = %+v", got)
	}
}

// A session that would otherwise start from zero picks up the ancestor's ledger
// rather than re-compressing everything it already did.
func TestSessionInheritsFromParent(t *testing.T) {
	base := t.TempDir()
	parentRoot := filepath.Join(base, "parent")
	if err := NewStateStore(parentRoot).Save(sampleState()); err != nil {
		t.Fatalf("seed parent: %v", err)
	}
	var warns []string
	sess, err := newSession(Options{
		ArchiveBaseDir: base, SessionID: "child", ParentSessionIDs: []string{"parent"},
		OnWarn: func(m string) { warns = append(warns, m) },
	})
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	if len(sess.State().Blocks) != 1 {
		t.Fatalf("child has %d blocks, want the parent's 1", len(sess.State().Blocks))
	}
	if !strings.Contains(strings.Join(warns, "\n"), "inherited") {
		t.Fatalf("the host was not told about the inheritance: %v", warns)
	}
}

// A session with its own ledger must not have it replaced.
func TestSessionDoesNotInheritOverOwnState(t *testing.T) {
	base := t.TempDir()
	if err := NewStateStore(filepath.Join(base, "parent")).Save(sampleState()); err != nil {
		t.Fatalf("seed parent: %v", err)
	}
	own := sampleState()
	own.SessionID = "child"
	own.Blocks = append(own.Blocks, noa.CompressionBlock{BlockID: "b9", Tier: 1, Active: true})
	if err := NewStateStore(filepath.Join(base, "child")).Save(own); err != nil {
		t.Fatalf("seed child: %v", err)
	}
	sess, err := newSession(Options{
		ArchiveBaseDir: base, SessionID: "child", ParentSessionIDs: []string{"parent"},
	})
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	if len(sess.State().Blocks) != 2 {
		t.Fatalf("child has %d blocks, want its own 2 kept", len(sess.State().Blocks))
	}
}

func TestSubagentArchiveRoot(t *testing.T) {
	got := SubagentArchiveRoot("/base", "sess", "agent7")
	want := filepath.Join("/base", "sess", "subagents", "agent7")
	if got != want {
		t.Fatalf("SubagentArchiveRoot = %q, want %q", got, want)
	}
}
