package noaadapter

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/noa"
)

func sampleState() noa.CompressionState {
	st := noa.CreateInitialState("sess", "/archive/sess")
	st.NextBlockID = 3
	st.MessageRefs.ByRaw = map[string]string{"h_abc": "m00001", "h_def": "m00002"}
	st.MessageRefs.ByRef = map[string]string{"m00001": "h_abc", "m00002": "h_def"}
	st.TokenSnapshot = map[string]int{"m00001": 120, "m00002": 4100}
	st.Nudge.LastPerMessageNudgeTokens = 118000
	st.Nudge.LastNudgeShownTokens = 116000
	st.Nudge.LastShownByTier = map[noa.Tier]int{2: 110000}
	st.Stats = noa.Stats{TokensCompressed: 48200, CompressionCount: 7}
	st.Blocks = []noa.CompressionBlock{{
		BlockID: "b1", Tier: 1, Topic: "auth", Summary: "decided on JWT",
		DirectMessageIDs: []string{"h_abc"}, EffectiveMessageIDs: []string{"h_abc", "h_def"},
		DirectBlockIDs: nil, ArchivePath: "/archive/sess/tier1/b1.md", ArchiveRel: "tier1/b1.md",
		CompressedTokens: 12480, StartRef: "m00001", EndRef: "m00002",
		CreatedAt: 1789000000, Active: true, CompressCallID: "toolu_1",
	}}
	return st
}

func TestStateStoreRoundTrip(t *testing.T) {
	root := t.TempDir()
	s := NewStateStore(root)
	want := sampleState()
	if err := s.Save(want); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// A fresh store reads from disk rather than the cache.
	got, err := NewStateStore(root).Load("sess", "/archive/sess")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.NextBlockID != want.NextBlockID || got.SessionID != want.SessionID {
		t.Fatalf("identity lost: %+v", got)
	}
	if len(got.Blocks) != 1 {
		t.Fatalf("got %d blocks, want 1", len(got.Blocks))
	}
	b := got.Blocks[0]
	w := want.Blocks[0]
	if b.BlockID != w.BlockID || b.Tier != w.Tier || b.Summary != w.Summary ||
		b.ArchivePath != w.ArchivePath || b.CompressCallID != w.CompressCallID {
		t.Fatalf("block round trip lost fields:\ngot  %+v\nwant %+v", b, w)
	}
	if got.TokenSnapshot["m00002"] != 4100 {
		t.Fatalf("token snapshot = %v, want the frozen numbers preserved", got.TokenSnapshot)
	}
	if got.Nudge.LastShownByTier[2] != 110000 {
		t.Fatalf("per-tier stamps = %v", got.Nudge.LastShownByTier)
	}
	if got.Stats != want.Stats {
		t.Fatalf("stats = %+v, want %+v", got.Stats, want.Stats)
	}
}

// The cache must be written even when there is nowhere to persist to. A store
// that skipped it would hand back a zero state next time, and the model would
// re-compress the same range every single turn.
func TestStateStoreCachesWithoutAFile(t *testing.T) {
	s := NewStateStore("")
	st := sampleState()
	if err := s.Save(st); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := s.Load("sess", "/archive/sess")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got.Blocks) != 1 {
		t.Fatalf("got %d blocks from the in-memory store, want 1", len(got.Blocks))
	}
}

func TestStateStoreMissingFileIsFresh(t *testing.T) {
	got, err := NewStateStore(t.TempDir()).Load("sess", "/a")
	if err != nil {
		t.Fatalf("Load of a missing file = %v, want a clean fresh state", err)
	}
	if len(got.Blocks) != 0 || got.NextBlockID != 1 {
		t.Fatalf("fresh state = %+v", got)
	}
}

// A damaged state file is recoverable by replay, so it must surface as an error
// rather than taking the session down.
func TestStateStoreCorruptFileReportsError(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "state.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := NewStateStore(root).Load("sess", "/a")
	if err == nil {
		t.Fatal("a corrupt state file was accepted")
	}
	if len(got.Blocks) != 0 {
		t.Fatal("a fresh state must be returned alongside the error")
	}
}

func TestStateStoreRejectsUnknownVersion(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "state.json"), []byte(`{"version":999}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := NewStateStore(root).Load("sess", "/a")
	if err == nil || !strings.Contains(err.Error(), "version") {
		t.Fatalf("err = %v, want a version mismatch", err)
	}
}

// Temp + rename: a crash mid-write must not leave a truncated file where a
// valid one was.
func TestStateStoreWriteIsAtomic(t *testing.T) {
	root := t.TempDir()
	s := NewStateStore(root)
	if err := s.Save(sampleState()); err != nil {
		t.Fatalf("Save: %v", err)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".noa-state-") {
			t.Fatalf("a temp file survived: %s", e.Name())
		}
	}
	if !s.Exists() {
		t.Fatal("Exists reports no state file after a successful save")
	}
}

// A session that starts up damaged must still run — the state is rebuildable.
func TestSessionSurvivesCorruptState(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "sess")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "state.json"), []byte("{broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	var warns []string
	sess, err := newSession(Options{
		ArchiveBaseDir: root, SessionID: "sess",
		OnWarn: func(m string) { warns = append(warns, m) },
	})
	if err != nil {
		t.Fatalf("newSession refused to start over a damaged state file: %v", err)
	}
	if len(sess.State().Blocks) != 0 {
		t.Fatal("a damaged state must yield an empty one")
	}
	if len(warns) == 0 {
		t.Fatal("the host was not told the state file was unreadable")
	}
}

func TestSessionArchiveRootIncludesSessionID(t *testing.T) {
	base := t.TempDir()
	sess, err := newSession(Options{ArchiveBaseDir: base, SessionID: "abc123"})
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	if sess.ArchiveRoot() != filepath.Join(base, "abc123") {
		t.Fatalf("archive root = %q, want the base joined with the session id", sess.ArchiveRoot())
	}
}

func TestSessionReportsConfigWarnings(t *testing.T) {
	cfg := noa.DefaultConfig(200000)
	cfg.Tiers.MaxTier = 7 // outside the range the tier prompts cover
	var warns []string
	if _, err := newSession(Options{
		ArchiveBaseDir: t.TempDir(), SessionID: "s", Config: &cfg,
		OnWarn: func(m string) { warns = append(warns, m) },
	}); err != nil {
		t.Fatalf("newSession: %v", err)
	}
	joined := strings.Join(warns, "\n")
	if !strings.Contains(joined, "MaxTier") {
		t.Fatalf("warnings = %v, want one naming MaxTier", warns)
	}
}
