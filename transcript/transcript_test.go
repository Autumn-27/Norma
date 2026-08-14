package transcript

import (
	"context"
	"os"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

func TestRecordAndLoadRoundTrip(t *testing.T) {
	s := NewStore(t.TempDir())
	w := s.NewWriter("sess", "")
	w.RecordMessage(llm.UserText("hi"), llm.Usage{})
	w.RecordMessage(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("yo")}}, llm.Usage{InputTokens: 10, OutputTokens: 5})
	if err := w.Err(); err != nil {
		t.Fatalf("write err: %v", err)
	}
	recs, err := s.Load(s.MainPath("sess"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("want 2 records, got %d", len(recs))
	}
	msgs, total := Messages(recs)
	if len(msgs) != 2 || msgs[0].Text() != "hi" || msgs[1].Text() != "yo" {
		t.Fatalf("reconstructed messages wrong: %+v", msgs)
	}
	if total != (llm.Usage{InputTokens: 10, OutputTokens: 5}) {
		t.Fatalf("usage accumulation wrong: %+v", total)
	}
}

// A boundary record is persisted for observability but skipped when
// reconstructing the conversation (compaction rebuilds the working set itself).
func TestBoundaryRecordSkippedInReconstruction(t *testing.T) {
	s := NewStore(t.TempDir())
	w := s.NewWriter("sess", "")
	w.RecordMessage(llm.UserText("before"), llm.Usage{})
	w.RecordBoundary(llm.BoundaryMeta{Trigger: "auto", PreTokens: 1234, MessagesSummarized: 3})
	w.RecordMessage(llm.UserText("after"), llm.Usage{})

	recs, _ := s.Load(s.MainPath("sess"))
	if len(recs) != 3 {
		t.Fatalf("want 3 records on disk (incl. boundary), got %d", len(recs))
	}
	// The boundary record is present and typed.
	var sawBoundary bool
	for _, r := range recs {
		if r.Type == "boundary" {
			sawBoundary = true
			if r.Boundary == nil || r.Boundary.PreTokens != 1234 {
				t.Fatalf("boundary meta not persisted: %+v", r.Boundary)
			}
		}
	}
	if !sawBoundary {
		t.Fatal("boundary record not written")
	}
	// But reconstruction skips it.
	msgs, _ := Messages(recs)
	if len(msgs) != 2 || msgs[0].Text() != "before" || msgs[1].Text() != "after" {
		t.Fatalf("boundary should be skipped in Messages: %+v", msgs)
	}
}

func TestParentChainAndLastUUID(t *testing.T) {
	s := NewStore(t.TempDir())
	w := s.NewWriter("sess", "")
	w.RecordMessage(llm.UserText("a"), llm.Usage{})
	w.RecordMessage(llm.UserText("b"), llm.Usage{})
	recs, _ := s.Load(s.MainPath("sess"))
	if recs[0].ParentUUID != "" {
		t.Fatalf("first record should have no parent, got %q", recs[0].ParentUUID)
	}
	if recs[1].ParentUUID != recs[0].UUID {
		t.Fatalf("chain broken: rec1.parent=%q, rec0.uuid=%q", recs[1].ParentUUID, recs[0].UUID)
	}
	if LastUUID(recs) != recs[1].UUID {
		t.Fatalf("LastUUID mismatch")
	}
	if LastUUID(nil) != "" {
		t.Fatal("LastUUID(nil) should be empty")
	}
}

func TestResumeWriterContinuesChain(t *testing.T) {
	s := NewStore(t.TempDir())
	w := s.NewWriter("sess", "")
	w.RecordMessage(llm.UserText("first"), llm.Usage{})
	recs, _ := s.Load(s.MainPath("sess"))
	last := LastUUID(recs)

	rw := s.ResumeWriter("sess", last)
	rw.RecordMessage(llm.UserText("second"), llm.Usage{})

	recs2, _ := s.Load(s.MainPath("sess"))
	if len(recs2) != 2 {
		t.Fatalf("want 2 records after resume, got %d", len(recs2))
	}
	if recs2[1].ParentUUID != last {
		t.Fatalf("resumed record should chain to %q, got %q", last, recs2[1].ParentUUID)
	}
}

func TestSubagentSidechainPath(t *testing.T) {
	s := NewStore(t.TempDir())
	w := s.NewWriter("sess", "agent1")
	w.RecordMessage(llm.UserText("sub"), llm.Usage{})
	// Written to the sidechain path, not the main file.
	if recs, _ := s.Load(s.AgentPath("sess", "agent1")); len(recs) != 1 || !recs[0].IsSidechain {
		t.Fatalf("subagent record not on sidechain: %+v", recs)
	}
	if recs, _ := s.Load(s.MainPath("sess")); len(recs) != 0 {
		t.Fatalf("main file should be empty, got %d", len(recs))
	}
}

func TestLoadMissingFileIsFreshSession(t *testing.T) {
	s := NewStore(t.TempDir())
	recs, err := s.Load(s.MainPath("does-not-exist"))
	if err != nil || recs != nil {
		t.Fatalf("missing file should yield (nil, nil), got (%v, %v)", recs, err)
	}
}

func TestMessagesFiltersNilAndBoundary(t *testing.T) {
	recs := []Record{
		{Type: "message", Message: &llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextBlock("x")}}, Usage: &llm.Usage{InputTokens: 3}},
		{Type: "boundary", Boundary: &llm.BoundaryMeta{}},
		{Type: "message", Message: nil}, // defensive: nil message skipped
	}
	msgs, total := Messages(recs)
	if len(msgs) != 1 || total.InputTokens != 3 {
		t.Fatalf("Messages should keep only the real message: msgs=%d usage=%+v", len(msgs), total)
	}
}

func TestSessionIDContextRoundTrip(t *testing.T) {
	ctx := WithSessionID(context.Background(), "abc123")
	if got := SessionIDFrom(ctx); got != "abc123" {
		t.Fatalf("SessionIDFrom=%q, want abc123", got)
	}
	if got := SessionIDFrom(context.Background()); got != "" {
		t.Fatalf("empty context should yield \"\", got %q", got)
	}
}

func TestNewIDsAreDistinctAndNonEmpty(t *testing.T) {
	a, b := NewSessionID(), NewSessionID()
	if a == "" || b == "" || a == b {
		t.Fatalf("session ids should be non-empty and distinct: %q %q", a, b)
	}
	if id := NewAgentID(); id == "" || len(id) != 32 { // 16 random bytes hex-encoded
		t.Fatalf("agent id malformed: %q", id)
	}
}

func TestLoadCorruptLineErrors(t *testing.T) {
	s := NewStore(t.TempDir())
	path := s.MainPath("sess")
	// Write a valid record, then a corrupt (non-JSON) line.
	w := s.NewWriter("sess", "")
	w.RecordMessage(llm.UserText("ok"), llm.Usage{})
	if err := appendRaw(path, "{not valid json\n"); err != nil {
		t.Fatalf("setup append: %v", err)
	}
	if _, err := s.Load(path); err == nil {
		t.Fatal("Load should error on a corrupt line")
	}
}

// appendRaw appends a raw line to a transcript file for corruption testing.
func appendRaw(path, line string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line)
	return err
}
