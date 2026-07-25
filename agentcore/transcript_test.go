package agentcore

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
	"github.com/Autumn-27/norma/transcript"
)

// textServer replies with a single end_turn text message carrying usage.
func textServer(t *testing.T, text string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(sse(
			`data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`, ``,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}`, ``,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"`+text+`"}}`, ``,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`, ``,
			`data: {"type":"message_stop"}`, ``,
		)))
	}))
}

func drainPrompt(t *testing.T, s *Session, input string) {
	t.Helper()
	for _, err := range s.Prompt(context.Background(), input) {
		if err != nil {
			t.Fatalf("prompt: %v", err)
		}
	}
}

// TestTranscriptPersistAndResume drives a Session with a transcript store, then
// resumes a fresh Session from disk and continues the conversation.
func TestTranscriptPersistAndResume(t *testing.T) {
	srv := textServer(t, "first answer")
	defer srv.Close()
	prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	store := transcript.NewStore(t.TempDir())

	s := NewSession(Options{
		Provider:       prov,
		SystemPrompt:   []string{"test"},
		Tools:          []tool.CoreTool{},
		PermissionMode: permission.ModeBypass,
		Transcript:     store,
	})
	sid := s.SessionID()
	if sid == "" {
		t.Fatal("expected a generated session id")
	}
	drainPrompt(t, s, "first question")

	// On-disk records: user + assistant, assistant carries usage.
	recs, err := store.Load(store.MainPath(sid))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("records=%d, want 2", len(recs))
	}
	if recs[0].Message.Role != llm.RoleUser || recs[1].Message.Role != llm.RoleAssistant {
		t.Fatalf("roles=%v,%v", recs[0].Message.Role, recs[1].Message.Role)
	}
	if recs[1].ParentUUID != recs[0].UUID {
		t.Fatal("assistant record not chained to user record")
	}
	if recs[1].Usage == nil || recs[1].Usage.OutputTokens != 5 {
		t.Fatalf("assistant usage = %+v, want output=5", recs[1].Usage)
	}

	// Resume into a fresh Session and verify history is rebuilt.
	srv2 := textServer(t, "second answer")
	defer srv2.Close()
	prov2, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, BaseURL: srv2.URL, APIKey: "k", Model: "m"})
	resumed := NewSession(Options{
		Provider:       prov2,
		SystemPrompt:   []string{"test"},
		Tools:          []tool.CoreTool{},
		PermissionMode: permission.ModeBypass,
		Transcript:     store,
	})
	if err := resumed.Resume(sid); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(resumed.Messages()) != 2 {
		t.Fatalf("resumed messages=%d, want 2", len(resumed.Messages()))
	}
	if got := resumed.Messages()[1].Text(); got != "first answer" {
		t.Fatalf("resumed assistant text=%q", got)
	}

	// Continue the resumed conversation; new records append to the same file.
	drainPrompt(t, resumed, "second question")
	recs2, _ := store.Load(store.MainPath(sid))
	if len(recs2) != 4 {
		t.Fatalf("records after continue=%d, want 4", len(recs2))
	}
	if recs2[2].ParentUUID != recs2[1].UUID {
		t.Fatal("continued chain broken across resume")
	}
}

// TestRecorderFiresForSubagentSidechain checks the harness Recorder is invoked
// for each new message and that boundary records are skipped on reconstruction.
func TestTranscriptMessagesSkipsBoundary(t *testing.T) {
	recs := []transcript.Record{
		{Type: "message", Message: ptr(llm.UserText("q")), UUID: "1"},
		{Type: "boundary", Boundary: &llm.BoundaryMeta{Trigger: "auto", PreTokens: 100}, UUID: "2"},
		{Type: "message", Message: ptr(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("a")}}),
			Usage: &llm.Usage{OutputTokens: 7}, UUID: "3"},
	}
	msgs, usage := transcript.Messages(recs)
	if len(msgs) != 2 {
		t.Fatalf("messages=%d, want 2 (boundary skipped)", len(msgs))
	}
	if usage.OutputTokens != 7 {
		t.Fatalf("summed usage=%d, want 7", usage.OutputTokens)
	}
}

var _ harness.Recorder = (*transcript.Writer)(nil)

func ptr(m llm.Message) *llm.Message { return &m }

// TestResumeRestoresCompactedView writes a transcript shaped like one a compacted
// session produces — raw births, then the persisted post-boundary working set
// (boundary marker + summary + re-recorded tail) — and verifies a cold-start
// Resume rebuilds an array whose API view is the COMPACTED one (summary + tail),
// NOT the full raw history. This is the "don't re-summarize on resume" guarantee.
func TestResumeRestoresCompactedView(t *testing.T) {
	store := transcript.NewStore(t.TempDir())
	sid := transcript.NewSessionID()
	w := store.NewWriter(sid, "")

	// Raw births (the full history, retained on disk for audit).
	w.RecordMessage(llm.UserText("q1"), llm.Usage{})
	w.RecordMessage(llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("a1")}}, llm.Usage{})
	w.RecordMessage(llm.UserText("q2"), llm.Usage{}) // becomes the kept tail
	// Compaction fires: persist the post-boundary working set (mirrors recordNewBoundary).
	w.RecordBoundary(llm.BoundaryMeta{Trigger: "auto", PreTokens: 5000}) // observability (skipped on replay)
	w.RecordMessage(llm.BoundaryMessage(llm.BoundaryMeta{Trigger: "auto", PreTokens: 5000}), llm.Usage{})
	w.RecordMessage(llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.TextBlock("[summary] DIGEST")}}, llm.Usage{})
	w.RecordMessage(llm.UserText("q2"), llm.Usage{}) // tail re-recorded after the boundary
	if err := w.Err(); err != nil {
		t.Fatalf("write: %v", err)
	}

	srv := textServer(t, "next")
	defer srv.Close()
	prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	s := NewSession(Options{
		Provider:       prov,
		SystemPrompt:   []string{"test"},
		Tools:          []tool.CoreTool{},
		PermissionMode: permission.ModeBypass,
		Transcript:     store,
	})
	if err := s.Resume(sid); err != nil {
		t.Fatalf("resume: %v", err)
	}

	// The boundary marker survived reconstruction (persisted as a message, not skipped).
	full := s.Messages()
	if llm.LastBoundaryIndex(full) < 0 {
		t.Fatal("resumed array lost the compaction boundary")
	}
	// The API view is the compacted one: [summary, tail] — not the raw q1/a1/q2.
	api := llm.MessagesForAPI(full)
	if len(api) != 2 {
		t.Fatalf("compacted API view = %d msgs, want 2 (summary + tail); resume re-exposed raw history", len(api))
	}
	if !strings.Contains(api[0].Text(), "DIGEST") || api[1].Text() != "q2" {
		t.Fatalf("API view = %q,%q; want summary then tail", api[0].Text(), api[1].Text())
	}
}
