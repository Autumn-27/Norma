package compaction

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// captureProvider records each request and yields a scripted reply, optionally
// failing the first failN Stream calls with failErr (default: an overflow error).
type captureProvider struct {
	reqs    []llm.CompletionRequest
	reply   string
	failN   int
	failErr error
	calls   int
}

func (p *captureProvider) Stream(_ context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	p.calls++
	p.reqs = append(p.reqs, req)
	call := p.calls
	return func(yield func(llm.StreamEvent, error) bool) {
		if call <= p.failN {
			e := p.failErr
			if e == nil {
				e = errors.New("prompt is too long")
			}
			yield(llm.StreamEvent{}, e)
			return
		}
		if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: p.reply}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func asstText(s string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock(s)}}
}

// --- MicroCompact / AutoCompact are covered in compaction_test.go; these cover
// the three-mechanism spec surface (COMPACTION-SPEC-3.md). ---

func TestReactiveRunsAutoCompactOnly(t *testing.T) {
	c := New(smallWindow(), func(context.Context, []llm.Message) (string, error) { return "DIGEST", nil })
	var msgs []llm.Message
	for range 8 {
		msgs = append(msgs, bigMsg(llm.RoleUser, 4000))
	}
	out, ok := c.Reactive(context.Background(), msgs)
	if !ok {
		t.Fatal("reactive should shrink the API view")
	}
	if llm.LastBoundaryIndex(out) < 0 {
		t.Fatal("reactive must insert a boundary (it runs AutoCompact)")
	}
	// Non-destructive: full history retained + boundary + summary, no snip removal.
	if len(out) != len(msgs)+2 {
		t.Fatalf("len(out)=%d, want %d (history + boundary + summary)", len(out), len(msgs)+2)
	}
}

func TestReactiveNoSummarizerReturnsFalse(t *testing.T) {
	c := New(smallWindow(), nil) // summarize == nil → AutoCompact cannot run
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000), bigMsg(llm.RoleUser, 4000)}
	if _, ok := c.Reactive(context.Background(), msgs); ok {
		t.Fatal("with no summarizer, reactive cannot shrink (no snip fallback)")
	}
}

func TestCountPrefersLastInputTokens(t *testing.T) {
	c := New(Config{ContextWindow: 200000}, nil)
	msgs := []llm.Message{bigMsg(llm.RoleUser, 4000)}
	if got := c.count(msgs, 12345); got != 12345 {
		t.Fatalf("count should prefer lastInputTokens=12345, got %d", got)
	}
	if est := c.count(msgs, 0); est <= 0 {
		t.Fatal("estimation fallback should be > 0")
	}
}

func TestBufferKeyedOnEffectiveWindow(t *testing.T) {
	// Raw window 410k is above the 400k tier, but effectiveWindow 390k is below it.
	c := New(Config{ContextWindow: 410000, MaxOutputTokens: 20000}, nil)
	if c.buffer() != 13000 {
		t.Fatalf("buffer=%d, want 13000 (keyed on effectiveWindow=390000)", c.buffer())
	}
	c2 := New(Config{ContextWindow: 420000, MaxOutputTokens: 20000}, nil) // eff=400000
	if c2.buffer() != 30000 {
		t.Fatalf("buffer=%d, want 30000", c2.buffer())
	}
}

func TestMaxTurnGrowthMinCap(t *testing.T) {
	c := New(Config{ContextWindow: 200000, MaxOutputTokens: 64000}, nil)
	// reserved = min(64000, 20000) = 20000 → growth = 20000 + 15000.
	if c.maxTurnGrowth() != 35000 {
		t.Fatalf("maxTurnGrowth=%d, want 35000", c.maxTurnGrowth())
	}
}

func TestSummaryUsesContinuationPreamble(t *testing.T) {
	c := New(smallWindow(), func(context.Context, []llm.Message) (string, error) { return "DIGEST", nil })
	var msgs []llm.Message
	for range 8 {
		msgs = append(msgs, bigMsg(llm.RoleUser, 4000))
	}
	out, ok := c.autoCompact(context.Background(), msgs, 100, "auto")
	if !ok {
		t.Fatal("autoCompact did not run")
	}
	if api := llm.MessagesForAPI(out); !strings.HasPrefix(api[0].Text(), "This session is being continued") {
		t.Fatalf("summary should carry the continuation preamble, got %q", api[0].Text())
	}
}

func TestFormatCompactSummary(t *testing.T) {
	raw := "<analysis>\nscratch thoughts\n</analysis>\n\n<summary>\nThe outcome.\n</summary>"
	got := formatCompactSummary(raw)
	if strings.Contains(got, "analysis") || strings.Contains(got, "scratch") {
		t.Fatalf("<analysis> not stripped: %q", got)
	}
	if got != "Summary:\nThe outcome." {
		t.Fatalf("got %q, want %q", got, "Summary:\nThe outcome.")
	}
}

func TestGroupByAPIRoundAndTruncate(t *testing.T) {
	msgs := []llm.Message{llm.UserText("u0"), asstText("a1"), llm.UserText("u1"), asstText("a2")}
	if g := groupByAPIRound(msgs); len(g) != 3 {
		t.Fatalf("groups=%d, want 3 ([u0][a1 u1][a2])", len(g))
	}
	out, ok := truncateHeadForPTLRetry(msgs)
	if !ok {
		t.Fatal("should truncate (≥2 groups)")
	}
	if out[0].Text() != ptlRetryMarker {
		t.Fatalf("result starting with an assistant must be marker-prefixed, got %q", out[0].Text())
	}
	if _, ok := truncateHeadForPTLRetry([]llm.Message{llm.UserText("only")}); ok {
		t.Fatal("a single group cannot be truncated")
	}
}

func TestPairForReplayDropsDangling(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockToolUse, ID: "x", Name: "Read"}}}, // no result
		llm.UserText("keep me"),
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResultText("y", "orphan", false)}}, // no use
	}
	out := pairForReplay(msgs)
	if len(out) != 1 || out[0].Text() != "keep me" {
		t.Fatalf("dangling tool blocks should be dropped, got %+v", out)
	}
}

func TestProviderSummarizerReplaysRealMessages(t *testing.T) {
	p := &captureProvider{reply: "<summary>DONE</summary>"}
	s := ProviderSummarizer(p, 8000)
	head := []llm.Message{llm.UserText("hello"), asstText("hi")}
	got, err := s(context.Background(), head)
	if err != nil {
		t.Fatal(err)
	}
	if got != "Summary:\nDONE" {
		t.Fatalf("summary format: %q", got)
	}
	if len(p.reqs) != 1 {
		t.Fatalf("calls=%d, want 1", len(p.reqs))
	}
	req := p.reqs[0]
	if req.Thinking != "disabled" {
		t.Fatalf("thinking=%q, want disabled", req.Thinking)
	}
	if req.MaxTokens != 8000 { // min(20000, 8000)
		t.Fatalf("maxTokens=%d, want 8000", req.MaxTokens)
	}
	if len(req.System) != 1 || req.System[0] != summarySystemPrompt {
		t.Fatalf("system prompt not set: %v", req.System)
	}
	if len(req.Messages) != 3 || req.Messages[0].Text() != "hello" {
		t.Fatalf("real messages not replayed: %+v", req.Messages)
	}
	if !strings.Contains(req.Messages[2].Text(), "CRITICAL: Respond with TEXT ONLY") {
		t.Fatal("summary-request prompt must be the last message")
	}
}

func TestProviderSummarizerCapsAt20k(t *testing.T) {
	p := &captureProvider{reply: "<summary>X</summary>"}
	_, _ = ProviderSummarizer(p, 0)(context.Background(), []llm.Message{llm.UserText("hi")})
	if p.reqs[0].MaxTokens != compactMaxOutputTokens {
		t.Fatalf("maxTokens=%d, want %d", p.reqs[0].MaxTokens, compactMaxOutputTokens)
	}
}

func TestProviderSummarizerPTLRetry(t *testing.T) {
	p := &captureProvider{reply: "<summary>OK</summary>", failN: 1} // first call overflows
	s := ProviderSummarizer(p, 0)
	head := []llm.Message{llm.UserText("u0"), asstText("a1"), llm.UserText("u1"), asstText("a2")}
	got, err := s(context.Background(), head)
	if err != nil {
		t.Fatal(err)
	}
	if got != "Summary:\nOK" {
		t.Fatalf("got %q", got)
	}
	if len(p.reqs) != 2 {
		t.Fatalf("want 2 calls (overflow + truncated retry), got %d", len(p.reqs))
	}
	// The retry dropped the oldest API-round group (u0) and prepended the marker.
	retry := p.reqs[1].Messages
	if retry[0].Text() != ptlRetryMarker {
		t.Fatalf("retry should start with the PTL marker, got %q", retry[0].Text())
	}
	for _, m := range retry {
		if m.Text() == "u0" {
			t.Fatal("retry should have dropped the oldest group (u0)")
		}
	}
}

func TestProviderSummarizerStreamingRetry(t *testing.T) {
	p := &captureProvider{reply: "<summary>R</summary>", failN: 1, failErr: errors.New("network blip")}
	s := ProviderSummarizer(p, 0)
	got, err := s(context.Background(), []llm.Message{llm.UserText("hi")})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Summary:\nR" {
		t.Fatalf("got %q", got)
	}
	if len(p.reqs) != 2 { // transient failure retried once within streamSummary
		t.Fatalf("want 2 stream attempts, got %d", len(p.reqs))
	}
}

func TestProviderSummarizerNonOverflowErrorSurfaces(t *testing.T) {
	// A persistent non-overflow error exhausts streaming retries and surfaces.
	p := &captureProvider{reply: "x", failN: 99, failErr: errors.New("boom")}
	if _, err := ProviderSummarizer(p, 0)(context.Background(), []llm.Message{llm.UserText("hi")}); err == nil {
		t.Fatal("expected error to surface")
	}
	if len(p.reqs) != maxStreamingRetries {
		t.Fatalf("want %d stream attempts, got %d", maxStreamingRetries, len(p.reqs))
	}
}
