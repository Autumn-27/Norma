package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// emptyStop is a completed stream that produced no content (a normal `stop` with
// nothing before it) — the degenerate empty completion the retry targets.
var emptyStop = sse(`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, ``, `data: [DONE]`, ``)

// A streamed empty completion is re-requested; a subsequent non-empty attempt
// surfaces normally. Exactly two requests: one empty, one recovered.
func TestOpenAIStreamEmptyResponseRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Write([]byte(emptyStop))
			return
		}
		w.Write([]byte(sse(`data: {"choices":[{"index":0,"delta":{"content":"recovered"}}]}`, ``, `data: [DONE]`, ``)))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if msg.Text() != "recovered" {
		t.Fatalf("text=%q", msg.Text())
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d, want 2 (one empty + one retry)", calls.Load())
	}
}

// An always-empty stream is retried a bounded number of times, then surfaced as
// an empty message rather than looping forever.
func TestOpenAIStreamEmptyResponseExhausts(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(emptyStop))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if len(msg.Content) != 0 {
		t.Fatalf("want empty message after exhaustion, got %+v", msg.Content)
	}
	if calls.Load() != int32(emptyResponseRetries+1) {
		t.Fatalf("calls=%d, want %d", calls.Load(), emptyResponseRetries+1)
	}
}

// A max_tokens truncation with no content must NOT retry — that recovery is the
// harness's to drive, and re-requesting would fight it.
func TestOpenAIStreamEmptyMaxTokensNoRetry(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(sse(`data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`, ``, `data: [DONE]`, ``)))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	_ = collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if calls.Load() != 1 {
		t.Fatalf("calls=%d, want 1 (max_tokens must not retry)", calls.Load())
	}
}

// A stream that produces content on the first attempt is never retried.
func TestOpenAIStreamContentNoRetry(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(sse(`data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`, ``, `data: [DONE]`, ``)))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if msg.Text() != "hi" || calls.Load() != 1 {
		t.Fatalf("text=%q calls=%d", msg.Text(), calls.Load())
	}
}

// The non-streaming path retries an empty completion the same way.
func TestOpenAICompleteEmptyResponseRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		if calls.Add(1) == 1 {
			w.Write([]byte(`{"choices":[{"message":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0}}`))
			return
		}
		w.Write([]byte(`{"choices":[{"message":{"content":"recovered"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg, _, _, err := p.Complete(context.Background(), CompletionRequest{Messages: []Message{UserText("go")}})
	if err != nil {
		t.Fatalf("Complete err: %v", err)
	}
	if msg.Text() != "recovered" {
		t.Fatalf("text=%q", msg.Text())
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d, want 2", calls.Load())
	}
}

func TestIsEmptyResponseRetryable(t *testing.T) {
	for _, tc := range []struct {
		stop string
		want bool
	}{
		{"end_turn", true},
		{"", true},
		{"max_tokens", false},
		{"tool_use", false},
	} {
		if got := isEmptyResponseRetryable(tc.stop); got != tc.want {
			t.Fatalf("isEmptyResponseRetryable(%q)=%v, want %v", tc.stop, got, tc.want)
		}
	}
}
