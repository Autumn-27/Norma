package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestOpenAICompleteNonStreaming verifies the real non-streaming path: the wire
// carries stream:false (and no stream_options), and a single JSON response is
// parsed into text + thinking + tool_use with usage and stop reason.
func TestOpenAICompleteNonStreaming(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path=%s", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("content-type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"Hello","reasoning_content":"let me think","tool_calls":[{"id":"c1","type":"function","function":{"name":"Read","arguments":"{\"file_path\":\"a\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":4}}}`))
	}))
	defer srv.Close()

	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg, stop, usage, err := p.Complete(context.Background(), CompletionRequest{Messages: []Message{UserText("go")}})
	if err != nil {
		t.Fatalf("Complete err: %v", err)
	}
	// Wire must be non-streaming.
	if gotBody["stream"] != false {
		t.Fatalf("stream=%v, want false", gotBody["stream"])
	}
	if _, ok := gotBody["stream_options"]; ok {
		t.Fatal("stream_options must be omitted on a non-streaming request")
	}
	if msg.Text() != "Hello" {
		t.Fatalf("text=%q", msg.Text())
	}
	// reasoning_content → thinking block.
	var thinking string
	for _, b := range msg.Content {
		if b.Type == BlockThinking {
			thinking = b.Thinking
		}
	}
	if thinking != "let me think" {
		t.Fatalf("thinking=%q", thinking)
	}
	if u := msg.ToolUses(); len(u) != 1 || u[0].Name != "Read" || string(u[0].Input) != `{"file_path":"a"}` {
		t.Fatalf("tooluse=%+v", u)
	}
	if stop != "tool_use" { // mapped from finish_reason "tool_calls"
		t.Fatalf("stop=%q", stop)
	}
	if usage.InputTokens != 12 || usage.OutputTokens != 8 || usage.CacheReadTokens != 4 {
		t.Fatalf("usage=%+v", usage)
	}
}

// A 200 response whose body is an error object must surface as an error, not an
// empty completion — this is the failure mode that made flaky gateways look like
// a silently "completed" empty turn on the streaming path.
func TestOpenAICompleteErrorBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		w.Write([]byte(`{"error":{"message":"content filtered","type":"invalid_request_error"}}`))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	_, _, _, err := p.Complete(context.Background(), CompletionRequest{Messages: []Message{UserText("go")}})
	if err == nil || !strings.Contains(err.Error(), "content filtered") {
		t.Fatalf("want error surfacing the body, got %v", err)
	}
}

// TestAnthropicCompleteNonStreaming verifies the Messages API non-streaming path.
func TestAnthropicCompleteNonStreaming(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path=%s", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("content-type", "application/json")
		w.Write([]byte(`{"type":"message","content":[{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"text","text":"Hi"},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":2}}`))
	}))
	defer srv.Close()

	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", APIVersion: "2023-06-01"})
	msg, stop, usage, err := p.Complete(context.Background(), CompletionRequest{Messages: []Message{UserText("go")}})
	if err != nil {
		t.Fatalf("Complete err: %v", err)
	}
	if gotBody["stream"] != false {
		t.Fatalf("stream=%v, want false", gotBody["stream"])
	}
	if msg.Text() != "Hi" {
		t.Fatalf("text=%q", msg.Text())
	}
	if u := msg.ToolUses(); len(u) != 1 || u[0].Name != "Bash" || string(u[0].Input) != `{"command":"ls"}` {
		t.Fatalf("tooluse=%+v", u)
	}
	if stop != "tool_use" {
		t.Fatalf("stop=%q", stop)
	}
	// InputTokens includes cache-read (OpenAI-style total input semantics).
	if usage.InputTokens != 12 || usage.OutputTokens != 5 || usage.CacheReadTokens != 2 {
		t.Fatalf("usage=%+v", usage)
	}
}
