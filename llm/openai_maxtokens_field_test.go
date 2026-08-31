package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// maxTokensKeys reports which of the two mutually exclusive output-cap keys the
// request body carried, and with what value.
func maxTokensKeys(m map[string]any) (old, new_ float64, oldSet, newSet bool) {
	old, oldSet = m["max_tokens"].(float64)
	new_, newSet = m["max_completion_tokens"].(float64)
	return
}

// Default config keeps the classic key, so upgrading the SDK cannot change what
// existing gateways receive.
func TestOpenAIMaxTokensDefaultsToClassicKey(t *testing.T) {
	stub := sse(`data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}`, ``, `data: [DONE]`, ``)
	req := CompletionRequest{MaxTokens: 4096, Messages: []Message{UserText("hi")}}
	_, m := captureReq(t, Config{Format: FormatOpenAI}, req, stub)

	old, _, oldSet, newSet := maxTokensKeys(m)
	if !oldSet || old != 4096 {
		t.Fatalf("max_tokens=%v set=%v, want 4096", old, oldSet)
	}
	if newSet {
		t.Fatal("max_completion_tokens must be absent by default")
	}
}

// Opting in moves the cap onto the new key — and drops the old one entirely,
// since OpenAI rejects a request carrying both.
func TestOpenAIMaxCompletionTokensOptIn(t *testing.T) {
	stub := sse(`data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}`, ``, `data: [DONE]`, ``)
	req := CompletionRequest{MaxTokens: 8192, Messages: []Message{UserText("hi")}}
	cfg := Config{Format: FormatOpenAI, MaxTokensField: MaxTokensFieldCompletion}
	_, m := captureReq(t, cfg, req, stub)

	_, new_, oldSet, newSet := maxTokensKeys(m)
	if !newSet || new_ != 8192 {
		t.Fatalf("max_completion_tokens=%v set=%v, want 8192", new_, newSet)
	}
	if oldSet {
		t.Fatal("max_tokens must be dropped when the new key is selected")
	}
}

// An unrecognized value is not a request-breaking error: it falls back to the
// classic key rather than emitting a bogus one.
func TestOpenAIMaxTokensFieldUnknownFallsBack(t *testing.T) {
	stub := sse(`data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}`, ``, `data: [DONE]`, ``)
	req := CompletionRequest{MaxTokens: 512, Messages: []Message{UserText("hi")}}
	_, m := captureReq(t, Config{Format: FormatOpenAI, MaxTokensField: "nonsense"}, req, stub)

	old, _, oldSet, newSet := maxTokensKeys(m)
	if !oldSet || old != 512 || newSet {
		t.Fatalf("want max_tokens=512 only, got old=%v/%v new set=%v", old, oldSet, newSet)
	}
}

// MaxTokens==0 means "no cap": neither key may appear, whichever is selected.
func TestOpenAIMaxTokensZeroOmitsBothKeys(t *testing.T) {
	stub := sse(`data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}`, ``, `data: [DONE]`, ``)
	req := CompletionRequest{Messages: []Message{UserText("hi")}} // MaxTokens == 0
	for _, field := range []string{"", MaxTokensFieldCompletion} {
		_, m := captureReq(t, Config{Format: FormatOpenAI, MaxTokensField: field}, req, stub)
		if _, _, oldSet, newSet := maxTokensKeys(m); oldSet || newSet {
			t.Fatalf("field=%q: no cap should be sent, got old=%v new=%v", field, oldSet, newSet)
		}
	}
}

// The non-streaming path builds the same body, so the switch must hold there too.
func TestOpenAIMaxCompletionTokensNonStreaming(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &got)
		w.Header().Set("content-type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p, err := NewProvider(Config{
		Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m",
		MaxTokensField: MaxTokensFieldCompletion,
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	if _, _, _, err := p.Complete(context.Background(), CompletionRequest{
		MaxTokens: 2048, Messages: []Message{UserText("go")},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	_, new_, oldSet, newSet := maxTokensKeys(got)
	if !newSet || new_ != 2048 || oldSet {
		t.Fatalf("want max_completion_tokens=2048 only, got new=%v/%v old set=%v", new_, newSet, oldSet)
	}
}

// Anthropic names the field itself; the OpenAI-only switch must not leak into it.
func TestAnthropicIgnoresMaxTokensField(t *testing.T) {
	stub := sse(`data: {"type":"message_stop"}`, ``)
	req := CompletionRequest{MaxTokens: 1024, Messages: []Message{UserText("hi")}}
	cfg := Config{Format: FormatAnthropic, MaxTokensField: MaxTokensFieldCompletion}
	_, m := captureReq(t, cfg, req, stub)

	old, _, oldSet, newSet := maxTokensKeys(m)
	if !oldSet || old != 1024 {
		t.Fatalf("anthropic max_tokens=%v set=%v, want 1024", old, oldSet)
	}
	if newSet {
		t.Fatal("anthropic must never send max_completion_tokens")
	}
}
