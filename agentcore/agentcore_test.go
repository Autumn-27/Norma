package agentcore

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

func sse(lines ...string) string { return strings.Join(lines, "\n") + "\n" }

// TestEndToEnd drives the whole stack: agentcore.Session → harness loop → real
// Anthropic HTTP provider (fake SSE server) → real Bash tool, over two turns.
func TestEndToEnd(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			w.Write([]byte(sse(
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"Bash"}}`, ``,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"echo loop-works\"}"}}`, ``,
				`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}`, ``,
				`data: {"type":"message_stop"}`, ``,
			)))
			return
		}
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		if !strings.Contains(string(body), "loop-works") {
			t.Errorf("second request missing tool_result")
		}
		w.Write([]byte(sse(
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}`, ``,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The command ran."}}`, ``,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`, ``,
			`data: {"type":"message_stop"}`, ``,
		)))
	}))
	defer srv.Close()

	prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	opts := Options{
		Provider:       prov,
		SystemPrompt:   []string{"You are a test agent."},
		Tools:          []tool.CoreTool{tool.NewBash()},
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
	}

	var sawText string
	s := NewSession(opts)
	for ev, err := range s.Prompt(context.Background(), "run echo") {
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		if ev.Kind == harness.KindText {
			sawText += ev.Text
		}
	}
	if sawText != "The command ran." {
		t.Fatalf("streamed text=%q", sawText)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("model calls=%d", calls)
	}
	if len(s.Messages()) != 4 {
		t.Fatalf("messages=%d", len(s.Messages()))
	}
}

func TestRunConvenience(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(sse(
			`data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}`, ``,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`, ``,
			`data: [DONE]`, ``,
		)))
	}))
	defer srv.Close()
	prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	out, err := Run(context.Background(), Options{Provider: prov, SystemPrompt: []string{"hi"}, Tools: []tool.CoreTool{}}, "say hello")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hello" {
		t.Fatalf("out=%q", out)
	}
}
