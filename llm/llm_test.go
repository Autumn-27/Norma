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

func collect(t *testing.T, p Provider, req CompletionRequest) Message {
	t.Helper()
	acc := NewAccumulator()
	for ev, err := range p.Stream(context.Background(), req) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		acc.Add(ev)
	}
	return acc.Message()
}

func TestAccumulator(t *testing.T) {
	acc := NewAccumulator()
	for _, ev := range []StreamEvent{
		{Type: SETextDelta, Text: "Hel"},
		{Type: SETextDelta, Text: "lo"},
		{Type: SEToolUseStart, ToolID: "t1", ToolName: "Bash"},
		{Type: SEToolInputJSON, Text: `{"command":`},
		{Type: SEToolInputJSON, Text: `"ls"}`},
		{Type: SEMessageStop},
	} {
		acc.Add(ev)
	}
	msg := acc.Message()
	if msg.Text() != "Hello" {
		t.Fatalf("text=%q", msg.Text())
	}
	uses := msg.ToolUses()
	if len(uses) != 1 || uses[0].Name != "Bash" || string(uses[0].Input) != `{"command":"ls"}` {
		t.Fatalf("tooluse=%+v", uses)
	}
}

func sse(lines ...string) string {
	var b strings.Builder
	for _, l := range lines {
		b.WriteString(l)
		b.WriteString("\n")
	}
	return b.String()
}

func TestAnthropicAdapter(t *testing.T) {
	body := sse(
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}`,
		``,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi "}}`,
		``,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu1","name":"Read"}}`,
		``,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"a\"}"}}`,
		``,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
		``,
		`data: {"type":"message_stop"}`,
		``,
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path=%s", r.URL.Path)
		}
		w.Write([]byte(body))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if msg.Text() != "Hi " {
		t.Fatalf("text=%q", msg.Text())
	}
	if u := msg.ToolUses(); len(u) != 1 || u[0].Name != "Read" || string(u[0].Input) != `{"file_path":"a"}` {
		t.Fatalf("tooluse=%+v", u)
	}
}

// captureBody runs one request against a stub server and returns the raw request
// body the provider sent, so we can assert request-shaping (thinking/effort).
func captureBody(t *testing.T, cfg Config, stubSSE string) map[string]any {
	t.Helper()
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Write([]byte(stubSSE))
	}))
	defer srv.Close()
	cfg.BaseURL, cfg.APIKey, cfg.Model = srv.URL, "k", "m"
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal request body: %v\n%s", err, raw)
	}
	return m
}

func TestThinkingParamsAnthropic(t *testing.T) {
	stub := sse(`data: {"type":"message_stop"}`, ``)
	// set → fields present
	m := captureBody(t, Config{Format: FormatAnthropic, ThinkingType: "enabled", ReasoningEffort: "high"}, stub)
	if th, _ := m["thinking"].(map[string]any); th["type"] != "enabled" {
		t.Fatalf("thinking=%v", m["thinking"])
	}
	if oc, _ := m["output_config"].(map[string]any); oc["effort"] != "high" {
		t.Fatalf("output_config=%v", m["output_config"])
	}
	// unset → both omitted
	m = captureBody(t, Config{Format: FormatAnthropic}, stub)
	if _, ok := m["thinking"]; ok {
		t.Fatalf("thinking should be omitted: %v", m["thinking"])
	}
	if _, ok := m["output_config"]; ok {
		t.Fatalf("output_config should be omitted: %v", m["output_config"])
	}
}

func TestThinkingParamsOpenAI(t *testing.T) {
	stub := sse(`data: [DONE]`, ``)
	// set → fields present
	m := captureBody(t, Config{Format: FormatOpenAI, ThinkingType: "disabled", ReasoningEffort: "max"}, stub)
	if th, _ := m["thinking"].(map[string]any); th["type"] != "disabled" {
		t.Fatalf("thinking=%v", m["thinking"])
	}
	if m["reasoning_effort"] != "max" {
		t.Fatalf("reasoning_effort=%v", m["reasoning_effort"])
	}
	// unset → both omitted
	m = captureBody(t, Config{Format: FormatOpenAI}, stub)
	if _, ok := m["thinking"]; ok {
		t.Fatalf("thinking should be omitted: %v", m["thinking"])
	}
	if _, ok := m["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted: %v", m["reasoning_effort"])
	}
}

func TestOpenAIAdapter(t *testing.T) {
	body := sse(
		`data: {"choices":[{"delta":{"content":"Hi "},"finish_reason":null}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"Read","arguments":""}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"file_path\":\"a\"}"}}]}}]}`,
		``,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":7}}`,
		``,
		`data: [DONE]`,
		``,
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path=%s", r.URL.Path)
		}
		w.Write([]byte(body))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if msg.Text() != "Hi " {
		t.Fatalf("text=%q", msg.Text())
	}
	if u := msg.ToolUses(); len(u) != 1 || u[0].Name != "Read" || string(u[0].Input) != `{"file_path":"a"}` {
		t.Fatalf("tooluse=%+v", u)
	}
}

func TestToOpenAIMessagesThreadsToolResults(t *testing.T) {
	msgs := []Message{
		UserText("hi"),
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockText, Text: "calling"},
			{Type: BlockToolUse, ID: "c1", Name: "Bash", Input: []byte(`{"command":"ls"}`)},
		}},
		{Role: RoleUser, Content: []ContentBlock{ToolResultText("c1", "file.txt", false)}},
	}
	oa := toOpenAIMessages("sys", msgs)
	if len(oa) != 4 || oa[0].Role != "system" || oa[2].Role != "assistant" || oa[3].Role != "tool" {
		t.Fatalf("roles=%+v", oa)
	}
	if oa[2].ToolCalls[0].ID != "c1" || oa[3].ToolCallID != "c1" {
		t.Fatalf("pairing wrong: %+v", oa)
	}
}

// A thinking-only (or empty) assistant turn must not serialize to a bare
// {"role":"assistant"} message — OpenAI rejects that with 400 "content or
// tool_calls must be set". It should be skipped entirely.
func TestToOpenAIMessagesSkipsEmptyAssistant(t *testing.T) {
	msgs := []Message{
		UserText("hi"),
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockThinking, Thinking: "pondering, no answer emitted"},
		}},
		UserText("still there?"),
	}
	oa := toOpenAIMessages("", msgs)
	for _, m := range oa {
		if m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) == 0 {
			t.Fatalf("emitted empty assistant message: %+v", oa)
		}
	}
	// The two user turns survive; the empty assistant is dropped.
	if len(oa) != 2 {
		t.Fatalf("want 2 messages, got %d: %+v", len(oa), oa)
	}
}

// filterThinkingBlocks must drop messages left with no content — Anthropic
// rejects an empty content array. Covers both a thinking-only turn collapsing
// once thinking is stripped and a genuinely empty (truncated) turn.
func TestFilterThinkingBlocksDropsEmpty(t *testing.T) {
	msgs := []Message{
		UserText("hi"),
		{Role: RoleAssistant, Content: []ContentBlock{{Type: BlockThinking, Thinking: "only thinking"}}},
		{Role: RoleAssistant, Content: []ContentBlock{}}, // empty/truncated turn
		UserText("bye"),
	}

	// Thinking disabled: thinking-only turn collapses to empty and drops; the
	// already-empty turn drops too. Two user turns remain.
	got := filterThinkingBlocks(msgs, false)
	if len(got) != 2 {
		t.Fatalf("thinking-disabled: want 2, got %d: %+v", len(got), got)
	}
	for _, m := range got {
		if len(m.Content) == 0 {
			t.Fatalf("thinking-disabled: emitted empty message: %+v", got)
		}
	}

	// Thinking enabled: the thinking-only turn is kept (valid, must be
	// replayed); only the genuinely empty turn drops. Three messages remain.
	got = filterThinkingBlocks(msgs, true)
	if len(got) != 3 {
		t.Fatalf("thinking-enabled: want 3, got %d: %+v", len(got), got)
	}
	for _, m := range got {
		if len(m.Content) == 0 {
			t.Fatalf("thinking-enabled: emitted empty message: %+v", got)
		}
	}
}

// Dropping an empty assistant that sat between a tool result and a resume nudge
// leaves two adjacent user turns; mergeAdjacentSameRole must coalesce them so
// the sequence still alternates (Anthropic rejects non-alternating roles).
func TestMergeAdjacentSameRoleAfterDrop(t *testing.T) {
	history := []Message{
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockText, Text: "calling"},
			{Type: BlockToolUse, ID: "c1", Name: "Bash", Input: []byte(`{"command":"ls"}`)},
		}},
		{Role: RoleUser, Content: []ContentBlock{ToolResultText("c1", "file.txt", false)}},
		{Role: RoleAssistant, Content: []ContentBlock{}}, // empty/truncated turn → dropped
		UserText("continue where you left off"),           // resume nudge
	}

	merged := mergeAdjacentSameRole(filterThinkingBlocks(history, false))

	// assistant(tool_use) → user(tool_result + nudge, coalesced). Alternation holds.
	if len(merged) != 2 {
		t.Fatalf("want 2 messages, got %d: %+v", len(merged), merged)
	}
	for i := 1; i < len(merged); i++ {
		if merged[i].Role == merged[i-1].Role {
			t.Fatalf("adjacent same-role at %d: %+v", i, merged)
		}
	}
	if merged[1].Role != RoleUser || len(merged[1].Content) != 2 {
		t.Fatalf("user turns not coalesced: %+v", merged[1])
	}
}
