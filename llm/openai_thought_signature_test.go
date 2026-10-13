package llm

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// Gemini's OpenAI-compatible endpoint attaches a thought_signature to a
// tool_calls entry as {"extra_content":{"google":{"thought_signature":"..."}}}.
// A thinking model rejects a replayed tool call that doesn't echo this value
// back verbatim ("Function call is missing a thought_signature in functionCall
// parts"), so the non-streaming adapter must capture it on the tool_use block.
func TestOpenAIResponseCapturesThoughtSignature(t *testing.T) {
	msg, _, _, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":"","tool_calls":[
		{"id":"call-1","type":"function","function":{"name":"graph_overview","arguments":"{}"},
		 "extra_content":{"google":{"thought_signature":"sig-abc"}}}
	]},"finish_reason":"tool_calls"}]}`))
	if err != nil {
		t.Fatalf("parse err: %v", err)
	}
	uses := msg.ToolUses()
	if len(uses) != 1 {
		t.Fatalf("tooluse=%+v", uses)
	}
	if uses[0].ThoughtSignature != "sig-abc" {
		t.Fatalf("ThoughtSignature=%q, want %q", uses[0].ThoughtSignature, "sig-abc")
	}
}

// A tool call with no extra_content (every non-Gemini gateway) must not
// synthesize a signature out of thin air.
func TestOpenAIResponseWithoutThoughtSignature(t *testing.T) {
	msg, _, _, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":"","tool_calls":[
		{"id":"call-1","type":"function","function":{"name":"list_files","arguments":"{}"}}
	]},"finish_reason":"tool_calls"}]}`))
	if err != nil {
		t.Fatalf("parse err: %v", err)
	}
	if got := msg.ToolUses()[0].ThoughtSignature; got != "" {
		t.Fatalf("ThoughtSignature=%q, want empty", got)
	}
}

// Replaying the turn in the next request must echo the signature back
// verbatim on the same tool call, in the shape Gemini expects it.
func TestToOpenAIMessagesReplaysThoughtSignature(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockToolUse, ID: "call-1", Name: "graph_overview", Input: []byte(`{}`), ThoughtSignature: "sig-abc"},
		}},
	}
	out := toOpenAIMessages("", msgs)
	if len(out) != 1 || len(out[0].ToolCalls) != 1 {
		t.Fatalf("out=%+v", out)
	}
	tc := out[0].ToolCalls[0]
	if tc.ExtraContent == nil || tc.ExtraContent.Google == nil {
		t.Fatalf("ExtraContent not set: %+v", tc)
	}
	if got := tc.ExtraContent.Google.ThoughtSignature; got != "sig-abc" {
		t.Fatalf("ThoughtSignature=%q, want %q", got, "sig-abc")
	}
}

// A tool_use block with no signature (every non-Gemini turn) must not grow an
// extra_content wrapper that other gateways don't expect.
func TestToOpenAIMessagesOmitsExtraContentWhenNoSignature(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockToolUse, ID: "call-1", Name: "list_files", Input: []byte(`{}`)},
		}},
	}
	out := toOpenAIMessages("", msgs)
	tc := out[0].ToolCalls[0]
	if tc.ExtraContent != nil {
		t.Fatalf("ExtraContent=%+v, want nil", tc.ExtraContent)
	}
}

// End-to-end: a streamed tool call whose final delta carries extra_content
// must accumulate into a tool_use block with ThoughtSignature set, matching
// how SEThinkingSignature attaches to a thinking block.
func TestOpenAIStreamThoughtSignatureRoundTrip(t *testing.T) {
	body := sse(
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-1","type":"function","index":0,"function":{"name":"graph_overview"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig-abc"}}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	uses := msg.ToolUses()
	if len(uses) != 1 || uses[0].Name != "graph_overview" {
		t.Fatalf("tooluse=%+v", uses)
	}
	if uses[0].ThoughtSignature != "sig-abc" {
		t.Fatalf("ThoughtSignature=%q, want %q", uses[0].ThoughtSignature, "sig-abc")
	}
}
