package llm

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// Some gateways (OpenRouter) and vLLM's reasoning parsers put a reasoning
// model's thinking in delta.reasoning rather than delta.reasoning_content.
// Reading only the latter dropped the whole turn's thinking: a turn that emitted
// nothing but reasoning + tool_calls surfaced as an empty message, which the
// caller then rendered as a text-less run.
func TestOpenAIStreamReasoningAlias(t *testing.T) {
	started := map[int]bool{}
	evs, _, _, _ := parseOpenAIFrame(`{"choices":[{"index":0,"delta":{"reasoning":"I need to verify"}}]}`, started)
	if len(evs) != 1 || evs[0].Type != SEThinkingDelta || evs[0].Text != "I need to verify" {
		t.Fatalf("evs=%+v", evs)
	}
}

// reasoning_content wins when a provider sends both, so DeepSeek-style payloads
// keep their existing behaviour.
func TestOpenAIStreamReasoningContentWins(t *testing.T) {
	started := map[int]bool{}
	evs, _, _, _ := parseOpenAIFrame(`{"choices":[{"index":0,"delta":{"reasoning":"alias","reasoning_content":"canonical"}}]}`, started)
	if len(evs) != 1 || evs[0].Text != "canonical" {
		t.Fatalf("evs=%+v", evs)
	}
}

// End-to-end replay of a vLLM (qwen) stream whose whole turn is reasoning +
// tool_calls: the accumulated message must carry the thinking, not come back
// empty.
func TestOpenAIStreamVLLMReasoningReplay(t *testing.T) {
	body := sse(
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"reasoning":"I"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"reasoning":" need to verify"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call-1","type":"function","index":0,"function":{"name":"list_facts"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: {"choices":[],"usage":{"prompt_tokens":20860,"total_tokens":20899,"completion_tokens":39}}`,
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
	if thinking := thinkingOf(msg); thinking != "I need to verify" {
		t.Fatalf("thinking=%q", thinking)
	}
	if u := msg.ToolUses(); len(u) != 1 || u[0].Name != "list_facts" {
		t.Fatalf("tooluse=%+v", u)
	}
}

func TestOpenAIResponseReasoningAlias(t *testing.T) {
	msg, stop, _, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":"","reasoning":"let me think"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("parse err: %v", err)
	}
	if stop != "end_turn" {
		t.Fatalf("stop=%q", stop)
	}
	if thinking := thinkingOf(msg); thinking != "let me think" {
		t.Fatalf("thinking=%q", thinking)
	}
}

// A refused turn carries content:null and the reason in `refusal`; it must not
// decode to an empty assistant message.
func TestOpenAIRefusalBecomesText(t *testing.T) {
	msg, _, _, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":null,"refusal":"I can't help with that."},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("parse err: %v", err)
	}
	if msg.Text() != "I can't help with that." {
		t.Fatalf("text=%q", msg.Text())
	}
}

func TestOpenAIStreamRefusalBecomesText(t *testing.T) {
	started := map[int]bool{}
	evs, _, _, _ := parseOpenAIFrame(`{"choices":[{"index":0,"delta":{"refusal":"I can't"}}]}`, started)
	if len(evs) != 1 || evs[0].Type != SETextDelta || evs[0].Text != "I can't" {
		t.Fatalf("evs=%+v", evs)
	}
}

func thinkingOf(msg Message) string {
	var s string
	for _, b := range msg.Content {
		if b.Type == BlockThinking {
			s += b.Thinking
		}
	}
	return s
}
