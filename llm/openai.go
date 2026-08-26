package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
)

type openaiProvider struct{ cfg Config }

type oaMessage struct {
	Role             string       `json:"role"`
	Content          string       `json:"content,omitempty"`
	ReasoningContent string       `json:"reasoning_content,omitempty"`
	ToolCalls        []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string       `json:"tool_call_id,omitempty"`
}

type oaToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Index    int    `json:"index,omitempty"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
	} `json:"function"`
}

type oaThinking struct {
	Type string `json:"type"`
}

type oaReq struct {
	Model           string        `json:"model"`
	Messages        []oaMessage   `json:"messages"`
	Tools           []oaTool      `json:"tools,omitempty"`
	MaxTokens       int           `json:"max_tokens,omitempty"`
	Temperature     *float64      `json:"temperature,omitempty"`
	Stop            []string      `json:"stop,omitempty"`
	Thinking        *oaThinking   `json:"thinking,omitempty"`
	ReasoningEffort string        `json:"reasoning_effort,omitempty"`
	Stream          bool          `json:"stream"`
	StreamOptions   *oaStreamOpts `json:"stream_options,omitempty"`
}

type oaStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

// toOpenAIMessages flattens the unified model into OpenAI's role-based format:
// assistant tool_use → tool_calls; user-embedded tool_result → standalone
// role:tool messages; system prompt → leading system message (FR-03.6).
func toOpenAIMessages(system string, msgs []Message) []oaMessage {
	out := make([]oaMessage, 0, len(msgs)+1)
	if system != "" {
		out = append(out, oaMessage{Role: "system", Content: system})
	}
	for _, m := range msgs {
		switch m.Role {
		case RoleAssistant:
			om := oaMessage{Role: "assistant"}
			var text strings.Builder
			var reasoning strings.Builder
			for _, b := range m.Content {
				switch b.Type {
				case BlockText:
					text.WriteString(b.Text)
				case BlockThinking:
					// DeepSeek-style thinking mode requires the previous turn's
					// reasoning_content to be passed back verbatim; dropping it
					// makes the API reject the request with a 400. Thinking text
					// lives in b.Thinking (not b.Text) — see ContentBlock.
					reasoning.WriteString(b.Thinking)
				case BlockToolUse:
					tc := oaToolCall{ID: b.ID, Type: "function"}
					tc.Function.Name = b.Name
					tc.Function.Arguments = string(b.Input)
					if tc.Function.Arguments == "" {
						tc.Function.Arguments = "{}"
					}
					om.ToolCalls = append(om.ToolCalls, tc)
				}
			}
			om.Content = text.String()
			if s := reasoning.String(); s != "" {
				om.ReasoningContent = s
			}
			// OpenAI requires every assistant message to carry content or
			// tool_calls; Content's omitempty would otherwise serialize a
			// content-less turn to a bare {"role":"assistant"} and the API 400s
			// with "content or tool_calls must be set". Rather than drop such a
			// turn, keep it with a minimal placeholder: a thinking-only turn must
			// stay so its reasoning_content is passed back (DeepSeek-style
			// thinking mode requires it), and keeping the assistant turn also
			// preserves user/assistant alternation for gateways that enforce it
			// (dropping it would leave two adjacent user messages).
			if om.Content == "" && len(om.ToolCalls) == 0 {
				om.Content = "…"
			}
			out = append(out, om)
		case RoleUser:
			var text strings.Builder
			var toolMsgs []oaMessage
			for _, b := range m.Content {
				switch b.Type {
				case BlockText:
					text.WriteString(b.Text)
				case BlockToolResult:
					toolMsgs = append(toolMsgs, oaMessage{Role: "tool", ToolCallID: b.ToolUseID, Content: flattenText(b.Content)})
				}
			}
			out = append(out, toolMsgs...)
			if s := text.String(); s != "" {
				out = append(out, oaMessage{Role: "user", Content: s})
			}
		}
	}
	return out
}

func flattenText(blocks []ContentBlock) string {
	var s strings.Builder
	for _, b := range blocks {
		if b.Type == BlockText {
			s.WriteString(b.Text)
		}
	}
	return s.String()
}

func (p *openaiProvider) buildBody(req CompletionRequest, stream bool) ([]byte, error) {
	body := oaReq{
		Model:       p.cfg.Model,
		Messages:    toOpenAIMessages(joinSystem(req.System), req.Messages),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stop:        req.Stop,
		Stream:      stream,
	}
	// stream_options.include_usage is a streaming-only field; some gateways reject
	// it on a non-streaming request, so send it only when streaming.
	if stream {
		body.StreamOptions = &oaStreamOpts{IncludeUsage: true}
	}
	// A non-empty per-request override (req.Thinking) wins over Config.ThinkingType.
	thinkingType := p.cfg.ThinkingType
	if req.Thinking != "" {
		thinkingType = req.Thinking
	}
	if thinkingType != "" {
		body.Thinking = &oaThinking{Type: thinkingType}
	}
	body.ReasoningEffort = p.cfg.ReasoningEffort // omitempty drops it when unset
	for _, t := range req.Tools {
		var ot oaTool
		ot.Type = "function"
		ot.Function.Name = t.Name
		ot.Function.Description = t.Description
		ot.Function.Parameters = t.InputSchema
		body.Tools = append(body.Tools, ot)
	}
	return json.Marshal(body)
}

func (p *openaiProvider) Stream(ctx context.Context, req CompletionRequest) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		body, err := p.buildBody(req, true)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		resp, err := doStream(ctx, p.cfg, p.cfg.BaseURL+"/chat/completions", body, func(r *http.Request) {
			r.Header.Set("content-type", "application/json")
			r.Header.Set("authorization", "Bearer "+p.cfg.APIKey)
			r.Header.Set("accept", "text/event-stream")
		}, "openai")
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		defer resp.Body.Close()

		scan := newSSEScanner(resp.Body)
		started := map[int]bool{}
		var stopReason string
		var usage Usage
		for {
			_, data, serr := scan.next()
			if serr == io.EOF {
				yield(StreamEvent{Type: SEMessageDelta, StopReason: stopReason, Usage: usage}, nil)
				yield(StreamEvent{Type: SEMessageStop}, nil)
				return
			}
			if serr != nil {
				yield(StreamEvent{}, serr)
				return
			}
			data = strings.TrimSpace(data)
			if data == "" {
				continue
			}
			if data == "[DONE]" {
				if !yield(StreamEvent{Type: SEMessageDelta, StopReason: stopReason, Usage: usage}, nil) {
					return
				}
				yield(StreamEvent{Type: SEMessageStop}, nil)
				return
			}
			evs, sr, u, ok := parseOpenAIFrame(data, started)
			if sr != "" {
				stopReason = sr
			}
			if ok {
				usage = u
			}
			for _, ev := range evs {
				if !yield(ev, nil) {
					return
				}
			}
		}
	}
}

// Complete performs a real non-streaming completion: it sends stream:false and
// parses the single JSON response body into a full assistant Message. It returns
// the assembled message, the normalized stop reason, and token usage. A 200
// response whose body is an error object (some gateways do this instead of a 4xx)
// is surfaced as an error rather than an empty completion.
func (p *openaiProvider) Complete(ctx context.Context, req CompletionRequest) (Message, string, Usage, error) {
	body, err := p.buildBody(req, false)
	if err != nil {
		return Message{}, "", Usage{}, err
	}
	resp, err := doStream(ctx, p.cfg, p.cfg.BaseURL+"/chat/completions", body, func(r *http.Request) {
		r.Header.Set("content-type", "application/json")
		r.Header.Set("authorization", "Bearer "+p.cfg.APIKey)
		r.Header.Set("accept", "application/json")
	}, "openai")
	if err != nil {
		return Message{}, "", Usage{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return Message{}, "", Usage{}, err
	}
	return parseOpenAIResponse(raw)
}

// parseOpenAIResponse decodes a non-streaming Chat Completions response body into
// the neutral Message model. reasoning_content/reasoning (a reasoning model's
// thinking) is mapped to a thinking block; content to a text block; tool_calls to
// tool_use blocks — the same shape the streaming path accumulates.
func parseOpenAIResponse(raw []byte) (Message, string, Usage, error) {
	var r struct {
		Choices []struct {
			Message struct {
				Content          string       `json:"content"`
				ReasoningContent string       `json:"reasoning_content"`
				Reasoning        string       `json:"reasoning"`
				Refusal          string       `json:"refusal"`
				ToolCalls        []oaToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
		Error *struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return Message{}, "", Usage{}, fmt.Errorf("openai: decode response: %w (body: %s)", err, truncate(string(raw), 500))
	}
	if r.Error != nil {
		return Message{}, "", Usage{}, fmt.Errorf("openai: %s", r.Error.Message)
	}
	var usage Usage
	if r.Usage != nil {
		usage = Usage{
			InputTokens:     r.Usage.PromptTokens,
			OutputTokens:    r.Usage.CompletionTokens,
			CacheReadTokens: r.Usage.PromptTokensDetails.CachedTokens,
		}
	}
	if len(r.Choices) == 0 {
		return Message{}, "", usage, fmt.Errorf("openai: empty response (no choices; body: %s)", truncate(string(raw), 500))
	}
	ch := r.Choices[0]
	var blocks []ContentBlock
	if think := pickReasoning(ch.Message.ReasoningContent, ch.Message.Reasoning); think != "" {
		blocks = append(blocks, ContentBlock{Type: BlockThinking, Thinking: think})
	}
	if ch.Message.Content != "" {
		blocks = append(blocks, TextBlock(ch.Message.Content))
	} else if ch.Message.Refusal != "" {
		// A refused turn carries content:null and the reason in `refusal`. Reading
		// only content would hand the caller an empty assistant turn — the model
		// did answer, it just declined.
		blocks = append(blocks, TextBlock(ch.Message.Refusal))
	}
	for _, tc := range ch.Message.ToolCalls {
		in := tc.Function.Arguments
		if strings.TrimSpace(in) == "" {
			in = "{}"
		}
		blocks = append(blocks, ContentBlock{Type: BlockToolUse, ID: tc.ID, Name: tc.Function.Name, Input: []byte(in)})
	}
	return Message{Role: RoleAssistant, Content: blocks}, mapOpenAIFinish(ch.FinishReason), usage, nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

// pickReasoning returns the thinking text a Chat Completions payload carries.
// reasoning_content is the DeepSeek-style field most gateways use; a plain
// `reasoning` is what OpenRouter and vLLM's reasoning parsers emit. Reading only
// the first silently drops a whole turn of thinking — and when the model puts
// everything there (an unclosed </think>), the turn decodes to an empty message
// that looks like the model just said nothing.
func pickReasoning(reasoningContent, reasoning string) string {
	if reasoningContent != "" {
		return reasoningContent
	}
	return reasoning
}

func parseOpenAIFrame(data string, started map[int]bool) (evs []StreamEvent, stopReason string, usage Usage, hasUsage bool) {
	var f struct {
		Choices []struct {
			Delta struct {
				Content          string       `json:"content"`
				ReasoningContent string       `json:"reasoning_content"`
				Reasoning        string       `json:"reasoning"`
				Refusal          string       `json:"refusal"`
				ToolCalls        []oaToolCall `json:"tool_calls"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(data), &f); err != nil {
		return nil, "", Usage{}, false
	}
	if f.Usage != nil {
		usage = Usage{InputTokens: f.Usage.PromptTokens, OutputTokens: f.Usage.CompletionTokens, CacheReadTokens: f.Usage.PromptTokensDetails.CachedTokens}
		hasUsage = true
	}
	if len(f.Choices) == 0 {
		return nil, "", usage, hasUsage
	}
	ch := f.Choices[0]
	if think := pickReasoning(ch.Delta.ReasoningContent, ch.Delta.Reasoning); think != "" {
		evs = append(evs, StreamEvent{Type: SEThinkingDelta, Text: think})
	}
	if ch.Delta.Content != "" {
		evs = append(evs, StreamEvent{Type: SETextDelta, Text: ch.Delta.Content})
	}
	if ch.Delta.Refusal != "" {
		// See parseOpenAIResponse: a refusal streams in `refusal`, not `content`.
		evs = append(evs, StreamEvent{Type: SETextDelta, Text: ch.Delta.Refusal})
	}
	for _, tc := range ch.Delta.ToolCalls {
		if !started[tc.Index] && (tc.ID != "" || tc.Function.Name != "") {
			started[tc.Index] = true
			evs = append(evs, StreamEvent{Type: SEToolUseStart, ToolID: tc.ID, ToolName: tc.Function.Name})
		}
		if tc.Function.Arguments != "" {
			evs = append(evs, StreamEvent{Type: SEToolInputJSON, Text: tc.Function.Arguments})
		}
	}
	if ch.FinishReason != "" {
		stopReason = mapOpenAIFinish(ch.FinishReason)
	}
	return evs, stopReason, usage, hasUsage
}

func mapOpenAIFinish(r string) string {
	switch r {
	case "tool_calls":
		return "tool_use"
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		return r
	}
}
