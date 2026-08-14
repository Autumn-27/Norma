package llm

import (
	"context"
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"strings"
)

type openaiProvider struct{ cfg Config }

type oaMessage struct {
	Role       string       `json:"role"`
	Content    string       `json:"content,omitempty"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
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
	Model           string      `json:"model"`
	Messages        []oaMessage `json:"messages"`
	Tools           []oaTool    `json:"tools,omitempty"`
	MaxTokens       int         `json:"max_tokens,omitempty"`
	Temperature     *float64    `json:"temperature,omitempty"`
	Stop            []string    `json:"stop,omitempty"`
	Thinking        *oaThinking `json:"thinking,omitempty"`
	ReasoningEffort string      `json:"reasoning_effort,omitempty"`
	Stream          bool        `json:"stream"`
	StreamOptions   struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
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
			for _, b := range m.Content {
				switch b.Type {
				case BlockText:
					text.WriteString(b.Text)
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
			// OpenAI requires an assistant message to carry content or
			// tool_calls. A turn that produced only thinking (reasoning), or
			// an empty/truncated turn, has neither once thinking is stripped
			// here — and Content's omitempty would drop an empty string, so
			// the message would serialize to {"role":"assistant"} and the API
			// rejects it with 400 "content or tool_calls must be set". Skip it.
			if om.Content == "" && len(om.ToolCalls) == 0 {
				continue
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

func (p *openaiProvider) buildBody(req CompletionRequest) ([]byte, error) {
	body := oaReq{
		Model:       p.cfg.Model,
		Messages:    toOpenAIMessages(joinSystem(req.System), req.Messages),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stop:        req.Stop,
		Stream:      true,
	}
	body.StreamOptions.IncludeUsage = true
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
		body, err := p.buildBody(req)
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

func parseOpenAIFrame(data string, started map[int]bool) (evs []StreamEvent, stopReason string, usage Usage, hasUsage bool) {
	var f struct {
		Choices []struct {
			Delta struct {
				Content          string       `json:"content"`
				ReasoningContent string       `json:"reasoning_content"`
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
	if ch.Delta.ReasoningContent != "" {
		evs = append(evs, StreamEvent{Type: SEThinkingDelta, Text: ch.Delta.ReasoningContent})
	}
	if ch.Delta.Content != "" {
		evs = append(evs, StreamEvent{Type: SETextDelta, Text: ch.Delta.Content})
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
