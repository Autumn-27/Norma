// Package llm defines a vendor-neutral message model and a pluggable Provider
// interface, with adapters that speak the Anthropic Messages API and the OpenAI
// Chat Completions API. The internal model mirrors Anthropic's content-block
// shape (the richer superset, including thinking and structured tool results);
// the OpenAI adapter translates at its boundary. This lets the harness be
// written once against a single representation while remaining compatible with
// any model reachable through either wire format.
package llm

import "encoding/json"

// Role identifies the author of a message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// BlockType enumerates the kinds of content a message block may carry.
type BlockType string

const (
	BlockText       BlockType = "text"
	BlockThinking   BlockType = "thinking"
	BlockToolUse    BlockType = "tool_use"
	BlockToolResult BlockType = "tool_result"
)

// ContentBlock is one piece of message content. Which fields are populated
// depends on Type. This is the Anthropic content-block union.
type ContentBlock struct {
	Type BlockType `json:"type"`

	// Text block.
	Text string `json:"text,omitempty"`

	// Thinking block (extended thinking). Anthropic requires the field to be
	// named "thinking" (not "text") and requires "signature" when replaying
	// the block in multi-turn history.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// ToolUse: a model request to invoke a tool.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// ToolResult: the outcome of a tool, fed back to the model.
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   []ContentBlock `json:"content,omitempty"`
	IsError   bool           `json:"is_error,omitempty"`
}

// TextBlock builds a text content block.
func TextBlock(s string) ContentBlock { return ContentBlock{Type: BlockText, Text: s} }

// ToolResultText builds a tool_result block carrying a single text block.
func ToolResultText(toolUseID, text string, isError bool) ContentBlock {
	return ContentBlock{
		Type:      BlockToolResult,
		ToolUseID: toolUseID,
		Content:   []ContentBlock{TextBlock(text)},
		IsError:   isError,
	}
}

// Message is a single conversation turn.
type Message struct {
	Role    Role           `json:"role"`
	Content []ContentBlock `json:"content"`
}

// UserText builds a user message with one text block.
func UserText(s string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{TextBlock(s)}}
}

// ToolUses returns the tool_use blocks of a message.
func (m Message) ToolUses() []ContentBlock {
	var out []ContentBlock
	for _, b := range m.Content {
		if b.Type == BlockToolUse {
			out = append(out, b)
		}
	}
	return out
}

// Text concatenates all text blocks of a message.
func (m Message) Text() string {
	var s string
	for _, b := range m.Content {
		if b.Type == BlockText {
			s += b.Text
		}
	}
	return s
}

// Usage reports token accounting, including prompt-cache tokens (FR-03.8).
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

// Add accumulates token counts from another Usage.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadTokens += o.CacheReadTokens
	u.CacheWriteTokens += o.CacheWriteTokens
}

// ToolSchema is a tool definition exposed to the model.
type ToolSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// CompletionRequest is a single, provider-neutral model invocation.
type CompletionRequest struct {
	// System is the host-controlled system prompt, in segments. The SDK injects
	// no presets (design decision 1). Segments are joined with blank lines.
	System []string
	// DynamicBoundary is the index in System separating cacheable static prefix
	// from per-request dynamic content; providers may set a cache breakpoint
	// there (FR-02.2). 0 disables.
	DynamicBoundary int
	Messages        []Message
	Tools           []ToolSchema
	MaxTokens       int
	Temperature     *float64
	Stop            []string
	// Thinking overrides the provider's thinking mode for THIS request only:
	// "" inherits Config.ThinkingType; "enabled"/"disabled" force that mode.
	// Compaction's summary call sets "disabled" so the summarizer never spends
	// tokens on reasoning and never replays thinking blocks (which Anthropic
	// rejects when thinking is off).
	Thinking string
}
