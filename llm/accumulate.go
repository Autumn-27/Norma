package llm

import (
	"encoding/json"
	"strings"
)

// Accumulator reassembles a complete assistant Message from a stream of
// normalized StreamEvents. The harness feeds every event to Add; once the
// stream ends, Message returns the finished assistant turn (text, thinking, and
// tool_use blocks in arrival order).
type Accumulator struct {
	blocks     []ContentBlock
	toolInputs []*strings.Builder
	toolSigs   []string // ThoughtSignature for the tool_use block at the same index as toolInputs
	// toolPos maps a provider-assigned StreamEvent.ToolIndex to its position in
	// toolSigs, so a signature frame naming index 0 lands on call 0 even after
	// call 1 has started.
	toolPos    map[int]int
	textBuf    *strings.Builder
	thinkBuf   *strings.Builder
	thinkSig   string // signature for the current thinking block
	StopReason string
	Usage      Usage
}

// NewAccumulator returns an empty Accumulator.
func NewAccumulator() *Accumulator { return &Accumulator{} }

// Add folds one event into the accumulator.
func (a *Accumulator) Add(ev StreamEvent) {
	switch ev.Type {
	case SEMessageStart:
		a.Usage.Add(ev.Usage)
	case SETextDelta:
		if a.textBuf == nil {
			a.flushThinking()
			a.textBuf = &strings.Builder{}
		}
		a.textBuf.WriteString(ev.Text)
	case SEThinkingDelta:
		if a.thinkBuf == nil {
			a.flushText()
			a.thinkBuf = &strings.Builder{}
		}
		a.thinkBuf.WriteString(ev.Text)
	case SEThinkingSignature:
		a.thinkSig = ev.Text
	case SEToolUseStart:
		a.flushText()
		a.flushThinking()
		a.blocks = append(a.blocks, ContentBlock{Type: BlockToolUse, ID: ev.ToolID, Name: ev.ToolName})
		a.toolInputs = append(a.toolInputs, &strings.Builder{})
		a.toolSigs = append(a.toolSigs, "")
		if a.toolPos == nil {
			a.toolPos = map[int]int{}
		}
		a.toolPos[ev.ToolIndex] = len(a.toolSigs) - 1
	case SEToolInputJSON:
		if n := len(a.toolInputs); n > 0 {
			a.toolInputs[n-1].WriteString(ev.Text)
		}
	case SEToolUseSignature:
		// Prefer the provider's index; adapters that do not set one (Anthropic,
		// the Responses API) leave it 0, which for them means "the call in
		// flight" — the fallback covers that and an index we never saw start.
		if pos, ok := a.toolPos[ev.ToolIndex]; ok {
			a.toolSigs[pos] = ev.Text
		} else if n := len(a.toolSigs); n > 0 {
			a.toolSigs[n-1] = ev.Text
		}
	case SEMessageDelta:
		if ev.StopReason != "" {
			a.StopReason = ev.StopReason
		}
		a.Usage.MergeEcho(ev.Usage)
	case SEMessageStop:
		a.flushText()
		a.flushThinking()
	}
}

func (a *Accumulator) flushText() {
	if a.textBuf != nil {
		if s := a.textBuf.String(); s != "" {
			a.blocks = append(a.blocks, TextBlock(s))
		}
		a.textBuf = nil
	}
}

func (a *Accumulator) flushThinking() {
	if a.thinkBuf != nil {
		if s := a.thinkBuf.String(); s != "" {
			a.blocks = append(a.blocks, ContentBlock{
				Type:      BlockThinking,
				Thinking:  s,
				Signature: a.thinkSig,
			})
		}
		a.thinkBuf = nil
		a.thinkSig = ""
	}
}

// Message finalizes and returns the assembled assistant message. Partial tool
// inputs are validated; malformed JSON falls back to "{}".
func (a *Accumulator) Message() Message {
	a.flushText()
	a.flushThinking()
	ti := 0
	out := make([]ContentBlock, len(a.blocks))
	for i, b := range a.blocks {
		if b.Type == BlockToolUse {
			raw := strings.TrimSpace(a.toolInputs[ti].String())
			if raw == "" || !json.Valid([]byte(raw)) {
				raw = "{}"
			}
			b.Input = json.RawMessage(raw)
			b.ThoughtSignature = a.toolSigs[ti]
			ti++
		}
		out[i] = b
	}
	return Message{Role: RoleAssistant, Content: out}
}
