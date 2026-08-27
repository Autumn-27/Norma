package llm

import "encoding/json"

// Compact boundaries make context compaction non-destructive: the full
// conversation is retained in the message array, a boundary marker is inserted,
// and only the messages from the last boundary onward are sent to the model
// (the boundary marker itself is never sent — it only marks position).

// BlockCompactBoundary marks a compaction point. A message consisting solely of
// this block is a boundary marker; the summary that follows is a normal message.
const BlockCompactBoundary BlockType = "compact_boundary"

// BoundaryMeta is the structured payload of a compact boundary. It records
// why compaction ran and what it condensed, for diagnostics, analytics, and
// transcript replay.
type BoundaryMeta struct {
	// Trigger is what caused compaction: "auto" (proactive threshold),
	// "reactive" (recovery from a prompt-too-long error), or "manual".
	Trigger string `json:"trigger,omitempty"`
	// PreTokens is the token count just before compaction.
	PreTokens int `json:"pre_tokens"`
	// MessagesSummarized is how many messages were condensed into the summary.
	MessagesSummarized int `json:"messages_summarized,omitempty"`
	// ActiveSkills names the invoked skills re-injected verbatim after this
	// boundary (diagnostics; the instructions themselves ride in the messages).
	ActiveSkills []string `json:"active_skills,omitempty"`
}

// BoundaryMessage builds a compact-boundary marker carrying its metadata. It
// carries no model-visible text (the marker is stripped before the API call).
func BoundaryMessage(meta BoundaryMeta) Message {
	payload, _ := json.Marshal(meta)
	return Message{Role: RoleUser, Content: []ContentBlock{{
		Type:  BlockCompactBoundary,
		Input: payload,
	}}}
}

// ParseBoundaryMeta extracts the metadata from a boundary marker. It accepts
// both the structured object form and the legacy bare-integer form (which
// encoded only the pre-compaction token count).
func ParseBoundaryMeta(m Message) (BoundaryMeta, bool) {
	if !IsBoundaryMarker(m) {
		return BoundaryMeta{}, false
	}
	raw := m.Content[0].Input
	var meta BoundaryMeta
	if err := json.Unmarshal(raw, &meta); err == nil {
		return meta, true
	}
	var pre int
	if err := json.Unmarshal(raw, &pre); err == nil {
		return BoundaryMeta{PreTokens: pre}, true
	}
	return BoundaryMeta{}, true
}

// IsBoundaryMarker reports whether m is a compact-boundary marker.
func IsBoundaryMarker(m Message) bool {
	return len(m.Content) == 1 && m.Content[0].Type == BlockCompactBoundary
}

// LastBoundaryIndex returns the index of the last boundary marker, or -1.
func LastBoundaryIndex(msgs []Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		if IsBoundaryMarker(msgs[i]) {
			return i
		}
	}
	return -1
}

// MessagesAfterBoundary returns the slice from the last boundary marker onward
// (boundary marker included), or all messages if there is no boundary. This is
// the post-compaction "working set"; everything before it is retained history.
func MessagesAfterBoundary(msgs []Message) []Message {
	i := LastBoundaryIndex(msgs)
	if i < 0 {
		return msgs
	}
	return msgs[i:]
}

// MessagesForAPI returns the messages actually sent to the model: the working
// set after the last boundary, with boundary markers stripped (they are
// position markers, not content) and any tool block left unpaired by the
// boundary cut dropped (pairToolBlocks). The summary that follows a boundary is
// kept.
func MessagesForAPI(msgs []Message) []Message {
	working := MessagesAfterBoundary(msgs)
	out := make([]Message, 0, len(working))
	for _, m := range working {
		if IsBoundaryMarker(m) {
			continue
		}
		out = append(out, m)
	}
	return pairToolBlocks(out)
}

// pairToolBlocks drops tool blocks left unpaired within msgs: a tool_result
// whose tool_use is absent, or a tool_use whose tool_result is absent. The
// working set is sliced at the last boundary, and that cut can fall inside a
// tool exchange — leaving the post-boundary view starting with a tool_result
// whose tool_use was summarized away (or ending with a tool_use whose result
// is beyond the tail). Sent verbatim, the first makes the gateway 400 ("a
// message with role 'tool' must be a response to a preceding message with
// 'tool_calls'"). Well-formed histories have no unpaired blocks and pass through
// unchanged; this also repairs a resumed transcript that already baked in an
// orphan from an earlier compaction. Drops (not synthesizes), matching the
// summarizer's pairForReplay.
func pairToolBlocks(msgs []Message) []Message {
	useIDs := make(map[string]bool)
	resIDs := make(map[string]bool)
	for _, m := range msgs {
		for _, b := range m.Content {
			switch b.Type {
			case BlockToolUse:
				useIDs[b.ID] = true
			case BlockToolResult:
				resIDs[b.ToolUseID] = true
			}
		}
	}
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		kept := make([]ContentBlock, 0, len(m.Content))
		for _, b := range m.Content {
			switch b.Type {
			case BlockToolUse:
				if !resIDs[b.ID] {
					continue
				}
			case BlockToolResult:
				if !useIDs[b.ToolUseID] {
					continue
				}
			}
			kept = append(kept, b)
		}
		if len(kept) > 0 {
			out = append(out, Message{Role: m.Role, Content: kept})
		}
	}
	return out
}
