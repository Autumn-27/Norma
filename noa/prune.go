package noa

import (
	"sort"
	"strconv"
	"strings"
)

// isRenderedSummaryMessage identifies a summary this package rendered.
//
// All four conditions are load-bearing. The id prefix alone would let a
// host-authored message that happens to share the naming be silently deleted by
// a rebuild, so role, content type and the header text are checked too.
func isRenderedSummaryMessage(m CoreMessage) bool {
	if !strings.HasPrefix(m.ID, SummaryIDPrefix) {
		return false
	}
	return m.Role == RoleUser &&
		m.ContentType == CTText &&
		strings.HasPrefix(m.Text, SummaryHeader)
}

// ArchiveTag renders the pointer that lets the model (or a human) find a
// block's original content on disk. It is the last line of every summary.
func ArchiveTag(b CompressionBlock) string {
	var sb strings.Builder
	sb.WriteString(`<noa-archive block="`)
	sb.WriteString(b.BlockID)
	sb.WriteString(`" tier="`)
	sb.WriteString(strconv.Itoa(int(b.Tier)))
	sb.WriteString(`" range="`)
	sb.WriteString(b.StartRef)
	sb.WriteString("-")
	sb.WriteString(b.EndRef)
	sb.WriteString(`" path="`)
	sb.WriteString(b.ArchivePath)
	sb.WriteString(`"/>`)
	return sb.String()
}

// RenderSummaryMessage builds the message that stands in for a compressed
// range: header, the model's summary, and the archive pointer.
func RenderSummaryMessage(b CompressionBlock) CoreMessage {
	var sb strings.Builder
	sb.WriteString(SummaryHeader)
	if b.Topic != "" {
		sb.WriteString(" — ")
		sb.WriteString(b.Topic)
	}
	if s := strings.TrimSpace(b.Summary); s != "" {
		sb.WriteString("\n")
		sb.WriteString(s)
	}
	sb.WriteString("\n\n")
	sb.WriteString(ArchiveTag(b))
	return CoreMessage{
		ID:          SummaryMessageID(b.BlockID),
		Role:        RoleUser,
		ContentType: CTText,
		Text:        sb.String(),
	}
}

// prune replaces every compressed range with its summary and repairs whatever
// the substitution broke.
func prune(io NodeIO, _ PipelineContext) NodeIO {
	covered := CoveredMessageIDs(io.State)
	if len(covered) == 0 {
		return io
	}

	// Reconcile the coverage set for integrity BEFORE anything is dropped.
	// CoveredMessageIDs is the union over every active block's effective span, so
	// it can slice a unit that a single block's compress-time turn-integrity check
	// (run on that block's DIRECT ids) never saw whole — across two blocks, or
	// after a tier promotion changed a block's effective coverage. Two units must
	// stay wholly in or wholly out of coverage, or the rebuilt request is rejected:
	//   - a thinking turn — an assistant act kept while its reasoning run is hidden
	//     leaves a tool_calls message with no reasoning_content, which strict
	//     thinking providers 400 on;
	//   - a Compress exchange — its call is exempt from orphan stripping, so a
	//     half-covered pair leaves an assistant tool_calls with no tool result.
	// Both only ever un-cover messages, so neither can create a new split, and a
	// single pass reaches a fixed point.
	keepReasoningForVisibleTurns(io.Messages, covered)
	keepCompressPairsWhole(io.Messages, covered)

	indexOf := make(map[string]int, len(io.Messages))
	for i, m := range io.Messages {
		if m.ID != "" {
			if _, dup := indexOf[m.ID]; !dup {
				indexOf[m.ID] = i
			}
		}
	}

	firstUserIndex := -1
	for i, m := range io.Messages {
		if m.Role == RoleUser {
			firstUserIndex = i
			break
		}
	}

	type anchor struct {
		insertAt int
		block    CompressionBlock
		blockNum int
	}
	var anchors []anchor
	anchoredSummaryIDs := map[string]bool{}

	for _, b := range ActiveBlocks(io.State) {
		insertAt := 0
		if i, ok := indexOf[SummaryMessageID(b.BlockID)]; ok {
			// The summary is already in view — keep it where it is, so the byte
			// layout stays stable across turns and the prefix cache holds.
			insertAt = i
			anchoredSummaryIDs[SummaryMessageID(b.BlockID)] = true
		} else {
			earliest := -1
			for _, id := range b.EffectiveMessageIDs {
				if i, ok := indexOf[id]; ok && (earliest < 0 || i < earliest) {
					earliest = i
				}
			}
			if earliest >= 0 {
				insertAt = earliest
			}
		}
		n, _ := strconv.Atoi(strings.TrimPrefix(b.BlockID, "b"))
		anchors = append(anchors, anchor{insertAt: insertAt, block: b, blockNum: n})
	}

	// Stable order: by insertion point, then by block number so two blocks
	// anchored at the same index always render in creation order.
	sort.SliceStable(anchors, func(i, j int) bool {
		if anchors[i].insertAt != anchors[j].insertAt {
			return anchors[i].insertAt < anchors[j].insertAt
		}
		return anchors[i].blockNum < anchors[j].blockNum
	})

	out := make([]CoreMessage, 0, len(io.Messages)+len(anchors))
	next := 0
	for i, m := range io.Messages {
		for next < len(anchors) && anchors[next].insertAt == i {
			out = append(out, RenderSummaryMessage(anchors[next].block))
			next++
		}
		switch {
		case i == firstUserIndex:
			// The first user message is the session's task definition. Losing it
			// costs the model the reason it is doing any of this, so it survives
			// even when a block claims to cover it.
			out = append(out, m)
		case covered[m.ID]:
			// Replaced by the summary already emitted at this block's anchor.
		case anchoredSummaryIDs[m.ID]:
			// The pre-existing rendering of a summary we just re-emitted.
		default:
			out = append(out, m)
		}
	}
	for ; next < len(anchors); next++ {
		out = append(out, RenderSummaryMessage(anchors[next].block))
	}

	out = stripOrphanedToolCalls(out)
	out = stripOrphanedToolResults(out)
	out = stripOrphanedReasoning(out)

	io.Messages = out
	return io
}

// stripOrphanedToolCalls drops a tool call whose result is gone.
//
// Compress is exempt: its call is the addressable anchor of a block's summary in
// the conversation flow, and it must survive even after hide-compress-calls has
// removed the result.
func stripOrphanedToolCalls(msgs []CoreMessage) []CoreMessage {
	haveResult := map[string]bool{}
	for _, m := range msgs {
		if m.ContentType == CTToolResult && m.ToolCallID != "" {
			haveResult[m.ToolCallID] = true
		}
	}
	out := make([]CoreMessage, 0, len(msgs))
	for _, m := range msgs {
		if m.ContentType == CTToolCall && m.ToolCallID != "" &&
			m.ToolName != CompressToolName && !haveResult[m.ToolCallID] {
			continue
		}
		out = append(out, m)
	}
	return out
}

// stripOrphanedToolResults drops a result whose call is gone. Sent verbatim it
// makes strict providers reject the request.
func stripOrphanedToolResults(msgs []CoreMessage) []CoreMessage {
	haveCall := map[string]bool{}
	for _, m := range msgs {
		if m.ContentType == CTToolCall && m.ToolCallID != "" {
			haveCall[m.ToolCallID] = true
		}
	}
	out := make([]CoreMessage, 0, len(msgs))
	for _, m := range msgs {
		if m.ContentType == CTToolResult && m.ToolCallID != "" && !haveCall[m.ToolCallID] {
			continue
		}
		out = append(out, m)
	}
	return out
}

// stripOrphanedReasoning drops a reasoning run that no longer precedes an
// assistant act.
//
// This is a hot path, not dead code: Norma projects thinking blocks explicitly,
// and Anthropic rejects a replayed request whose thinking block has lost the
// tool_use it belonged to.
func stripOrphanedReasoning(msgs []CoreMessage) []CoreMessage {
	keep := make([]bool, len(msgs))
	for i := range keep {
		keep[i] = true
	}
	for i := 0; i < len(msgs); {
		if msgs[i].ContentType != CTReasoning {
			i++
			continue
		}
		runStart := i
		for i < len(msgs) && msgs[i].ContentType == CTReasoning {
			i++
		}
		// The run survives only if an assistant act follows immediately.
		if i >= len(msgs) || !isAssistantAct(msgs[i]) {
			for j := runStart; j < i; j++ {
				keep[j] = false
			}
		}
	}
	out := make([]CoreMessage, 0, len(msgs))
	for i, m := range msgs {
		if keep[i] {
			out = append(out, m)
		}
	}
	return out
}

// keepReasoningForVisibleTurns un-covers the reasoning run of any turn that
// keeps a visible assistant act.
//
// A thinking provider validates that every assistant message replaying tool_calls
// carries the reasoning_content it was produced with. If coverage hides the run
// while an act of the same turn survives, the rebuilt assistant message loses
// that content and the request 400s ("reasoning_content ... must be passed back").
// Widening the other way — hiding the act too — would drop work the model still
// needs, so the run is freed to stay beside its act, not the act cut to match a
// hidden run.
//
// A turn whose acts are ALL covered is left untouched: its reasoning stays
// covered and, with nothing visible pointing back to it, stripOrphanedReasoning
// removes it later.
func keepReasoningForVisibleTurns(msgs []CoreMessage, covered map[string]bool) {
	for _, g := range ComputeTurnGroups(msgs) {
		if len(g.ReasoningIdx) == 0 {
			continue
		}
		actVisible := false
		for _, i := range g.ActIdx {
			if !covered[msgs[i].ID] {
				actVisible = true
				break
			}
		}
		if !actVisible {
			continue
		}
		for _, i := range g.ReasoningIdx {
			delete(covered, msgs[i].ID)
		}
	}
}

// keepCompressPairsWhole un-covers both halves of a Compress exchange whenever
// coverage caught only one of them.
//
// AdjustBoundariesForToolPairs deliberately excludes Compress from range
// widening — it is hard-protected and never itself compressed — so a range
// boundary can fall between a Compress call and its result and leave the union
// covering just one side. Because stripOrphanedToolCalls exempts the Compress
// call, a covered-away result would strand that call as an unpaired tool_calls
// message, which strict providers reject ("tool_calls must be followed by tool
// messages"). Freeing both halves keeps the pair intact in the view.
//
// A pair with one half missing from msgs entirely cannot be reconciled here; the
// provider-layer send guard is the final backstop for that.
func keepCompressPairsWhole(msgs []CoreMessage, covered map[string]bool) {
	callMsgID := map[string]string{}
	resultMsgID := map[string]string{}
	compressCallIDs := map[string]bool{}
	for _, m := range msgs {
		switch {
		case m.ContentType == CTToolCall && m.ToolName == CompressToolName && m.ToolCallID != "":
			compressCallIDs[m.ToolCallID] = true
			callMsgID[m.ToolCallID] = m.ID
		case m.ContentType == CTToolResult && m.ToolCallID != "":
			resultMsgID[m.ToolCallID] = m.ID
		}
	}
	for tcid := range compressCallIDs {
		cID, okCall := callMsgID[tcid]
		rID, okResult := resultMsgID[tcid]
		if !okCall || !okResult {
			continue // a missing half — left to the send guard
		}
		if covered[cID] != covered[rID] {
			delete(covered, cID)
			delete(covered, rID)
		}
	}
}

// isAssistantAct reports whether m is something a reasoning run can belong to.
func isAssistantAct(m CoreMessage) bool {
	return m.Role == RoleAssistant && (m.ContentType == CTText || m.ContentType == CTToolCall)
}

// pruneNode is the pipeline stage wrapper.
func pruneNode() PipelineNode {
	return nodeFunc{
		name:    "prune",
		enabled: func(io NodeIO, _ PipelineContext) bool { return len(io.State.Blocks) > 0 },
		run:     prune,
	}
}
