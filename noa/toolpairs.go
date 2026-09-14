package noa

// AdjustBoundariesForToolPairs widens [start, end] so no tool exchange is cut
// in half. A range that takes a tool_use without its tool_result (or the
// reverse) leaves the rebuilt request with an unpaired block, which strict
// providers reject outright.
//
// The scan is bounded by maxScan in each direction: a tool exchange is local,
// and an unbounded walk would let one stray id drag the whole conversation in.
// Once the walk starts pulling messages in it stops at the first one that does
// not belong — the pair is contiguous, so a gap means we are past it.
func AdjustBoundariesForToolPairs(start, end int, msgs []CoreMessage, maxScan int) (int, int) {
	if start < 0 || end >= len(msgs) || start > end {
		return start, end
	}
	inRange := map[string]bool{}
	for i := start; i <= end; i++ {
		m := msgs[i]
		// Compress is hard-protected and never part of a compressible range, so
		// its exchange must not drag the boundary around.
		if m.ToolCallID != "" && m.ToolName != CompressToolName {
			inRange[m.ToolCallID] = true
		}
	}
	if len(inRange) == 0 {
		return start, end
	}

	// The scan limits are computed from the ORIGINAL boundaries. Deriving them
	// from the moving ones would let the cap slide forward with every message
	// pulled in, making maxScan unbounded in practice.
	forwardLimit := min(end+maxScan, len(msgs)-1)
	backwardLimit := max(start-maxScan, 0)

	for i := end + 1; i <= forwardLimit; i++ {
		if msgs[i].ToolCallID == "" || !inRange[msgs[i].ToolCallID] {
			break
		}
		end = i
	}
	for i := start - 1; i >= backwardLimit; i-- {
		if msgs[i].ToolCallID == "" || !inRange[msgs[i].ToolCallID] {
			break
		}
		start = i
	}
	return start, end
}

// AdjustBoundariesForReasoningPairs widens [start, end] so a reasoning run and
// the assistant act it belongs to stay together.
//
// Anthropic validates that a replayed thinking block still precedes the tool_use
// it reasoned about; splitting them makes the request fail. Norma projects
// thinking explicitly, so this is a hot path rather than dead code.
func AdjustBoundariesForReasoningPairs(start, end int, msgs []CoreMessage) (int, int) {
	if start < 0 || end >= len(msgs) || start > end {
		return start, end
	}

	// A reasoning run at the tail pulls in the assistant burst that follows it.
	if msgs[end].ContentType == CTReasoning {
		i := end + 1
		for i < len(msgs) && msgs[i].ContentType == CTReasoning {
			i++
		}
		if i < len(msgs) && isAssistantAct(msgs[i]) {
			for i < len(msgs) && isAssistantAct(msgs[i]) {
				end = i
				i++
			}
		}
	}

	// An assistant act at the head pulls in the reasoning run before it.
	if isAssistantAct(msgs[start]) {
		i := start - 1
		for i >= 0 && msgs[i].ContentType == CTReasoning {
			start = i
			i--
		}
	}
	return start, end
}

// ApplyPairBoundaryAdjustments runs both widenings to a fixed point, at most
// twice. Two passes are enough because each widening can expose work for the
// other exactly once: a tool pair pulled in may carry a reasoning run, and that
// run's assistant burst may carry one more tool result. A third pass has never
// changed anything, so the loop exits as soon as a round is a no-op.
func ApplyPairBoundaryAdjustments(start, end int, msgs []CoreMessage) (int, int) {
	for range 2 {
		s, e := start, end
		s, e = AdjustBoundariesForReasoningPairs(s, e, msgs)
		s, e = AdjustBoundariesForToolPairs(s, e, msgs, ToolPairMaxScan)
		if s == start && e == end {
			break
		}
		start, end = s, e
	}
	return start, end
}
