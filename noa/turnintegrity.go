package noa

import "maps"

// TurnGroup is one assistant turn as the provider sees it: the reasoning run,
// the assistant acts that follow it, and the tool results those acts produced.
//
// The group is the unit turn-integrity protects. A provider that validates
// thinking-to-tool_use pairing rejects a request where half a group survived.
type TurnGroup struct {
	// ReasoningIdx are the indices of the leading reasoning run.
	ReasoningIdx []int
	// ActIdx are the assistant text / tool-call indices.
	ActIdx []int
	// ResultIdx are the tool results belonging to those acts.
	ResultIdx []int
}

// indices flattens the group into every index it owns.
func (g TurnGroup) indices() []int {
	out := make([]int, 0, len(g.ReasoningIdx)+len(g.ActIdx)+len(g.ResultIdx))
	out = append(out, g.ReasoningIdx...)
	out = append(out, g.ActIdx...)
	out = append(out, g.ResultIdx...)
	return out
}

// ComputeTurnGroups partitions messages into assistant turns.
//
// Only groups that actually have a reasoning run matter for turn integrity —
// a turn with no thinking cannot lose it — but they are all returned so callers
// can reason about the structure uniformly.
func ComputeTurnGroups(msgs []CoreMessage) []TurnGroup {
	var groups []TurnGroup
	i := 0
	for i < len(msgs) {
		if msgs[i].ContentType != CTReasoning && !isAssistantAct(msgs[i]) {
			i++
			continue
		}
		var g TurnGroup
		for i < len(msgs) && msgs[i].ContentType == CTReasoning {
			g.ReasoningIdx = append(g.ReasoningIdx, i)
			i++
		}
		callIDs := map[string]bool{}
		for i < len(msgs) && isAssistantAct(msgs[i]) {
			g.ActIdx = append(g.ActIdx, i)
			if msgs[i].ToolCallID != "" {
				callIDs[msgs[i].ToolCallID] = true
			}
			i++
		}
		// The acts' results follow immediately; anything else ends the turn.
		for i < len(msgs) && msgs[i].ContentType == CTToolResult && callIDs[msgs[i].ToolCallID] {
			g.ResultIdx = append(g.ResultIdx, i)
			i++
		}
		if len(g.ReasoningIdx) > 0 || len(g.ActIdx) > 0 {
			groups = append(groups, g)
		}
	}
	return groups
}

// WithdrawSplitTurns removes from folded any turn that would be left broken: a
// group whose reasoning is being compressed away while one of its tool calls
// stays visible.
//
// The visible call would then have no reasoning run to point back to, and a
// strict-echo provider rejects the rebuilt request. Withdrawing the whole group
// is the conservative repair — the range simply compresses a little less.
//
// Returns the surviving set and the indices withdrawn.
func WithdrawSplitTurns(msgs []CoreMessage, folded map[string]bool) (map[string]bool, []string) {
	if len(folded) == 0 {
		return folded, nil
	}
	out := make(map[string]bool, len(folded))
	maps.Copy(out, folded)
	var withdrawn []string

	for _, g := range ComputeTurnGroups(msgs) {
		if len(g.ReasoningIdx) == 0 {
			continue // nothing to lose
		}
		reasoningFolded := false
		for _, i := range g.ReasoningIdx {
			if out[msgs[i].ID] {
				reasoningFolded = true
				break
			}
		}
		if !reasoningFolded {
			continue
		}
		callStaysVisible := false
		for _, i := range g.ActIdx {
			if msgs[i].ContentType == CTToolCall && !out[msgs[i].ID] {
				callStaysVisible = true
				break
			}
		}
		if !callStaysVisible {
			continue
		}
		for _, i := range g.indices() {
			if out[msgs[i].ID] {
				delete(out, msgs[i].ID)
				withdrawn = append(withdrawn, msgs[i].ID)
			}
		}
	}
	return out, withdrawn
}
