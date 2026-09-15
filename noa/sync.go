package noa

// syncBlocks reconciles block liveness against the messages actually present,
// and prunes the token snapshot down to live refs.
//
// A block goes inactive when it has been absorbed by a higher tier, or when
// neither its covered messages nor its rendered summary are present any more
// (the host trimmed history, or a resume started from a shorter view).
func syncBlocks(io NodeIO, _ PipelineContext) NodeIO {
	state := CloneState(io.State)

	present := make(map[string]bool, len(io.Messages))
	liveRefs := make(map[string]bool, len(io.Messages))
	for _, m := range io.Messages {
		if m.ID == "" {
			continue
		}
		present[m.ID] = true
		if ref, ok := state.MessageRefs.ByRaw[m.ID]; ok && ref != BlockedRef {
			liveRefs[ref] = true
		}
	}

	// Snapshot pruning: entries for refs no longer in view are dead weight. The
	// length check keeps the common case (nothing changed) allocation-free.
	if len(state.TokenSnapshot) != len(liveRefs) {
		kept := make(map[string]int, len(liveRefs))
		for ref, n := range state.TokenSnapshot {
			if liveRefs[ref] {
				kept[ref] = n
			}
		}
		state.TokenSnapshot = kept
	}

	consumed := map[string]bool{}
	for _, b := range state.Blocks {
		for _, id := range b.DirectBlockIDs {
			consumed[id] = true
		}
	}

	for i := range state.Blocks {
		b := &state.Blocks[i]
		if consumed[b.BlockID] {
			// Absorbed by a higher tier; its content now belongs to the parent.
			b.Active = false
			continue
		}
		b.Active = blockStillPresent(*b, present)
	}
	io.State = state
	return io
}

// blockStillPresent reports whether a block still stands for something in view:
// any covered message, or its own rendered summary.
func blockStillPresent(b CompressionBlock, present map[string]bool) bool {
	if present[SummaryMessageID(b.BlockID)] {
		return true
	}
	for _, id := range b.EffectiveMessageIDs {
		if present[id] {
			return true
		}
	}
	return false
}

// syncBlocksNode is the pipeline stage wrapper.
func syncBlocksNode() PipelineNode {
	return nodeFunc{
		name: "sync-blocks",
		enabled: func(io NodeIO, _ PipelineContext) bool {
			return len(io.State.Blocks) > 0 || len(io.State.TokenSnapshot) > 0
		},
		run: syncBlocks,
	}
}
