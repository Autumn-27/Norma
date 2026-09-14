package noa

// assignRefsNode hands out refs to messages that do not have one yet.
func assignRefsNode() PipelineNode {
	return nodeFunc{
		name: "assign-refs",
		run: func(io NodeIO, ctx PipelineContext) NodeIO {
			state := CloneState(io.State)
			res := AssignRefs(io.Messages, AssignRefsOptions{
				Existing:  state.MessageRefs,
				NextIndex: HighestUsedIndex(state.MessageRefs) + 1,
				IsProtected: func(m CoreMessage) bool {
					return IsMessageProtected(m, ctx.Config)
				},
				ShouldSkip: func(m CoreMessage) bool {
					// Synthetic messages are rendered fresh each turn and are never
					// addressable, so they must not consume ref numbers.
					return isRenderedSummaryMessage(m) || m.ID == NudgeMessageID
				},
			})
			state.MessageRefs = res.Map
			io.State = state
			return io
		},
	}
}

// ProcessTurnInput is one turn's projection request.
type ProcessTurnInput struct {
	// Messages is the freshly projected view of the full history.
	Messages []CoreMessage
	State    CompressionState
	Config   Config
	// TokenCount is the measured size of the context this turn, used for every
	// pressure and growth decision.
	TokenCount int
	// CountTokens overrides the built-in estimator.
	CountTokens TokenCountFn
}

// ProcessTurnResult is what the host renders and sends.
type ProcessTurnResult struct {
	Messages []CoreMessage
	State    CompressionState
	// Nudge is nil when nothing was injected this turn.
	Nudge *NudgeDecision
	// TruncatedCount is how many tool results emergency-truncate shortened.
	TruncatedCount int
}

// ProcessTurn projects one turn's view. It is a pure function: the input
// messages and state are not mutated, and the result is used for a single
// request.
//
// Node order is not negotiable — each stage assumes what the previous ones
// established.
func ProcessTurn(in ProcessTurnInput) ProcessTurnResult {
	ensureMaps(&in.State)
	ctx := PipelineContext{
		Config:      in.Config,
		TokenCount:  in.TokenCount,
		CountTokens: in.CountTokens,
	}
	io := NodeIO{
		Messages: append([]CoreMessage(nil), in.Messages...),
		State:    in.State,
	}
	io = RunPipeline(pipelineNodes(), io, ctx)
	return ProcessTurnResult{
		Messages:       io.Messages,
		State:          io.State,
		Nudge:          io.Nudge,
		TruncatedCount: io.TruncatedCount,
	}
}

// pipelineNodes is the per-turn projection, in order.
//
// The later stages land in M3/M4; the first three are what make the view
// correct, and the rest only make it smaller or louder.
func pipelineNodes() []PipelineNode {
	return []PipelineNode{
		assignRefsNode(),
		syncBlocksNode(),
		pruneNode(),
	}
}
