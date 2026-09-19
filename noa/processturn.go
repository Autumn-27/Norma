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
// The order encodes dependencies, not preference:
//
//	assign-refs          refs must exist before anything can address a message
//	sync-blocks          block liveness before prune decides what to hide
//	prune                the view must be final before it is measured
//	hide-compress-calls  trims history that prune left behind
//	emergency-truncate   runs BEFORE the nudge so the nudge can report it
//	nudge-inject         appends to a view nothing else will touch
func pipelineNodes() []PipelineNode {
	return append(sizingNodes(), emergencyTruncateNode(), nudgeInjectNode())
}

// sizingNodes is the prefix of the pipeline that decides how big the request
// will be: refs, block liveness, prune, and the compress-call trim. None of
// them reads TokenCount, which is exactly why they can run BEFORE the size is
// known. Everything after them (emergency-truncate, nudge-inject) reads the
// size and must not run inside a measurement — emergency-truncate in particular
// would shrink the very view being measured.
func sizingNodes() []PipelineNode {
	return []PipelineNode{
		assignRefsNode(),
		syncBlocksNode(),
		pruneNode(),
		hideCompressCallsNode(),
	}
}

// ProjectedTokenCount measures what this turn's request will actually weigh —
// the only honest input to the pressure ladder.
//
// The raw history still holds every message prune is about to replace with a
// summary, plus tool results whose paired call was compressed away (orphans
// prune strips every turn). Counting it therefore reports the size of
// everything ever said, a number that only grows: a compression that removes
// 40k tokens from the request leaves it untouched. Keyed on that, the ladder
// stays pinned in the pressure band, re-nudges on every turn, and eventually
// shreds tool results out of a request that comfortably fits.
//
// So the count is taken on the pruned view — after prune has hidden covered
// content AND stripped the orphans, before emergency-truncate has cut anything.
// The cost is one extra run of the sizing prefix per turn: O(n) over the view,
// against a provider round trip.
func ProjectedTokenCount(msgs []CoreMessage, state CompressionState, cfg Config, count TokenCountFn) int {
	ensureMaps(&state)
	io := RunPipeline(sizingNodes(), NodeIO{
		Messages: append([]CoreMessage(nil), msgs...),
		State:    CloneState(state),
	}, PipelineContext{Config: cfg, CountTokens: count})

	if count == nil {
		count = DefaultCountTokens
	}
	total := 0
	for _, m := range io.Messages {
		total += count(m.Text)
	}
	return total
}
