package noa

// NodeIO is what flows between pipeline nodes: the working message view and the
// state being threaded through it.
type NodeIO struct {
	Messages []CoreMessage
	State    CompressionState
	// Nudge is set by the nudge-inject node, nil when nothing is injected.
	Nudge *NudgeDecision
	// TruncatedCount is set by emergency-truncate: how many tool results were
	// mechanically shortened this turn. The nudge renderer uses it to tell the
	// model not to re-run those tools.
	TruncatedCount int
}

// PipelineContext is the read-only environment every node sees.
type PipelineContext struct {
	Config Config
	// TokenCount is the measured size of the context this turn.
	TokenCount int
	// CountTokens is the token estimator; nil means DefaultCountTokens.
	CountTokens TokenCountFn
}

// count resolves the token estimator, defaulting when unset.
func (c PipelineContext) count(s string) int {
	if c.CountTokens == nil {
		return DefaultCountTokens(s)
	}
	return c.CountTokens(s)
}

// PipelineNode is one stage of the per-turn view projection.
type PipelineNode interface {
	Name() string
	// Enabled reports whether the node should run at all this turn. A disabled
	// node is skipped entirely, not run as a no-op.
	Enabled(io NodeIO, ctx PipelineContext) bool
	Run(io NodeIO, ctx PipelineContext) NodeIO
}

// nodeFunc adapts a plain function into a PipelineNode.
type nodeFunc struct {
	name    string
	enabled func(NodeIO, PipelineContext) bool
	run     func(NodeIO, PipelineContext) NodeIO
}

func (n nodeFunc) Name() string { return n.name }
func (n nodeFunc) Enabled(io NodeIO, ctx PipelineContext) bool {
	return n.enabled == nil || n.enabled(io, ctx)
}
func (n nodeFunc) Run(io NodeIO, ctx PipelineContext) NodeIO { return n.run(io, ctx) }

// RunPipeline threads io through the nodes in order. The order is not
// negotiable: each node assumes the invariants the previous ones established
// (refs exist before boundaries are resolved, blocks are synced before prune
// hides their coverage, and so on).
func RunPipeline(nodes []PipelineNode, io NodeIO, ctx PipelineContext) NodeIO {
	for _, n := range nodes {
		if !n.Enabled(io, ctx) {
			continue
		}
		io = n.Run(io, ctx)
	}
	return io
}
