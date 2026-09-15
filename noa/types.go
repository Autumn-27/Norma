// Package noa is a model-driven context-compression engine: a pure core with no
// host dependency, no I/O, and no model calls.
//
// The essential difference from zip-style compression: the model writes the
// summaries, this package only orchestrates. Summary text arrives as an argument
// (the Compress tool's parameter); noa decides WHEN to compress, WHICH range to
// compress, tracks state, applies a compress decision, and prunes ranges. It
// never calls a model.
//
// That is precisely why the core can be a pure library — its one external
// dependency, the summarizer, is reduced to a string input.
//
// Ported from acp-kernel (MIT, Copyright © 2026 ranxianglei). See
// docs/noa-实施方案.md for the full specification, including every deliberate
// deviation from upstream.
package noa

// Role is the speaker of a CoreMessage.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentType is the kind of content a CoreMessage carries. One host message
// flattens into 0..N CoreMessages, one per content block.
type ContentType string

const (
	CTText       ContentType = "text"
	CTToolCall   ContentType = "tool-call"
	CTToolResult ContentType = "tool-result"
	CTReasoning  ContentType = "reasoning"
)

// CoreMessage is the flattened intermediate representation the pipeline works
// on. The host projects its own message type into these and reassembles them
// afterwards; noa never sees the host's types.
//
// Text is the bare body with any ref tag already stripped — the tag must not
// reach the identity hash, or the id would drift every time the tag is
// re-rendered.
type CoreMessage struct {
	ID          string
	Role        Role
	ContentType ContentType
	Text        string
	ToolName    string
	ToolCallID  string
}

// Tier is a block's compression level, 1..Config.Tiers.MaxTier. A block's tier
// is fixed at creation and never changes.
type Tier int

// CompressionBlock is one compressed range: the model-written summary that
// stands in for it, the messages it covers, and the archive file holding the
// originals.
type CompressionBlock struct {
	BlockID string
	Tier    Tier
	Topic   string
	Summary string

	// DirectMessageIDs are the messages this block newly covers (excluding those
	// already covered by an absorbed child block).
	DirectMessageIDs []string
	// EffectiveMessageIDs are every message this block covers, inherited children
	// included. This is what prune hides and what the block span reports.
	EffectiveMessageIDs []string
	// DirectBlockIDs are the lower-tier blocks this block absorbed.
	DirectBlockIDs []string

	// ArchivePath is the absolute path of the markdown file holding the original
	// content. ArchiveRel is the same path relative to the archive root, so a
	// moved session directory can be relocated.
	ArchivePath string
	ArchiveRel  string

	// CompressedTokens is how much the absorbed content cost, for reporting.
	CompressedTokens int
	// StartRef/EndRef span the block's ACTUAL coverage (derived from
	// EffectiveMessageIDs), not the refs the model requested — otherwise the
	// model's ledger drifts from reality.
	StartRef  string
	EndRef    string
	CreatedAt int64

	Active bool
	// CompressCallID is the tool_use id of the Compress call that created this
	// block, used by hide-compress-calls to decide which calls to keep.
	CompressCallID string
}

// MessageRefMap is the two-way index between host message ids and the mNNNNN
// refs the model addresses them by.
//
// Core invariant: a ref, once assigned, is NEVER reassigned. Refs are
// per-session snapshots handed out when a message is first rendered; no
// compression renumbers them.
type MessageRefMap struct {
	ByRaw map[string]string // host id -> "m00001", or BlockedRef
	ByRef map[string]string // "m00001" -> host id
}

// NudgeState is the bookkeeping that paces nudge injection.
type NudgeState struct {
	// LastPerMessageNudgeTokens is the growth baseline.
	LastPerMessageNudgeTokens int
	// LastNudgeShownTokens is the token count at the last injection.
	LastNudgeShownTokens int
	// LastShownByTier paces each tier independently.
	LastShownByTier map[Tier]int
}

// Stats are cumulative counters for reporting.
type Stats struct {
	TokensCompressed int
	CompressionCount int
}

// CompressionState is everything noa remembers about a session. It is plain
// data: serialisable, rebuildable from the transcript, and never holding a
// reference to the host.
type CompressionState struct {
	Blocks      []CompressionBlock
	MessageRefs MessageRefMap
	// TokenSnapshot freezes each ref's token count at first render. The number in
	// a ref tag must never drift: a changed tag is a changed byte is an
	// invalidated prefix cache.
	TokenSnapshot map[string]int
	Nudge         NudgeState
	Stats         Stats
	NextBlockID   int
	// ArchiveRoot is <ArchiveBaseDir>/<SessionID>, computed by the host adapter.
	ArchiveRoot string
	SessionID   string
}

// CreateInitialState returns the zero state for a fresh session.
func CreateInitialState(sessionID, archiveRoot string) CompressionState {
	return CompressionState{
		Blocks:        nil,
		MessageRefs:   MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}},
		TokenSnapshot: map[string]int{},
		Nudge:         NudgeState{LastShownByTier: map[Tier]int{}},
		NextBlockID:   1,
		ArchiveRoot:   archiveRoot,
		SessionID:     sessionID,
	}
}
