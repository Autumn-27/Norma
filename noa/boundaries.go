package noa

import "fmt"

// BoundaryKind distinguishes what a resolved range's endpoints referred to.
type BoundaryKind string

const (
	// BoundaryMessage means both endpoints were message refs: raw conversation
	// is being compressed, producing a tier-1 block.
	BoundaryMessage BoundaryKind = "message"
	// BoundaryBlock means at least one endpoint was a block id: existing blocks
	// are being consolidated, producing a higher tier. Such a range is exempt
	// from the per-batch character floor, since summaries are short by design.
	BoundaryBlock BoundaryKind = "block"
)

// boundaryRef is a parsed endpoint.
type boundaryRef struct {
	kind BoundaryKind
	// raw is the normalised form: "m00012" or "b5".
	raw string
	// num is the numeric part, used to fall back to a padded lookup.
	num int
}

// ParseBoundary classifies one endpoint of a requested range. Message refs and
// block ids share the field, so the leading letter is what tells them apart —
// without it "175" could mean either, and the whole tier mechanism would have
// nothing to branch on.
func ParseBoundary(s string) (boundaryRef, bool) {
	if n, ok := RefToIndex(s); ok {
		return boundaryRef{kind: BoundaryMessage, raw: IndexToRef(n), num: n}, true
	}
	if id, ok := ParseBlockID(s); ok {
		var n int
		fmt.Sscanf(id, "b%d", &n)
		return boundaryRef{kind: BoundaryBlock, raw: id, num: n}, true
	}
	return boundaryRef{}, false
}

// BoundaryNotFoundKind explains why an endpoint could not be resolved. The
// distinction drives which corrective text the model is given.
type BoundaryNotFoundKind string

const (
	// NotFoundUnknown means the ref belongs to no message in this session: a
	// typo, or a ref carried over from a different session generation. Refs are
	// never reassigned, so a compression in this session cannot have caused it.
	NotFoundUnknown BoundaryNotFoundKind = "unknown"
	// NotFoundConsumed means the ref is known but its content is already folded
	// into a block — the model is trying to compress something twice.
	NotFoundConsumed BoundaryNotFoundKind = "consumed"
	// NotFoundInvalid means the string did not parse as a ref or block id.
	NotFoundInvalid BoundaryNotFoundKind = "invalid"
)

// BoundaryNotFoundError carries the failed endpoint and why it failed.
type BoundaryNotFoundError struct {
	Ref  string
	Kind BoundaryNotFoundKind
	Msg  string
}

func (e *BoundaryNotFoundError) Error() string { return e.Msg }

// ResolvedRange is a request's endpoints mapped onto the current view.
type ResolvedRange struct {
	StartIndex int
	EndIndex   int
	Kind       BoundaryKind
	// MessageIDs are the ids inside [StartIndex, EndIndex], with rendered
	// summaries excluded (they are projections, not content).
	MessageIDs []string
	// NestedBlockIDs are the active blocks whose anchor falls inside the span.
	NestedBlockIDs []string
	// Warnings records endpoints that were snapped to a containing block.
	Warnings []string
}

// resolveContext bundles what boundary resolution needs to look things up.
type resolveContext struct {
	messages []CoreMessage
	state    CompressionState
	// indexOf maps a host message id to its position in messages.
	indexOf map[string]int
	// blockAnchor maps a block id to the position its summary occupies.
	blockAnchor map[string]int
}

func newResolveContext(messages []CoreMessage, state CompressionState) *resolveContext {
	rc := &resolveContext{
		messages:    messages,
		state:       state,
		indexOf:     make(map[string]int, len(messages)),
		blockAnchor: map[string]int{},
	}
	for i, m := range messages {
		if m.ID == "" {
			continue
		}
		if _, dup := rc.indexOf[m.ID]; !dup {
			rc.indexOf[m.ID] = i
		}
	}
	for _, b := range state.Blocks {
		if !b.Active {
			continue
		}
		if i, ok := rc.indexOf[SummaryMessageID(b.BlockID)]; ok {
			rc.blockAnchor[b.BlockID] = i
			continue
		}
		earliest := -1
		for _, id := range b.EffectiveMessageIDs {
			if i, ok := rc.indexOf[id]; ok && (earliest < 0 || i < earliest) {
				earliest = i
			}
		}
		if earliest >= 0 {
			rc.blockAnchor[b.BlockID] = earliest
		}
	}
	return rc
}

// activeOwnerAnchor finds the visible anchor of an active block that absorbed a
// child covering rawID. A ref that points into an absorbed child is not stale —
// the content is still represented, just one tier up — so the endpoint snaps to
// the owner rather than failing.
func (rc *resolveContext) activeOwnerAnchor(rawID string) (int, string, bool) {
	for _, owner := range rc.state.Blocks {
		if !owner.Active || len(owner.DirectBlockIDs) == 0 {
			continue
		}
		for _, childID := range owner.DirectBlockIDs {
			child := FindBlock(&rc.state, childID)
			if child == nil {
				continue
			}
			for _, id := range child.EffectiveMessageIDs {
				if id != rawID {
					continue
				}
				if at, ok := rc.blockAnchor[owner.BlockID]; ok {
					return at, owner.BlockID, true
				}
			}
		}
	}
	return 0, "", false
}

// resolveAnchorIndex maps one endpoint onto a position in the current view.
func (rc *resolveContext) resolveAnchorIndex(ref boundaryRef) (int, string, *BoundaryNotFoundError) {
	switch ref.kind {
	case BoundaryMessage:
		rawID, ok := rc.state.MessageRefs.ByRef[ref.raw]
		if !ok {
			rawID, ok = rc.state.MessageRefs.ByRef[IndexToRef(ref.num)]
		}
		if !ok {
			return 0, "", &BoundaryNotFoundError{Ref: ref.raw, Kind: NotFoundUnknown,
				Msg: fmt.Sprintf("%s does not exist in this session", ref.raw)}
		}
		if i, ok := rc.indexOf[rawID]; ok {
			return i, "", nil
		}
		if at, owner, ok := rc.activeOwnerAnchor(rawID); ok {
			return at, fmt.Sprintf("Snapped %s to block %s, which absorbed it.", ref.raw, owner), nil
		}
		return 0, "", &BoundaryNotFoundError{Ref: ref.raw, Kind: NotFoundConsumed,
			Msg: fmt.Sprintf("%s is already covered by an active block", ref.raw)}

	case BoundaryBlock:
		b := FindBlock(&rc.state, ref.raw)
		if b == nil {
			return 0, "", &BoundaryNotFoundError{Ref: ref.raw, Kind: NotFoundUnknown,
				Msg: fmt.Sprintf("block %s does not exist in this session", ref.raw)}
		}
		if b.Active {
			if at, ok := rc.blockAnchor[ref.raw]; ok {
				return at, "", nil
			}
			return 0, "", &BoundaryNotFoundError{Ref: ref.raw, Kind: NotFoundConsumed,
				Msg: fmt.Sprintf("block %s is active but has no visible content to compress", ref.raw)}
		}
		// Inactive: it may have been absorbed, in which case the owner stands in.
		for _, id := range b.EffectiveMessageIDs {
			if at, owner, ok := rc.activeOwnerAnchor(id); ok {
				return at, fmt.Sprintf("Snapped %s to block %s, which absorbed it.", ref.raw, owner), nil
			}
		}
		return 0, "", &BoundaryNotFoundError{Ref: ref.raw, Kind: NotFoundConsumed,
			Msg: fmt.Sprintf("block %s was already consumed by a higher-tier block", ref.raw)}
	}
	return 0, "", &BoundaryNotFoundError{Ref: ref.raw, Kind: NotFoundInvalid,
		Msg: fmt.Sprintf("%q is neither a message ref (mNNNNN) nor a block id (bN)", ref.raw)}
}

// ResolveBoundaries maps a requested range onto the current view.
//
// Reversed endpoints are swapped rather than rejected: the model meant a span,
// and the order it wrote it in carries no information.
func ResolveBoundaries(startRef, endRef string, messages []CoreMessage, state CompressionState) (ResolvedRange, *BoundaryNotFoundError) {
	start, ok := ParseBoundary(startRef)
	if !ok {
		return ResolvedRange{}, &BoundaryNotFoundError{Ref: startRef, Kind: NotFoundInvalid,
			Msg: fmt.Sprintf("%q is neither a message ref (mNNNNN) nor a block id (bN)", startRef)}
	}
	end, ok := ParseBoundary(endRef)
	if !ok {
		return ResolvedRange{}, &BoundaryNotFoundError{Ref: endRef, Kind: NotFoundInvalid,
			Msg: fmt.Sprintf("%q is neither a message ref (mNNNNN) nor a block id (bN)", endRef)}
	}

	rc := newResolveContext(messages, state)
	var warnings []string

	si, w1, err := rc.resolveAnchorIndex(start)
	if err != nil {
		return ResolvedRange{}, err
	}
	if w1 != "" {
		warnings = append(warnings, w1)
	}
	ei, w2, err := rc.resolveAnchorIndex(end)
	if err != nil {
		return ResolvedRange{}, err
	}
	if w2 != "" {
		warnings = append(warnings, w2)
	}
	if si > ei {
		si, ei = ei, si
	}

	kind := BoundaryMessage
	if start.kind == BoundaryBlock || end.kind == BoundaryBlock {
		kind = BoundaryBlock
	}

	out := ResolvedRange{StartIndex: si, EndIndex: ei, Kind: kind, Warnings: warnings}
	for i := si; i <= ei && i < len(messages); i++ {
		m := messages[i]
		if m.ID == "" || isRenderedSummaryMessage(m) {
			continue
		}
		out.MessageIDs = append(out.MessageIDs, m.ID)
	}
	out.NestedBlockIDs = nestedActiveBlocks(rc, si, ei)
	return out, nil
}

// nestedActiveBlocks lists the active blocks anchored inside [si, ei].
func nestedActiveBlocks(rc *resolveContext, si, ei int) []string {
	var out []string
	for _, b := range rc.state.Blocks {
		if !b.Active {
			continue
		}
		if at, ok := rc.blockAnchor[b.BlockID]; ok && at >= si && at <= ei {
			out = append(out, b.BlockID)
		}
	}
	return out
}
