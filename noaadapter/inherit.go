package noaadapter

import (
	"path/filepath"

	"github.com/Autumn-27/norma/noa"
)

// maxInheritDepth bounds the walk up the parent chain. Deep enough for any real
// nesting, shallow enough that a cycle or a mis-linked session cannot spin.
const maxInheritDepth = 8

// DeriveChildState hands a subagent the parent's compression ledger.
//
// What carries over and what does not follows from what each field means:
//
//   - Blocks, MessageRefs, TokenSnapshot, NextBlockID are the ledger. Without
//     them a child that sees inherited context could not address it, and would
//     allocate block ids that collide with the parent's.
//   - Nudge is reset. Pacing describes one conversation's pressure history; the
//     child starts its own, and inheriting a mid-session cadence would either
//     silence it or make it nag immediately.
//   - Stats is reset so the child reports its own work.
//
// Inherited blocks keep their original ArchivePath. It is absolute and the file
// is still there, so the child can follow a compression back to its originals
// exactly as the parent could — while its own new archives land in its own
// directory.
func DeriveChildState(parent noa.CompressionState, childSessionID, childArchiveRoot string) noa.CompressionState {
	child := noa.CloneState(parent)
	child.SessionID = childSessionID
	child.ArchiveRoot = childArchiveRoot
	child.Nudge = noa.NudgeState{LastShownByTier: map[noa.Tier]int{}}
	child.Stats = noa.Stats{}
	return child
}

// SubagentArchiveRoot is where a subagent's own archives live.
func SubagentArchiveRoot(base, sessionID, agentID string) string {
	return filepath.Join(base, sessionID, "subagents", agentID)
}

// InheritOptions describes one link in a parent chain.
type InheritOptions struct {
	// ArchiveBaseDir is shared by every session in the chain.
	ArchiveBaseDir string
	// SessionID is the session being opened.
	SessionID string
	// ParentIDs are candidate ancestors, nearest first. The walk stops at the
	// first one with blocks.
	ParentIDs []string
}

// InheritState finds a usable ledger for a session that has none of its own.
//
// A session can arrive empty for ordinary reasons — a fork, a subagent, a
// transcript copied without its sidecar — and starting from zero would make the
// model re-compress everything the ancestor already compressed. So the chain is
// walked until a state with blocks turns up.
//
// Returns ok=false when nothing was found, which is the normal case for a
// genuinely new session.
func InheritState(o InheritOptions) (noa.CompressionState, string, bool) {
	root := filepath.Join(o.ArchiveBaseDir, o.SessionID)
	for i, parentID := range o.ParentIDs {
		if i >= maxInheritDepth {
			break
		}
		if parentID == "" || parentID == o.SessionID {
			continue
		}
		parentRoot := filepath.Join(o.ArchiveBaseDir, parentID)
		store := NewStateStore(parentRoot)
		if !store.Exists() {
			continue
		}
		st, err := store.Load(parentID, parentRoot)
		if err != nil || len(st.Blocks) == 0 {
			continue
		}
		return DeriveChildState(st, o.SessionID, root), parentID, true
	}
	return noa.CreateInitialState(o.SessionID, root), "", false
}
