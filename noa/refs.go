package noa

import (
	"fmt"
	"maps"
	"regexp"
	"strconv"
	"strings"
)

var refPattern = regexp.MustCompile(`^m0*(\d{1,5})$`)

// IndexToRef renders a 1-based index as mNNNNN. Out-of-range input is a
// programming error and yields "".
func IndexToRef(index int) string {
	if index < MinRefIndex || index > MaxRefIndex {
		return ""
	}
	return fmt.Sprintf("m%0*d", RefWidth, index)
}

// RefToIndex parses mNNNNN back to its index, tolerating surrounding space,
// upper case, and missing zero padding (m5 == m00005). Returns ok=false for
// anything else.
func RefToIndex(ref string) (int, bool) {
	m := refPattern.FindStringSubmatch(strings.ToLower(strings.TrimSpace(ref)))
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	if err != nil || n < MinRefIndex || n > MaxRefIndex {
		return 0, false
	}
	return n, true
}

// HighestUsedIndex is the largest ref index handed out so far, skipping BLOCKED
// entries (which occupy no number).
func HighestUsedIndex(m MessageRefMap) int {
	highest := 0
	for _, ref := range m.ByRaw {
		if ref == BlockedRef {
			continue
		}
		if i, ok := RefToIndex(ref); ok && i > highest {
			highest = i
		}
	}
	return highest
}

// AssignRefsOptions parameterises a pass over the messages.
type AssignRefsOptions struct {
	// Existing is the map to extend. It is copied, not mutated.
	Existing MessageRefMap
	// NextIndex is where allocation resumes; below MinRefIndex it is clamped.
	NextIndex int
	// IsProtected marks a message as hard-protected: it is recorded as BLOCKED
	// and never becomes addressable.
	IsProtected func(CoreMessage) bool
	// ShouldSkip leaves a message out of the map entirely.
	ShouldSkip func(CoreMessage) bool
}

// AssignRefsResult carries the extended map.
type AssignRefsResult struct {
	Map           MessageRefMap
	NextIndex     int
	NewlyAssigned int
}

// AssignRefs hands out a ref to every message that does not have one.
//
// CORE INVARIANT: a ref, once assigned, is never reassigned. A message already
// present in ByRaw is skipped outright, so no compression, prune, or rebuild
// can renumber it. Every ref the model has ever seen stays valid for the life
// of the session.
func AssignRefs(messages []CoreMessage, opts AssignRefsOptions) AssignRefsResult {
	out := MessageRefMap{
		ByRaw: make(map[string]string, len(opts.Existing.ByRaw)+len(messages)),
		ByRef: make(map[string]string, len(opts.Existing.ByRef)+len(messages)),
	}
	maps.Copy(out.ByRaw, opts.Existing.ByRaw)
	maps.Copy(out.ByRef, opts.Existing.ByRef)

	cursor := max(opts.NextIndex, MinRefIndex)
	assigned := 0

	for _, m := range messages {
		if m.ID == "" {
			continue
		}
		if opts.ShouldSkip != nil && opts.ShouldSkip(m) {
			continue
		}
		if _, seen := out.ByRaw[m.ID]; seen {
			continue // never reassign
		}
		if opts.IsProtected != nil && opts.IsProtected(m) {
			out.ByRaw[m.ID] = BlockedRef
			continue
		}
		ref, idx := allocateFreeRef(out, cursor)
		if ref == "" {
			break // ref space exhausted; remaining messages stay unaddressable
		}
		cursor = idx + 1
		out.ByRaw[m.ID] = ref
		out.ByRef[ref] = m.ID
		assigned++
	}
	return AssignRefsResult{Map: out, NextIndex: cursor, NewlyAssigned: assigned}
}

// allocateFreeRef finds the first unused ref at or after start.
func allocateFreeRef(m MessageRefMap, start int) (string, int) {
	start = max(start, MinRefIndex)
	for i := start; i <= MaxRefIndex; i++ {
		ref := IndexToRef(i)
		if _, taken := m.ByRef[ref]; !taken {
			return ref, i
		}
	}
	return "", 0
}

// RefForRaw returns the ref assigned to a host message id, if any.
func RefForRaw(m MessageRefMap, rawID string) (string, bool) {
	r, ok := m.ByRaw[rawID]
	return r, ok
}

// RawForRef resolves a ref back to its host message id.
func RawForRef(m MessageRefMap, ref string) (string, bool) {
	r, ok := m.ByRef[ref]
	return r, ok
}
