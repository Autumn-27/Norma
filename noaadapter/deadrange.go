package noaadapter

import (
	"fmt"
	"sort"
	"strings"

	"github.com/Autumn-27/norma/noa"
)

// DeadRangeTracker breaks the loop where a model keeps asking to compress refs
// that cannot resolve.
//
// A small model handed a stale ref will often resubmit the identical batch: the
// rejection tells it the range is dead, but not that repeating it is futile.
// Each retry costs a full turn, and the failure ladder alone is slow to bite
// because a parse-clean call that reclaims nothing still looks like progress.
//
// So the second sighting of the same range set is refused outright, with the
// live ranges named — a rejection the model can act on instead of repeat.
type DeadRangeTracker struct {
	// seen counts how many times each range-set signature has failed.
	seen map[string]int
}

// NewDeadRangeTracker returns an empty tracker.
func NewDeadRangeTracker() *DeadRangeTracker {
	return &DeadRangeTracker{seen: map[string]int{}}
}

// signature identifies a batch by its ranges, order-independent: the model may
// reshuffle a retry without changing what it is asking for.
func signature(ranges []noa.CompressRange) string {
	parts := make([]string, 0, len(ranges))
	for _, r := range ranges {
		parts = append(parts, strings.ToLower(strings.TrimSpace(r.StartRef))+".."+
			strings.ToLower(strings.TrimSpace(r.EndRef)))
	}
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

// Check reports whether this batch has already failed often enough to refuse.
// The returned string is the refusal to show the model; "" means proceed.
func (d *DeadRangeTracker) Check(ranges []noa.CompressRange, state noa.CompressionState) string {
	if len(ranges) == 0 {
		return ""
	}
	if d.seen[signature(ranges)] < noa.DeadRepeatReject {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "This exact range set has already failed %d times and will fail again — "+
		"the refs do not resolve against the current context. Do not resubmit it.",
		noa.DeadRepeatReject)

	// Naming what IS live turns a dead end into a next step.
	if spans := noa.ActiveBlockSpans(state); len(spans) > 0 {
		var live []string
		for _, s := range spans {
			live = append(live, fmt.Sprintf("%s(T%d)=%s–%s", s.BlockID, s.Tier, s.StartRef, s.EndRef))
		}
		fmt.Fprintf(&b, "\nLive blocks you can target instead: %s.", strings.Join(live, ", "))
	}
	b.WriteString("\nOtherwise use the refs shown in the <noa-ref> tags of the current context.")
	return b.String()
}

// Record notes that a batch failed.
func (d *DeadRangeTracker) Record(ranges []noa.CompressRange) {
	if len(ranges) == 0 {
		return
	}
	d.seen[signature(ranges)]++
}

// Reset clears the tracker after a successful compression: the view changed, so
// what was dead before may not be now.
func (d *DeadRangeTracker) Reset() {
	d.seen = map[string]int{}
}
