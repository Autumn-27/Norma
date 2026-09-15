package noa

import (
	"fmt"
	"regexp"
	"strings"
)

// FormatTokens renders a token count compactly: exact below 1000, one decimal
// below 10000, whole thousands above.
func FormatTokens(n int) string {
	switch {
	case n < 1000:
		return fmt.Sprintf("%d", n)
	case n < 10000:
		return fmt.Sprintf("%.1fK", float64(n)/1000)
	default:
		return fmt.Sprintf("%dK", roundHalfUp(float64(n)/1000))
	}
}

// PanelInput is what the Compress tool reports back to the model.
type PanelInput struct {
	Result ApplyResult
	// BeforeTokens and AfterTokens bracket the compression for the headline.
	BeforeTokens int
	AfterTokens  int
	// Messages is the post-compression view, used to mark partial coverage.
	Messages []CoreMessage
}

// FormatPanel renders the Compress tool's result.
//
// Every block line reports the span the block ACTUALLY covers, derived from its
// effective coverage rather than the refs the model asked for. Reporting the
// request instead would let the model's ledger drift from what really happened
// — it would believe it had compressed something it had not.
func FormatPanel(in PanelInput) string {
	var b strings.Builder
	r := in.Result

	if len(r.BlocksCreated) == 0 {
		b.WriteString("▣ noa | 0 blocks created")
		if len(r.Errors) > 0 {
			b.WriteString("\n")
			for _, e := range r.Errors {
				fmt.Fprintf(&b, "  error: %s\n", e)
			}
		}
		for _, w := range r.Warnings {
			fmt.Fprintf(&b, "  warn: %s\n", w)
		}
		return strings.TrimRight(b.String(), "\n")
	}

	reclaimed := in.BeforeTokens - in.AfterTokens
	fmt.Fprintf(&b, "▣ noa | %s → %s tokens (~%s reclaimed)\n",
		FormatTokens(in.BeforeTokens), FormatTokens(in.AfterTokens), FormatTokens(max(reclaimed, 0)))

	covered := map[string]bool{}
	for _, blk := range r.BlocksCreated {
		for _, id := range blk.EffectiveMessageIDs {
			covered[id] = true
		}
	}
	for _, blk := range r.BlocksCreated {
		star := ""
		if spanHasUncovered(blk, r.State, covered) {
			// The span contains refs this block did not take — usually the
			// protected zone or a higher-tier block. Flagging it stops the model
			// assuming the whole span is gone.
			star = "*"
		}
		fmt.Fprintf(&b, "  %s(T%d)=%s–%s%s → %s\n",
			blk.BlockID, blk.Tier, blk.StartRef, blk.EndRef, star, blk.ArchivePath)
	}

	if len(r.SurvivingBlockIDs) > 0 {
		var parts []string
		for _, id := range r.SurvivingBlockIDs {
			if blk := FindBlock(&r.State, id); blk != nil {
				parts = append(parts, fmt.Sprintf("%s(T%d)", id, blk.Tier))
			} else {
				parts = append(parts, id)
			}
		}
		fmt.Fprintf(&b, "  note: %s was inside the requested range but was not consumed\n", strings.Join(parts, ", "))
		b.WriteString("        (only the lowest tier present is absorbed). Compress that layer separately.\n")
	}
	for _, e := range r.Errors {
		fmt.Fprintf(&b, "  error: %s\n", e)
	}
	for _, w := range r.Warnings {
		fmt.Fprintf(&b, "  warn: %s\n", w)
	}
	return strings.TrimRight(b.String(), "\n")
}

// spanHasUncovered reports whether the block's ref span contains a ref that is
// still live and not covered by this batch.
func spanHasUncovered(blk CompressionBlock, state CompressionState, covered map[string]bool) bool {
	lo, okLo := RefToIndex(blk.StartRef)
	hi, okHi := RefToIndex(blk.EndRef)
	if !okLo || !okHi {
		return false
	}
	for ref, rawID := range state.MessageRefs.ByRef {
		n, ok := RefToIndex(ref)
		if !ok || n < lo || n > hi {
			continue
		}
		if !covered[rawID] {
			return true
		}
	}
	return false
}

// panelBlockRE counts the block lines a panel reported. The tier label carries
// its own parentheses, so the pattern anchors on the id and the "=" rather than
// scanning to a closing paren.
var panelBlockRE = regexp.MustCompile(`\bb\d+\(T\d+\)=`)

// PanelBlockCount reports how many blocks a rendered panel describes.
//
// The failure counter needs this: a panel is not an error, but a panel with no
// blocks means nothing was reclaimed, and that has to count as a failed attempt
// or a model can loop forever on calls that are technically successful.
func PanelBlockCount(panel string) int {
	return len(panelBlockRE.FindAllString(panel, -1))
}

// NoRangesMessage is the neutral response to a call with nothing to do. It
// counts as neither success nor failure: the model asked a valid question and
// got a valid answer.
const NoRangesMessage = "No ranges provided."
