package noa

import (
	"fmt"
	"sort"
	"strings"
)

// Truncation defaults. These bound how much of a large tool result survives;
// head and tail together usually carry the command and its verdict, which is
// what a later reader actually needs.
const (
	truncateMinOutputTokens = 1000
	truncateKeepPrefixChars = 2000
	truncateKeepSuffixChars = 2000
)

// TruncateResult reports what the valve did.
type TruncateResult struct {
	Messages       []CoreMessage
	TruncatedCount int
	SavedTokens    int
}

// TruncateLargeToolOutputs is the mechanical last resort.
//
// Everything else in noa depends on the model choosing to compress. This does
// not: when the context is about to overflow, the largest tool results are cut
// down regardless. It is the only thing standing between an uncooperative model
// and a request the provider rejects outright.
//
// It rewrites only what is SENT. The stored history keeps the originals, so the
// next projection sees full text again, message identity is unchanged, and the
// truncation lifts by itself once usage falls back.
func TruncateLargeToolOutputs(msgs []CoreMessage, tokenCount int, cfg Config, count TokenCountFn) TruncateResult {
	if count == nil {
		count = DefaultCountTokens
	}
	res := TruncateResult{Messages: msgs}
	limit := cfg.ModelContextLimit
	if limit <= 0 || cfg.Truncate.Threshold <= 0 {
		return res
	}
	threshold := int(cfg.Truncate.Threshold * float64(limit))
	if tokenCount < threshold {
		return res
	}
	// Aim below the threshold, not at it: landing exactly on the line would
	// trigger again next turn for a handful of tokens.
	target := int(float64(threshold) * 0.9)

	protectRecent := max(cfg.PreserveRecentMessages, 0)
	cutoff := len(msgs) - protectRecent

	type candidate struct {
		idx    int
		tokens int
	}
	var candidates []candidate
	for i, m := range msgs {
		if i >= cutoff {
			break
		}
		if m.ContentType != CTToolResult || m.Text == "" {
			continue
		}
		if strings.Contains(m.Text, TruncationMarker) {
			continue // already truncated; cutting it again reclaims nothing
		}
		n := count(m.Text)
		if n < truncateMinOutputTokens {
			continue
		}
		candidates = append(candidates, candidate{idx: i, tokens: n})
	}
	if len(candidates) == 0 {
		return res
	}
	// Largest first: the fewest edits that get under the target.
	sort.SliceStable(candidates, func(a, b int) bool {
		return candidates[a].tokens > candidates[b].tokens
	})

	out := append([]CoreMessage(nil), msgs...)
	saved := 0
	for _, c := range candidates {
		if tokenCount-saved <= target {
			break
		}
		orig := out[c.idx].Text
		if len([]rune(orig)) <= truncateKeepPrefixChars+truncateKeepSuffixChars {
			continue
		}
		replaced := truncateBody(orig, c.tokens)
		saved += c.tokens - count(replaced)
		out[c.idx].Text = replaced
		res.TruncatedCount++
	}
	if res.TruncatedCount == 0 {
		return res
	}
	res.Messages = out
	res.SavedTokens = saved
	return res
}

// truncateBody keeps the head and tail around a marker naming the original size.
func truncateBody(s string, tokens int) string {
	r := []rune(s)
	head := string(r[:truncateKeepPrefixChars])
	tail := string(r[len(r)-truncateKeepSuffixChars:])
	return fmt.Sprintf("%s\n\n...%s — original ~%d tokens]...\n\n%s",
		head, TruncationMarker, tokens, tail)
}

// emergencyTruncateNode is the pipeline stage wrapper.
func emergencyTruncateNode() PipelineNode {
	return nodeFunc{
		name: "emergency-truncate",
		enabled: func(_ NodeIO, ctx PipelineContext) bool {
			limit := ctx.Config.ModelContextLimit
			return limit > 0 && ctx.TokenCount >= int(ctx.Config.Truncate.Threshold*float64(limit))
		},
		run: func(io NodeIO, ctx PipelineContext) NodeIO {
			r := TruncateLargeToolOutputs(io.Messages, ctx.TokenCount, ctx.Config, ctx.CountTokens)
			io.Messages = r.Messages
			io.TruncatedCount = r.TruncatedCount
			return io
		},
	}
}
