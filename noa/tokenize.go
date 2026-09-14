package noa

import "math"

// TokenCountFn estimates a text's token cost. Injected so the host can supply a
// real tokenizer; DefaultCountTokens is the built-in approximation.
type TokenCountFn func(string) int

// DefaultCountTokens approximates tokens as one per CJK rune plus one per four
// other runes.
//
// Deviation from upstream (deliberate): counting is by RUNE, not UTF-16 code
// unit. Upstream mirrors JavaScript's String.length so its numbers match the
// reference implementation byte for byte; noa has no such requirement, and rune
// semantics are the natural Go reading. The only visible difference is that
// astral-plane characters (emoji) count once instead of twice.
func DefaultCountTokens(text string) int {
	if text == "" {
		return 0
	}
	cjk, other := 0, 0
	for _, r := range text {
		if isCJK(r) {
			cjk++
		} else {
			other++
		}
	}
	return cjk + ceilDiv(other, 4)
}

// EstimateTokensFast skips the CJK scan; used where a rough figure suffices.
func EstimateTokensFast(text string) int { return ceilDiv(len([]rune(text)), 4) }

// CountMessageTokens is the per-message cost under a given counter.
func CountMessageTokens(m CoreMessage, count TokenCountFn) int {
	if count == nil {
		count = DefaultCountTokens
	}
	return count(m.Text)
}

func isCJK(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF: // CJK Unified Ideographs
		return true
	case r >= 0x3040 && r <= 0x30FF: // Hiragana + Katakana
		return true
	case r >= 0xAC00 && r <= 0xD7AF: // Hangul Syllables
		return true
	}
	return false
}

func ceilDiv(n, d int) int {
	if n <= 0 {
		return 0
	}
	return (n + d - 1) / d
}

// roundHalfUp matches JavaScript's Math.round for the non-negative values used
// here (Go's math.Round is half-away-from-zero, which agrees on positives).
func roundHalfUp(f float64) int { return int(math.Round(f)) }
