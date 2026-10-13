package llm

import "testing"

// Real Anthropic reports the input side in both message_start and message_delta;
// the accumulator must not double it.
func TestUsageMergeEchoRealAnthropicShape(t *testing.T) {
	var u Usage
	u.Add(Usage{InputTokens: 120, CacheReadTokens: 80}) // message_start
	u.MergeEcho(Usage{InputTokens: 120, CacheReadTokens: 80, OutputTokens: 34})
	if u.InputTokens != 120 || u.CacheReadTokens != 80 {
		t.Fatalf("echoed input double-counted: %+v", u)
	}
	if u.OutputTokens != 34 {
		t.Fatalf("output lost: %+v", u)
	}
}

// Anthropic-compatible emulations (observed: Z.ai's coding-plan endpoint) send
// zeros in message_start and the real input counts only in message_delta; the
// accumulator must recover them instead of dropping the input side.
func TestUsageMergeEchoZeroStartRecoveredFromDelta(t *testing.T) {
	var u Usage
	u.Add(Usage{InputTokens: 0}) // message_start: zeros
	u.MergeEcho(Usage{InputTokens: 120, CacheReadTokens: 0, OutputTokens: 34})
	if u.InputTokens != 120 {
		t.Fatalf("input not recovered from delta: %+v", u)
	}
	if u.OutputTokens != 34 {
		t.Fatalf("output lost: %+v", u)
	}
}

// The anthropic frame parser must put the delta's input side on the
// SEMessageDelta event (previously it carried output only).
func TestParseAnthropicFrameDeltaCarriesInput(t *testing.T) {
	ev, ok, err := parseAnthropicFrame(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":987,"output_tokens":42,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}`)
	if err != nil || !ok || ev.Type != SEMessageDelta {
		t.Fatalf("frame parse: ok=%v err=%v type=%v", ok, err, ev.Type)
	}
	if ev.Usage.InputTokens != 987 || ev.Usage.OutputTokens != 42 {
		t.Fatalf("delta usage not carried: %+v", ev.Usage)
	}

	// end-to-end through the accumulator: start zeros + delta real
	start, _, _ := parseAnthropicFrame(`{"type":"message_start","message":{"usage":{"input_tokens":0,"output_tokens":0}}}`)
	acc := NewAccumulator()
	acc.Add(start)
	acc.Add(ev)
	u := acc.Usage
	if u.InputTokens != 987 || u.OutputTokens != 42 {
		t.Fatalf("accumulator: input not recovered: %+v", u)
	}
}

// ...and with a real-Anthropic stream the input must not double.
func TestAccumulatorRealAnthropicStreamNoDoubleCount(t *testing.T) {
	start, _, _ := parseAnthropicFrame(`{"type":"message_start","message":{"usage":{"input_tokens":500,"cache_read_input_tokens":200,"output_tokens":1}}}`)
	delta, _, _ := parseAnthropicFrame(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":500,"cache_read_input_tokens":200,"output_tokens":77}}`)
	acc := NewAccumulator()
	acc.Add(start)
	acc.Add(delta)
	u := acc.Usage
	if u.InputTokens != 700 || u.CacheReadTokens != 200 || u.OutputTokens != 77 {
		t.Fatalf("real-Anthropic shape miscounted: %+v", u)
	}
}

// A FULL echo cannot prove the merge works: the delta alone already produces the
// expected totals, so dropping message_start entirely would leave the test
// above green. This feeds a PARTIAL echo — the delta repeats input_tokens but
// omits the cache fields, a shape Anthropic has used — so the cache counts
// survive only if message_start is folded in and only if the fold takes the max
// per field rather than the latest value.
func TestAccumulatorPartialEchoKeepsStartCacheCounts(t *testing.T) {
	start, _, _ := parseAnthropicFrame(`{"type":"message_start","message":{"usage":{"input_tokens":500,"cache_read_input_tokens":200,"cache_creation_input_tokens":40,"output_tokens":1}}}`)
	delta, _, _ := parseAnthropicFrame(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":500,"output_tokens":77}}`)

	acc := NewAccumulator()
	acc.Add(start)
	acc.Add(delta)
	u := acc.Usage

	// message_start normalizes to Input = 500+200+40 = 740; the delta's partial
	// echo normalizes to Input = 500. Max keeps 740 — summing would give 1240.
	if u.InputTokens != 740 {
		t.Errorf("InputTokens = %d, want 740 (max of the echoed totals, not a sum)", u.InputTokens)
	}
	if u.CacheReadTokens != 200 {
		t.Errorf("CacheReadTokens = %d, want 200 — the delta omitted it, so it must come from message_start", u.CacheReadTokens)
	}
	if u.CacheWriteTokens != 40 {
		t.Errorf("CacheWriteTokens = %d, want 40 — same, recovered from message_start", u.CacheWriteTokens)
	}
	if u.OutputTokens != 77 {
		t.Errorf("OutputTokens = %d, want 77 (message_start's placeholder dropped, delta's accumulated)", u.OutputTokens)
	}
}
