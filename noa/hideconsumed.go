package noa

import (
	"encoding/json"
	"strings"
)

// hideCompressCalls trims the historical Compress calls still sitting in the
// view.
//
// A Compress call carries its full summary text in its arguments. Once the
// block is gone that text is pure waste; while the block is alive the text is
// still redundant, because the summary is already rendered in context AND
// stored in the archive. So: drop the calls whose blocks are gone, and stub the
// arguments of the ones that remain.
//
// The calls of live blocks are NOT dropped: prune exempts Compress from orphan
// stripping precisely because that call is the summary's addressable anchor in
// the conversation flow.
func hideCompressCalls(io NodeIO, _ PipelineContext) NodeIO {
	allBlockCallIDs := map[string]bool{}
	activeCallIDs := map[string]bool{}
	for _, b := range io.State.Blocks {
		if b.CompressCallID == "" {
			continue
		}
		allBlockCallIDs[b.CompressCallID] = true
		if b.Active {
			activeCallIDs[b.CompressCallID] = true
		}
	}

	// A call with no block yet is one that just failed. Keeping the last couple
	// lets the model see its own mistake and correct it; keeping more would
	// accumulate noise.
	orphanKeep := map[string]bool{}
	kept := 0
	for i := len(io.Messages) - 1; i >= 0 && kept < KeepLastOrphaned; i-- {
		m := io.Messages[i]
		if m.ContentType == CTToolCall && m.ToolName == CompressToolName &&
			m.ToolCallID != "" && !allBlockCallIDs[m.ToolCallID] {
			orphanKeep[m.ToolCallID] = true
			kept++
		}
	}

	keep := func(callID string) bool {
		return activeCallIDs[callID] || orphanKeep[callID]
	}

	// Identify the exchanges to remove wholesale.
	dropCallIDs := map[string]bool{}
	for _, m := range io.Messages {
		if m.ContentType == CTToolCall && m.ToolName == CompressToolName &&
			m.ToolCallID != "" && !keep(m.ToolCallID) {
			dropCallIDs[m.ToolCallID] = true
		}
	}

	out := make([]CoreMessage, 0, len(io.Messages))
	for _, m := range io.Messages {
		if m.ToolCallID != "" && dropCallIDs[m.ToolCallID] {
			continue // the call and its result go together
		}
		if m.ContentType == CTToolCall && m.ToolName == CompressToolName && m.ToolCallID != "" {
			m.Text = compactCompressText(m.Text)
		}
		out = append(out, m)
	}
	io.Messages = out
	return io
}

// compactCompressText stubs each summary inside a Compress call's arguments.
//
// Parsing is best-effort: anything it cannot understand is returned untouched.
// Mangling a call's arguments would be worse than leaving them long.
func compactCompressText(text string) string {
	brace := strings.Index(text, "{")
	if brace < 0 {
		return text
	}
	// Some providers prefix the arguments with prose; preserve it verbatim.
	prefix, body := text[:brace], text[brace:]

	var obj map[string]any
	if json.Unmarshal([]byte(body), &obj) != nil {
		return text
	}
	raw, ok := obj["content"]
	if !ok {
		return text
	}

	// content may be an array or a JSON-encoded string of one; it has to go back
	// in whichever form it arrived, or the shape changes under the model.
	contentWasString := false
	var arr []any
	switch v := raw.(type) {
	case []any:
		arr = v
	case string:
		if json.Unmarshal([]byte(v), &arr) != nil {
			return text
		}
		contentWasString = true
	default:
		return text
	}

	changed := false
	for _, it := range arr {
		m, ok := it.(map[string]any)
		if !ok {
			continue
		}
		s, ok := m["summary"].(string)
		if !ok {
			continue
		}
		if stub, cut := stubSummary(s); cut {
			m["summary"] = stub
			changed = true
		}
	}
	if !changed {
		return text
	}

	if contentWasString {
		encoded, err := json.Marshal(arr)
		if err != nil {
			return text
		}
		obj["content"] = string(encoded)
	} else {
		obj["content"] = arr
	}
	rebuilt, err := json.Marshal(obj)
	if err != nil {
		return text
	}
	return prefix + string(rebuilt)
}

// stubSummary shortens one summary, reporting whether it changed.
func stubSummary(s string) (string, bool) {
	r := []rune(s)
	if len(r) <= SummaryStubChars {
		return s, false
	}
	return string(r[:SummaryStubChars-1]) + "…", true
}

// hideCompressCallsNode is the pipeline stage wrapper.
func hideCompressCallsNode() PipelineNode {
	return nodeFunc{
		name: "hide-compress-calls",
		enabled: func(io NodeIO, _ PipelineContext) bool {
			for _, m := range io.Messages {
				if m.ContentType == CTToolCall && m.ToolName == CompressToolName {
					return true
				}
			}
			return false
		},
		run: hideCompressCalls,
	}
}
