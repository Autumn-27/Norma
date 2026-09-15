package noaadapter

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// newCompressTool builds the one tool noa exposes.
func newCompressTool(sess *Session) tool.CoreTool {
	return tool.Build(tool.Spec{
		Name:        noa.CompressToolName,
		Description: noa.CompressToolDescription,
		Schema:      noa.CompressToolSchema(),
		// The model's arguments reach runCompress exactly as they arrived.
		//
		// ParseCompressArgs exists to recover ranges from a call a provider
		// mangled — truncated mid-object, wrapped in a code fence, carrying a
		// trailing comma, or with the array stringified. None of those is valid
		// JSON, so the harness's schema check would reject them first and the
		// recovery would never run; the model would get "invalid tool input"
		// instead of the shape guidance formatParseError writes, and the turn's
		// one compression attempt would be spent on a validation error.
		//
		// The schema is still advertised (Schema above), so the model knows what
		// to aim for. This only removes the gate, not the instruction.
		RawInput: true,
		// Not read-only: it mutates session state and writes archives.
		ReadOnly: func(json.RawMessage) bool { return false },
		// Not concurrency-safe: two compressions racing on the same state would
		// interleave block allocation and coverage.
		Concurrent: func(json.RawMessage) bool { return false },
		// No permission prompt: it touches only noa's own archive directory, and
		// asking would interrupt the very flow it exists to keep smooth.
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Decision{Behavior: permission.Allow}
		},
		Run: func(ctx context.Context, input json.RawMessage, tc *tool.ToolContext) (tool.Result, error) {
			return runCompress(sess, input, tc)
		},
	})
}

func runCompress(sess *Session, input json.RawMessage, tc *tool.ToolContext) (tool.Result, error) {
	parsed := noa.ParseCompressArgs(input)

	// A parse failure is a real error: the model produced something unusable and
	// must see that, so it corrects the shape rather than the content.
	if !parsed.Diagnostics.Ok && len(parsed.Ranges) == 0 {
		sess.recordParseFailure()
		return errorResult(formatParseError(parsed.Diagnostics)), nil
	}
	if len(parsed.Ranges) == 0 {
		// Neither success nor failure: a valid question with a valid answer.
		// Counting it either way would distort the failure ladder.
		return textResult(noa.NoRangesMessage), nil
	}

	// A range set that has already failed twice will fail identically again;
	// refusing it outright costs one short message instead of a whole turn.
	if refusal := sess.checkDeadRange(parsed.Ranges); refusal != "" {
		sess.recordParseFailure()
		return errorResult(refusal), nil
	}

	callID := ""
	if tc != nil {
		callID = tc.ToolUseID
	}

	before := sess.tokensBefore()
	res := sess.applyCompression(parsed.Ranges, callID)
	after := before - res.TokensCompressed

	panel := noa.FormatPanel(noa.PanelInput{
		Result: res, BeforeTokens: before, AfterTokens: max(after, 0),
	})
	if parsed.Diagnostics.Salvaged || parsed.Diagnostics.TailRepaired {
		panel += "\n  note: the arguments were malformed and were repaired before parsing; " +
			"check the call shape if this recurs."
	}
	// A panel is not an error even when it reports nothing created — the call was
	// well-formed. But it reclaimed nothing, so the failure ladder must still
	// count it, which applyCompression already did.
	return textResult(panel), nil
}

// formatParseError tells the model what shape its arguments arrived in.
func formatParseError(d noa.CompressParseDiagnostics) string {
	var b strings.Builder
	switch d.Kind {
	case noa.ParseEmptyInput:
		b.WriteString("Compress received no arguments.")
	case noa.ParseMissingContent:
		b.WriteString("Compress arguments have no `content` field.")
	case noa.ParseContentNotList:
		b.WriteString("Compress `content` is neither an array of ranges nor a JSON-encoded one.")
	case noa.ParseNoValidRanges:
		b.WriteString("No usable ranges: every entry was missing startId, endId or summary.")
	case noa.ParseTruncated:
		b.WriteString("The arguments were cut off mid-object and could not be salvaged.")
	default:
		b.WriteString("Compress arguments could not be parsed as JSON.")
	}
	b.WriteString(" Expected shape: " +
		`{ "content": [{ "startId": "m00012", "endId": "m00045", "summary": "..." }] }`)
	if d.InvalidItems > 0 {
		fmt.Fprintf(&b, "\n%d entr%s dropped: %s", d.InvalidItems,
			plural(d.InvalidItems, "y was", "ies were"), strings.Join(d.InvalidReasons, "; "))
	}
	if d.RawPrefix != "" {
		fmt.Fprintf(&b, "\nReceived (first %d chars): %s", len(d.RawPrefix), d.RawPrefix)
	}
	return b.String()
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

func textResult(s string) tool.Result {
	return tool.Result{Content: []llm.ContentBlock{llm.TextBlock(s)}}
}

func errorResult(s string) tool.Result {
	return tool.Result{Content: []llm.ContentBlock{llm.TextBlock(s)}, IsError: true}
}
