package noaadapter

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/noa"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// harness.execOne validates a call against the tool's InputSchema before
// dispatching it (harness/tools.go), so a test that calls runCompress directly
// is testing a path the model may not be able to reach. These tests put the
// arguments through the same gate the harness uses, so the lenient parsing the
// rest of the suite exercises is known to be reachable in production.
//
// Compress opts out of that gate (tool.Spec.RawInput) because the shapes it
// exists to repair are, by definition, not valid JSON. The opt-out is asserted
// here rather than assumed.

// gate mirrors harness.execOne's validation step exactly, including the
// AcceptsRawInput opt-out the harness consults first.
func gate(t *testing.T, input string) error {
	t.Helper()
	ct := newCompressTool(simSession(t, 40000))
	if r, ok := ct.(interface{ AcceptsRawInput() bool }); ok && r.AcceptsRawInput() {
		return nil
	}
	return tool.ValidateInput(ct.InputSchema(), json.RawMessage(input))
}

// The opt-out has to be on, or every lenient-parsing test below is testing a
// path production cannot reach.
func TestCompressOptsOutOfSchemaValidation(t *testing.T) {
	ct := newCompressTool(simSession(t, 40000))
	r, ok := ct.(interface{ AcceptsRawInput() bool })
	if !ok || !r.AcceptsRawInput() {
		t.Fatal("Compress does not report AcceptsRawInput, so harness.execOne validates its " +
			"arguments and ParseCompressArgs never sees a malformed call")
	}
	// The schema must still be advertised: it is what tells the model the shape
	// to aim for, and dropping it would be a much worse trade than the gate.
	if ct.InputSchema() == nil {
		t.Fatal("Compress advertises no schema; the model has nothing to aim for")
	}
	if _, ok := ct.InputSchema()["properties"]; !ok {
		t.Fatalf("the advertised schema has no properties: %v", ct.InputSchema())
	}
}

// The lenient parser exists to recover ranges from arguments a provider mangled.
// That recovery is only worth anything if the mangled arguments can REACH it.
//
// So: anything ParseCompressArgs can salvage must survive the schema gate.
// Where it does not, the salvage code is unreachable in production and the model
// gets the harness's terse validation error instead of noa's repair guidance.
func TestSalvageableArgumentsSurviveTheSchemaGate(t *testing.T) {
	// Every one of these is a shape ParseCompressArgs is built to repair.
	corpus := map[string]string{
		"array of ranges":        `{"content":[{"startId":"m00001","endId":"m00009","summary":"s"}]}`,
		"JSON-encoded array":     `{"content":"[{\"startId\":\"m00001\",\"endId\":\"m00009\",\"summary\":\"s\"}]"}`,
		"trailing commas":        `{"content":[{"startId":"m00001","endId":"m00009","summary":"s",},],}`,
		"fenced in markdown":     "```json\n{\"content\":[{\"startId\":\"m00001\",\"endId\":\"m00009\",\"summary\":\"s\"}]}\n```",
		"truncated mid-object":   `{"content":[{"startId":"m00001","endId":"m00009","summary":"first"},{"startId":"m00010","end`,
		"double-encoded whole":   `"{\"content\":[{\"startId\":\"m00001\",\"endId\":\"m00009\",\"summary\":\"s\"}]}"`,
		"unterminated inner":     `{"content":"[{\"startId\":\"m00001\",\"endId\":\"m00009\",\"summary\":\"s\"]"}`,
		"raw newline in summary": "{\"content\":[{\"startId\":\"m00001\",\"endId\":\"m00009\",\"summary\":\"line one\nline two\"}]}",
	}

	var blocked []string
	for name, args := range corpus {
		parsed := noa.ParseCompressArgs([]byte(args))
		if len(parsed.Ranges) == 0 {
			t.Errorf("%s: the parser recovered nothing; the corpus entry is wrong", name)
			continue
		}
		if err := gate(t, args); err != nil {
			blocked = append(blocked, name+": "+err.Error())
		}
	}
	if len(blocked) > 0 {
		t.Fatalf("the schema gate rejects %d argument shapes the parser can repair, so the "+
			"repair never runs in production and the model sees a bare validation error "+
			"instead of noa's guidance:\n  %s", len(blocked), strings.Join(blocked, "\n  "))
	}
}

// The other half: shapes the parser correctly REJECTS must still reach it, so
// the model gets formatParseError's "expected shape" — which names Compress's
// contract — rather than a schema message that does not.
func TestUnparseableArgumentsReachNoasDiagnostics(t *testing.T) {
	corpus := map[string]string{
		"no content field":  `{"topic":"t"}`,
		"content is object": `{"content":{"startId":"m00001"}}`,
		"empty input":       ``,
		"ranges incomplete": `{"content":[{"startId":"m00001"}]}`,
	}
	for name, args := range corpus {
		t.Run(name, func(t *testing.T) {
			if err := gate(t, args); err != nil {
				t.Fatalf("the gate answered before noa did (%v), so formatParseError never runs "+
					"and the model is told nothing about the shape Compress wants", err)
			}
			res, err := runCompress(simSession(t, 40000), json.RawMessage(args), nil)
			if err != nil {
				t.Fatalf("runCompress returned a Go error: %v", err)
			}
			if !res.IsError {
				t.Fatalf("unparseable arguments produced a success result: %s", res.Content[0].Text)
			}
			if !strings.Contains(res.Content[0].Text, "Expected shape:") {
				t.Fatalf("the model is not told the expected shape:\n%s", res.Content[0].Text)
			}
		})
	}
}

// A parse failure feeds the failure ladder (recordParseFailure). If the gate
// answers first, the ladder never counts it — so a model stuck emitting a broken
// shape is never detected as stuck by the mechanism built to detect exactly that.
func TestParseFailuresReachTheFailureLadder(t *testing.T) {
	sess := simSession(t, 40000)
	broken := `{"content":[{"startId":"m00001","endId":"m00009","summ`

	if err := gate(t, broken); err != nil {
		t.Skipf("the gate rejects this before the tool runs (%v); "+
			"TestSalvageableArgumentsSurviveTheSchemaGate is the test that covers the consequence", err)
	}
	before := sess.attempts
	if _, err := runCompress(sess, json.RawMessage(broken), nil); err != nil {
		t.Fatalf("runCompress returned a Go error: %v", err)
	}
	if sess.attempts <= before {
		t.Fatalf("the attempt counter stayed at %d after an unparseable call; "+
			"the ladder cannot notice a model that is stuck", before)
	}
}

// The full production path, exercised the way the harness does it: resolve the
// built tool, validate, check permissions and concurrency, then Run with a real
// ToolContext. Tests that call runCompress skip all of this.
func TestCompressToolThroughTheBuiltToolInterface(t *testing.T) {
	sess := simSession(t, 40000)
	a := newSimAgent(t, sess, 1000000) // build history without compressing
	for range 12 {
		a.work(4000)
		a.observe()
	}

	ct := newCompressTool(sess)
	if ct.Name() != noa.CompressToolName {
		t.Fatalf("tool name = %q, want %q", ct.Name(), noa.CompressToolName)
	}

	// A range wide enough to clear the minimum-content rule, well clear of the
	// protected tail.
	refs := sess.State().MessageRefs
	if len(refs.ByRef) < 16 {
		t.Fatalf("only %d refs assigned; the fixture is too short", len(refs.ByRef))
	}
	args, _ := json.Marshal(map[string]any{"content": []map[string]any{{
		"startId": noa.IndexToRef(2), "endId": noa.IndexToRef(13),
		"summary": a.summary("through the tool interface"),
	}}})

	if err := tool.ValidateInput(ct.InputSchema(), args); err != nil {
		t.Fatalf("a well-formed call failed schema validation: %v", err)
	}
	if ct.IsReadOnly(args) {
		t.Fatal("Compress reports read-only; it writes archives and mutates state")
	}
	if ct.IsConcurrencySafe(args) {
		t.Fatal("Compress reports concurrency-safe; two racing compressions would interleave block allocation")
	}

	res, err := ct.Call(t.Context(), args, &tool.ToolContext{ToolUseID: "toolu_gate_1"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.IsError {
		t.Fatalf("a well-formed compression reported an error: %s", res.Content[0].Text)
	}
	if n := noa.PanelBlockCount(res.Content[0].Text); n != 1 {
		t.Fatalf("panel reports %d blocks, want 1:\n%s", n, res.Content[0].Text)
	}

	// The ToolUseID must have been carried into the block, or hide-compress-calls
	// cannot find the call to hide and RebuildStateFromLog cannot replay it.
	blocks := noa.ActiveBlocks(sess.State())
	if len(blocks) != 1 {
		t.Fatalf("%d active blocks, want 1", len(blocks))
	}
	if blocks[0].CompressCallID != "toolu_gate_1" {
		t.Fatalf("block records CompressCallID %q, want the ToolContext's %q — "+
			"without it the call cannot be hidden or replayed",
			blocks[0].CompressCallID, "toolu_gate_1")
	}
}

// The permission decision must be Allow without asking. A prompt here would
// interrupt the flow the tool exists to keep smooth, and in a headless run it
// would hang.
func TestCompressNeverPromptsForPermission(t *testing.T) {
	sess := simSession(t, 40000)
	ct := newCompressTool(sess)
	dec := ct.CheckPermissions(t.Context(), json.RawMessage(`{"content":[]}`), permission.Context{})
	if dec.Behavior != permission.Allow {
		t.Fatalf("permission behavior = %s, want allow", dec.Behavior)
	}
}
