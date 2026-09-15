package noa

import (
	"strings"
	"testing"
)

func parse(s string) CompressParseResult { return ParseCompressArgs([]byte(s)) }

func TestParseCompressArgsWellFormed(t *testing.T) {
	r := parse(`{"content":[{"startId":"m00012","endId":"m00045","summary":"did the thing","topic":"auth"}]}`)
	if !r.Diagnostics.Ok || r.Diagnostics.Kind != ParseOK {
		t.Fatalf("diagnostics = %+v, want ok", r.Diagnostics)
	}
	if len(r.Ranges) != 1 {
		t.Fatalf("Ranges = %+v, want 1", r.Ranges)
	}
	g := r.Ranges[0]
	if g.StartRef != "m00012" || g.EndRef != "m00045" || g.Summary != "did the thing" || g.Topic != "auth" {
		t.Fatalf("range = %+v", g)
	}
}

func TestParseCompressArgsEmpty(t *testing.T) {
	for _, in := range []string{"", "   ", "null"} {
		r := parse(in)
		if r.Diagnostics.Kind != ParseEmptyInput {
			t.Errorf("parse(%q) kind = %s, want %s", in, r.Diagnostics.Kind, ParseEmptyInput)
		}
	}
}

func TestParseCompressArgsMissingContent(t *testing.T) {
	r := parse(`{"topic":"auth"}`)
	if r.Diagnostics.Kind != ParseMissingContent {
		t.Fatalf("kind = %s, want %s", r.Diagnostics.Kind, ParseMissingContent)
	}
}

func TestParseCompressArgsMarkdownFence(t *testing.T) {
	r := parse("```json\n{\"content\":[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"}]}\n```")
	if !r.Diagnostics.Ok || len(r.Ranges) != 1 {
		t.Fatalf("a fenced payload must parse: %+v", r.Diagnostics)
	}
}

func TestParseCompressArgsTrailingCommas(t *testing.T) {
	r := parse(`{"content":[{"startId":"m1","endId":"m2","summary":"s",},],}`)
	if !r.Diagnostics.Ok || len(r.Ranges) != 1 {
		t.Fatalf("trailing commas must be tolerated: %+v", r.Diagnostics)
	}
}

// Models emit literal newlines inside summaries constantly; JSON forbids them.
func TestParseCompressArgsRawNewlinesInStrings(t *testing.T) {
	r := parse("{\"content\":[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"line one\nline two\"}]}")
	if !r.Diagnostics.Ok || len(r.Ranges) != 1 {
		t.Fatalf("raw newlines must be escaped and parsed: %+v", r.Diagnostics)
	}
	if !strings.Contains(r.Ranges[0].Summary, "line one") || !strings.Contains(r.Ranges[0].Summary, "line two") {
		t.Fatalf("summary = %q, want both lines preserved", r.Ranges[0].Summary)
	}
}

// Providers without strict tool support stringify nested arrays.
func TestParseCompressArgsStringifiedContent(t *testing.T) {
	r := parse(`{"content":"[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"}]"}`)
	if !r.Diagnostics.Ok || len(r.Ranges) != 1 {
		t.Fatalf("a stringified content array must parse: %+v", r.Diagnostics)
	}
}

// The whole argument object arriving as a JSON string.
func TestParseCompressArgsDoubleStringified(t *testing.T) {
	r := parse(`"{\"content\":[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"}]}"`)
	if !r.Diagnostics.Ok || len(r.Ranges) != 1 {
		t.Fatalf("a double-stringified payload must parse: %+v", r.Diagnostics)
	}
}

// Qwen-style: the final object loses its closing brace, leaving `"]` where
// `"}]` belongs. Appending it is safe — valid JSON cannot be made parseable by
// adding a brace.
func TestParseCompressArgsTailRepair(t *testing.T) {
	r := parse(`{"content":"[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"]"}`)
	if !r.Diagnostics.Ok {
		t.Fatalf("tail repair failed: %+v", r.Diagnostics)
	}
	if !r.Diagnostics.TailRepaired {
		t.Fatal("TailRepaired must be recorded so the malformation is visible in logs")
	}
	if len(r.Ranges) != 1 || r.Ranges[0].StartRef != "m1" {
		t.Fatalf("Ranges = %+v", r.Ranges)
	}
}

// A response cut off mid-object: everything before the cut must still land.
func TestParseCompressArgsSalvagesTruncated(t *testing.T) {
	r := parse(`{"content":[{"startId":"m1","endId":"m2","summary":"first"},{"startId":"m3","endId":"m4","summary":"seco`)
	if len(r.Ranges) != 1 {
		t.Fatalf("Ranges = %+v, want the complete first entry salvaged", r.Ranges)
	}
	if r.Ranges[0].Summary != "first" {
		t.Fatalf("salvaged summary = %q", r.Ranges[0].Summary)
	}
	if !r.Diagnostics.Salvaged {
		t.Fatal("Salvaged must be recorded")
	}
	if r.Diagnostics.Kind != ParseTruncated {
		t.Fatalf("kind = %s, want %s", r.Diagnostics.Kind, ParseTruncated)
	}
}

func TestParseCompressArgsFieldAliases(t *testing.T) {
	cases := []string{
		`{"content":[{"startRef":"m1","endRef":"m2","summary":"s"}]}`,
		`{"content":[{"messageId":"m1","summary":"s"}]}`,
	}
	for _, in := range cases {
		r := parse(in)
		if !r.Diagnostics.Ok || len(r.Ranges) != 1 {
			t.Errorf("parse(%s) = %+v, want the aliases accepted", in, r.Diagnostics)
		}
	}
}

func TestParseCompressArgsTopLevelDefaults(t *testing.T) {
	r := parse(`{"topic":"shared","summaryMaxChars":30000,"content":[
		{"startId":"m1","endId":"m2","summary":"a"},
		{"startId":"m3","endId":"m4","summary":"b","topic":"own","summaryMaxChars":1000}]}`)
	if len(r.Ranges) != 2 {
		t.Fatalf("Ranges = %+v, want 2", r.Ranges)
	}
	if r.Ranges[0].Topic != "shared" {
		t.Fatalf("entry without its own topic = %q, want the top-level default", r.Ranges[0].Topic)
	}
	if r.Ranges[0].SummaryMaxChars == nil || *r.Ranges[0].SummaryMaxChars != 30000 {
		t.Fatalf("entry summaryMaxChars = %v, want the top-level default", r.Ranges[0].SummaryMaxChars)
	}
	if r.Ranges[1].Topic != "own" {
		t.Fatalf("entry with its own topic = %q, want it kept", r.Ranges[1].Topic)
	}
	if r.Ranges[1].SummaryMaxChars == nil || *r.Ranges[1].SummaryMaxChars != 1000 {
		t.Fatalf("per-entry summaryMaxChars = %v, want it to win", r.Ranges[1].SummaryMaxChars)
	}
}

func TestParseCompressArgsInvalidEntriesReported(t *testing.T) {
	r := parse(`{"content":[{"startId":"m1","endId":"m2","summary":"good"},{"startId":"m3"},{"summary":"no refs"}]}`)
	if len(r.Ranges) != 1 {
		t.Fatalf("Ranges = %+v, want only the valid entry", r.Ranges)
	}
	if r.Diagnostics.InvalidItems != 2 {
		t.Fatalf("InvalidItems = %d, want 2", r.Diagnostics.InvalidItems)
	}
	if len(r.Diagnostics.InvalidReasons) != 2 {
		t.Fatalf("InvalidReasons = %v, want one per dropped entry", r.Diagnostics.InvalidReasons)
	}
}

func TestParseCompressArgsNoValidRanges(t *testing.T) {
	r := parse(`{"content":[{"startId":"m1"}]}`)
	if r.Diagnostics.Kind != ParseNoValidRanges {
		t.Fatalf("kind = %s, want %s", r.Diagnostics.Kind, ParseNoValidRanges)
	}
	if r.Diagnostics.Ok {
		t.Fatal("Ok must be false when nothing usable was parsed")
	}
}

func TestParseCompressArgsMalformed(t *testing.T) {
	r := parse(`not json at all`)
	if r.Diagnostics.Ok {
		t.Fatalf("diagnostics = %+v, want failure", r.Diagnostics)
	}
	if r.Diagnostics.Kind != ParseMalformedJSON {
		t.Fatalf("kind = %s, want %s", r.Diagnostics.Kind, ParseMalformedJSON)
	}
}

func TestParseCompressArgsRawPrefixBounded(t *testing.T) {
	r := parse(strings.Repeat("x", 5000))
	if len(r.Diagnostics.RawPrefix) != rawPrefixLen {
		t.Fatalf("RawPrefix length = %d, want %d", len(r.Diagnostics.RawPrefix), rawPrefixLen)
	}
	if r.Diagnostics.Length != 5000 {
		t.Fatalf("Length = %d, want the full input length recorded", r.Diagnostics.Length)
	}
}

func TestStripTrailingCommasLeavesStringsAlone(t *testing.T) {
	in := `{"a":"x, }","b":[1,2,]}`
	got := stripTrailingCommas(in)
	if !strings.Contains(got, `"x, }"`) {
		t.Fatalf("stripTrailingCommas mangled a string literal: %q", got)
	}
	if strings.Contains(got, "2,]") {
		t.Fatalf("stripTrailingCommas left the trailing comma: %q", got)
	}
}

func TestEscapeRawNewlinesLeavesStructureAlone(t *testing.T) {
	in := "{\n  \"a\": \"line\nbreak\"\n}"
	got := escapeRawNewlinesInStrings(in)
	// Structural newlines stay; the one inside the literal is escaped.
	if strings.Count(got, "\n") != 2 {
		t.Fatalf("structural newlines changed: %q", got)
	}
	if !strings.Contains(got, `line\nbreak`) {
		t.Fatalf("the in-string newline was not escaped: %q", got)
	}
}

// Appending a brace can only ever rescue malformed input, never corrupt valid
// input — which is what makes the repair safe to attempt unconditionally.
func TestTailRepairRejectsWellFormedInput(t *testing.T) {
	if _, ok := tailRepair(`[{"startId":"m1","endId":"m2","summary":"s"}]`); ok {
		t.Fatal("tailRepair claimed to repair well-formed JSON")
	}
}

// An empty content list is a well-formed call that asks for nothing. Treating
// it as a parse failure would count it against the failure ladder and punish a
// model that correctly determined there was nothing to compress.
func TestParseCompressArgsEmptyContentIsNotAFailure(t *testing.T) {
	r := parse(`{"content":[]}`)
	if !r.Diagnostics.Ok {
		t.Fatalf("diagnostics = %+v, want an empty list to parse cleanly", r.Diagnostics)
	}
	if len(r.Ranges) != 0 {
		t.Fatalf("Ranges = %+v, want none", r.Ranges)
	}
	if r.Diagnostics.Kind != ParseOK {
		t.Fatalf("kind = %s, want %s", r.Diagnostics.Kind, ParseOK)
	}
}
