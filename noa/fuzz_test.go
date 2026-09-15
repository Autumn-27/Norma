package noa

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// ParseCompressArgs is the most adversarial surface in the package: it accepts
// whatever a model emitted, through whatever a provider did to it on the way.
// Its contract is that it never panics and never invents a range — recovering
// less is always acceptable, recovering something that was not asked for is not.
func FuzzParseCompressArgs(f *testing.F) {
	seeds := []string{
		`{"content":[{"startId":"m00012","endId":"m00045","summary":"did the thing"}]}`,
		`{"content":[]}`,
		`{"content":"[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"}]"}`,
		`{"content":[{"startId":"m1","endId":"m2","summary":"s",},],}`,
		"```json\n{\"content\":[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"}]}\n```",
		`{"content":"[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"]"}`,
		`{"content":[{"startId":"m1","endId":"m2","summary":"first"},{"startId":"m3","end`,
		`{"topic":"t","summaryMaxChars":30000,"content":[{"startId":"m1","endId":"m2","summary":"s"}]}`,
		`"{\"content\":[{\"startId\":\"m1\",\"endId\":\"m2\",\"summary\":\"s\"}]}"`,
		`{"content":[{"startId":"认证","endId":"m2","summary":"中文摘要 🔐"}]}`,
		``,
		`null`,
		`[]`,
		`{`,
		`not json`,
		`{"content":{"startId":"m1"}}`,
		`{"content":[[[[]]]]}`,
		`{"content":[{"summary":"` + strings.Repeat("x", 5000) + `"}]}`,
	}
	for _, s := range seeds {
		f.Add([]byte(s))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		res := ParseCompressArgs(data) // must not panic

		if res.Diagnostics.Length != len(data) {
			t.Fatalf("Length = %d, want the input size %d", res.Diagnostics.Length, len(data))
		}
		if len(res.Diagnostics.RawPrefix) > rawPrefixLen {
			t.Fatalf("RawPrefix is %d bytes, over the %d cap", len(res.Diagnostics.RawPrefix), rawPrefixLen)
		}

		// Every returned range must be usable: a half-formed one would reach the
		// engine and fail there instead, with a worse message.
		for i, r := range res.Ranges {
			if r.StartRef == "" || r.EndRef == "" || r.Summary == "" {
				t.Fatalf("range %d is incomplete: %+v", i, r)
			}
			if r.SummaryMaxChars != nil && *r.SummaryMaxChars < 0 {
				t.Fatalf("range %d has a negative summaryMaxChars: %d", i, *r.SummaryMaxChars)
			}
		}
		// The contract callers branch on:
		//   Ok  == "usable ranges were recovered"
		//   Kind == "how they arrived"
		// A salvaged parse is legitimately both Ok and ParseTruncated — dropping
		// the kind would hide the malformation from the logs, and dropping Ok
		// would throw away ranges that were successfully recovered.
		if res.Diagnostics.Ok {
			switch res.Diagnostics.Kind {
			case ParseOK, ParseTruncated:
			default:
				t.Fatalf("Ok with kind %s", res.Diagnostics.Kind)
			}
			if len(res.Ranges) == 0 && res.Diagnostics.Kind == ParseTruncated {
				t.Fatal("Ok and truncated but no ranges were recovered")
			}
		} else if len(res.Ranges) != 0 {
			t.Fatalf("not Ok but carrying %d ranges (kind %s) — callers cannot branch on that",
				len(res.Ranges), res.Diagnostics.Kind)
		}
	})
}

// Boundary parsing decides whether a range compresses raw messages or
// consolidates blocks, so a wrong answer changes the tier of the result.
func FuzzParseBoundary(f *testing.F) {
	for _, s := range []string{
		"m00001", "m1", "m99999", "b1", "b999", "b3(T2)", "  M00007  ",
		"", "m", "m0", "b0", "0", "-1", "m100000", "认证", "b1(T", "m00001x",
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		ref, ok := ParseBoundary(s) // must not panic
		if !ok {
			return
		}
		switch ref.kind {
		case BoundaryMessage:
			// A message boundary must round-trip: the engine looks the raw form
			// up in the ref map, so a normalisation that loses information would
			// silently miss.
			if ref.num < MinRefIndex || ref.num > MaxRefIndex {
				t.Fatalf("ParseBoundary(%q) produced out-of-range index %d", s, ref.num)
			}
			if ref.raw != IndexToRef(ref.num) {
				t.Fatalf("ParseBoundary(%q).raw = %q, want the canonical %q", s, ref.raw, IndexToRef(ref.num))
			}
		case BoundaryBlock:
			if ref.num < 1 {
				t.Fatalf("ParseBoundary(%q) produced block index %d", s, ref.num)
			}
			if !strings.HasPrefix(ref.raw, "b") {
				t.Fatalf("ParseBoundary(%q).raw = %q, want a bN form", s, ref.raw)
			}
		default:
			t.Fatalf("ParseBoundary(%q) reported ok with kind %q", s, ref.kind)
		}
	})
}

// Truncation rewrites message bodies. It must never produce invalid UTF-8 (a
// provider would reject it) and never grow what it was asked to shrink.
func FuzzTruncateBody(f *testing.F) {
	f.Add(strings.Repeat("a", 10000), 2500)
	f.Add(strings.Repeat("中文", 5000), 2500)
	f.Add(strings.Repeat("🔐", 4000), 2500)
	f.Add("short", 1)

	f.Fuzz(func(t *testing.T, body string, tokens int) {
		if !utf8.ValidString(body) {
			t.Skip("the input is not valid UTF-8 to begin with")
		}
		if len([]rune(body)) <= truncateKeepPrefixChars+truncateKeepSuffixChars {
			t.Skip("too short to truncate; the caller skips these")
		}
		out := truncateBody(body, tokens)

		if !utf8.ValidString(out) {
			t.Fatal("truncation produced invalid UTF-8")
		}
		if !strings.Contains(out, TruncationMarker) {
			t.Fatal("the truncation marker is missing; the result would be cut again next turn")
		}
		if len([]rune(out)) >= len([]rune(body)) {
			t.Fatalf("truncation grew the body from %d to %d runes", len([]rune(body)), len([]rune(out)))
		}
	})
}

// The ref map is the addressing system; a collision or a lost entry breaks
// every range the model holds.
func FuzzAssignRefs(f *testing.F) {
	f.Add("a|b|c", 1)
	f.Add("", 1)
	f.Add("x", 99999)
	f.Add("dup|dup|dup", 1)

	f.Fuzz(func(t *testing.T, idList string, start int) {
		if start < 1 || start > MaxRefIndex {
			t.Skip("out of the addressable range")
		}
		var msgs []CoreMessage
		for _, id := range strings.Split(idList, "|") {
			msgs = append(msgs, msg(id, RoleUser, CTText, "body"))
		}
		res := AssignRefs(msgs, AssignRefsOptions{
			Existing:  MessageRefMap{ByRaw: map[string]string{}, ByRef: map[string]string{}},
			NextIndex: start,
		})

		// The two directions must stay consistent, or a lookup in one would
		// disagree with the other.
		for raw, ref := range res.Map.ByRaw {
			if ref == BlockedRef {
				continue
			}
			if back := res.Map.ByRef[ref]; back != raw {
				t.Fatalf("ByRaw[%q]=%q but ByRef[%q]=%q", raw, ref, ref, back)
			}
		}
		seen := map[string]string{}
		for ref, raw := range res.Map.ByRef {
			if prev, dup := seen[ref]; dup {
				t.Fatalf("ref %q assigned to both %q and %q", ref, prev, raw)
			}
			seen[ref] = raw
		}
	})
}

// A summary stub travels back to the model inside a tool call's arguments, so
// it has to remain valid text of a bounded size.
func FuzzStubSummary(f *testing.F) {
	f.Add(strings.Repeat("a", 1000))
	f.Add(strings.Repeat("中", 1000))
	f.Add("short")
	f.Add("")

	f.Fuzz(func(t *testing.T, s string) {
		if !utf8.ValidString(s) {
			t.Skip("invalid UTF-8 input")
		}
		out, cut := stubSummary(s)
		if !utf8.ValidString(out) {
			t.Fatal("the stub is not valid UTF-8")
		}
		if !cut {
			if out != s {
				t.Fatal("an uncut summary was modified")
			}
			return
		}
		if n := len([]rune(out)); n != SummaryStubChars {
			t.Fatalf("stub is %d runes, want exactly %d", n, SummaryStubChars)
		}
		if !strings.HasSuffix(out, "…") {
			t.Fatal("the stub does not end with an ellipsis")
		}
	})
}
