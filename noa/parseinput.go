package noa

import (
	"encoding/json"
	"regexp"
	"strings"
)

// CompressParseKind classifies why parsing ended the way it did. The kind is
// reported back so the model is told what shape its output was in, not just
// that it failed.
type CompressParseKind string

const (
	ParseOK             CompressParseKind = "ok"
	ParseEmptyInput     CompressParseKind = "empty-input"
	ParseNotObject      CompressParseKind = "not-object"
	ParseMissingContent CompressParseKind = "missing-content"
	ParseContentNotList CompressParseKind = "content-not-array"
	ParseNoValidRanges  CompressParseKind = "no-valid-ranges"
	ParseTruncated      CompressParseKind = "truncated"
	ParseMalformedJSON  CompressParseKind = "malformed-json"
)

// CompressParseDiagnostics records how the arguments arrived.
type CompressParseDiagnostics struct {
	Ok             bool
	Kind           CompressParseKind
	InvalidItems   int
	InvalidReasons []string
	// RawPrefix is the first 800 characters of the raw input, for logs.
	RawPrefix string
	Length    int
	Keys      []string
	// Salvaged marks output recovered by bracket-matching rather than parsed.
	Salvaged bool
	// TailRepaired marks output fixed by appending a missing closing brace.
	TailRepaired bool
}

// CompressParseResult is the parsed call.
type CompressParseResult struct {
	Ranges          []CompressRange
	TopLevelTopic   string
	SummaryMaxChars *int
	Diagnostics     CompressParseDiagnostics
}

const rawPrefixLen = 800

// ParseCompressArgs decodes a Compress call as leniently as it safely can.
//
// Strict parsing here is not a virtue: a model's only compression attempt in a
// long turn can arrive with a trailing comma, a stringified array, or a
// truncated tail, and rejecting it loses the whole turn's compression. Each
// layer below exists because some provider or model produced exactly that shape.
func ParseCompressArgs(raw []byte) CompressParseResult {
	res := CompressParseResult{}
	d := &res.Diagnostics
	d.Length = len(raw)
	if len(raw) > rawPrefixLen {
		d.RawPrefix = string(raw[:rawPrefixLen])
	} else {
		d.RawPrefix = string(raw)
	}

	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		d.Kind = ParseEmptyInput
		return res
	}

	// Layer 1-2: the arguments may themselves be a JSON string, and may be
	// wrapped in a markdown fence.
	body := trimmed
	var asString string
	if json.Unmarshal([]byte(body), &asString) == nil {
		body = strings.TrimSpace(asString)
	}
	body = stripFence(body)

	obj, ok := tryParseLenient(body)
	if !ok {
		// Layer 6: salvage whole entries by bracket matching. This is what
		// recovers a response the provider cut off mid-object.
		if entries, salvaged := salvageContentEntries(body); salvaged {
			d.Salvaged = true
			d.Kind = ParseTruncated
			res.Ranges = entriesToRanges(entries, "", nil, d)
			d.Ok = len(res.Ranges) > 0
			return res
		}
		d.Kind = ParseMalformedJSON
		return res
	}
	for k := range obj {
		d.Keys = append(d.Keys, k)
	}

	res.TopLevelTopic = stringField(obj, "topic")
	res.SummaryMaxChars = intField(obj, "summaryMaxChars")

	rawContent, present := obj["content"]
	if !present {
		d.Kind = ParseMissingContent
		return res
	}

	entries, kind := decodeContent(rawContent, d)
	if kind != ParseOK {
		d.Kind = kind
		return res
	}

	res.Ranges = entriesToRanges(entries, res.TopLevelTopic, res.SummaryMaxChars, d)
	if len(res.Ranges) == 0 {
		d.Kind = ParseNoValidRanges
		return res
	}
	d.Kind = ParseOK
	d.Ok = true
	return res
}

// decodeContent unwraps the content field, which arrives as an array, or — from
// providers without strict tool support — as a JSON-encoded string of one.
func decodeContent(rawContent any, d *CompressParseDiagnostics) ([]map[string]any, CompressParseKind) {
	switch v := rawContent.(type) {
	case []any:
		return toEntryMaps(v), ParseOK
	case string:
		s := stripFence(strings.TrimSpace(v))
		if arr, ok := tryParseLenientArray(s); ok {
			return toEntryMaps(arr), ParseOK
		}
		// Layer 5: non-strict tool modes sometimes drop the final closing brace,
		// leaving `..."]` where `..."}]` belongs. Appending it is safe because
		// valid JSON cannot become parseable by adding a brace.
		if repaired, ok := tailRepair(s); ok {
			d.TailRepaired = true
			return toEntryMaps(repaired), ParseOK
		}
		if entries, ok := salvageContentEntries(s); ok {
			d.Salvaged = true
			return entries, ParseOK
		}
		return nil, ParseContentNotList
	default:
		return nil, ParseContentNotList
	}
}

func toEntryMaps(arr []any) []map[string]any {
	out := make([]map[string]any, 0, len(arr))
	for _, it := range arr {
		if m, ok := it.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// entriesToRanges validates each entry and applies the top-level defaults.
func entriesToRanges(entries []map[string]any, topTopic string, topMax *int, d *CompressParseDiagnostics) []CompressRange {
	var out []CompressRange
	for _, e := range entries {
		// Field aliases: models and providers disagree on the spelling, and the
		// disagreement is not worth losing a compression over.
		start := firstString(e, "startId", "startRef", "messageId")
		end := firstString(e, "endId", "endRef", "messageId")
		summary := stringField(e, "summary")

		var why []string
		if start == "" {
			why = append(why, "missing startId")
		}
		if end == "" {
			why = append(why, "missing endId")
		}
		if summary == "" {
			why = append(why, "missing summary")
		}
		if len(why) > 0 {
			d.InvalidItems++
			d.InvalidReasons = append(d.InvalidReasons, strings.Join(why, ", "))
			continue
		}
		r := CompressRange{StartRef: start, EndRef: end, Summary: summary, Topic: stringField(e, "topic")}
		if r.Topic == "" {
			r.Topic = topTopic
		}
		if m := intField(e, "summaryMaxChars"); m != nil {
			r.SummaryMaxChars = m
		} else {
			r.SummaryMaxChars = topMax
		}
		out = append(out, r)
	}
	return out
}

var fenceRE = regexp.MustCompile("(?s)^\\s*```[a-zA-Z0-9]*\\s*\\n(.*?)\\n?\\s*```\\s*$")

// stripFence removes a surrounding markdown code fence.
func stripFence(s string) string {
	if m := fenceRE.FindStringSubmatch(s); m != nil {
		return strings.TrimSpace(m[1])
	}
	return s
}

// tryParseLenient decodes an object, repairing the two malformations that show
// up in practice: trailing commas, and raw newlines inside string literals.
func tryParseLenient(s string) (map[string]any, bool) {
	var obj map[string]any
	if json.Unmarshal([]byte(s), &obj) == nil {
		return obj, true
	}
	if json.Unmarshal([]byte(stripTrailingCommas(s)), &obj) == nil {
		return obj, true
	}
	if json.Unmarshal([]byte(escapeRawNewlinesInStrings(s)), &obj) == nil {
		return obj, true
	}
	if json.Unmarshal([]byte(escapeRawNewlinesInStrings(stripTrailingCommas(s))), &obj) == nil {
		return obj, true
	}
	return nil, false
}

func tryParseLenientArray(s string) ([]any, bool) {
	var arr []any
	if json.Unmarshal([]byte(s), &arr) == nil {
		return arr, true
	}
	if json.Unmarshal([]byte(stripTrailingCommas(s)), &arr) == nil {
		return arr, true
	}
	if json.Unmarshal([]byte(escapeRawNewlinesInStrings(s)), &arr) == nil {
		return arr, true
	}
	return nil, false
}

// tailRepair fixes an array whose last object lost its closing brace.
//
// Deterministic and safe: well-formed JSON cannot be made parseable by
// appending a brace, so a successful repair proves the input was malformed in
// exactly this way.
func tailRepair(s string) ([]any, bool) {
	t := strings.TrimSpace(s)
	if !strings.HasSuffix(t, "]") {
		return nil, false
	}
	withoutBracket := strings.TrimSpace(strings.TrimSuffix(t, "]"))
	if !strings.HasSuffix(withoutBracket, `"`) {
		return nil, false
	}
	var arr []any
	if json.Unmarshal([]byte(withoutBracket+"}]"), &arr) == nil {
		return arr, true
	}
	return nil, false
}

var contentKeyRE = regexp.MustCompile(`"content"\s*:\s*\[`)

// salvageContentEntries recovers whole objects from a truncated array by
// matching braces by hand. Everything up to the cut survives; only the partial
// final object is lost.
func salvageContentEntries(s string) ([]map[string]any, bool) {
	loc := contentKeyRE.FindStringIndex(s)
	start := 0
	if loc != nil {
		start = loc[1]
	} else if i := strings.Index(s, "["); i >= 0 {
		start = i + 1
	} else {
		return nil, false
	}

	var out []map[string]any
	depth, objStart := 0, -1
	inStr, esc := false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			if depth == 0 {
				objStart = i
			}
			depth++
		case '}':
			depth--
			if depth == 0 && objStart >= 0 {
				var m map[string]any
				if json.Unmarshal([]byte(s[objStart:i+1]), &m) == nil {
					out = append(out, m)
				}
				objStart = -1
			}
		case ']':
			if depth == 0 {
				return out, len(out) > 0
			}
		}
	}
	return out, len(out) > 0
}

// stripTrailingCommas removes commas before a closing brace or bracket, outside
// string literals.
func stripTrailingCommas(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inStr, esc := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			b.WriteByte(c)
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
			b.WriteByte(c)
			continue
		}
		if c == ',' {
			j := i + 1
			for j < len(s) && (s[j] == ' ' || s[j] == '\n' || s[j] == '\r' || s[j] == '\t') {
				j++
			}
			if j < len(s) && (s[j] == '}' || s[j] == ']') {
				continue // drop the comma
			}
		}
		b.WriteByte(c)
	}
	return b.String()
}

// escapeRawNewlinesInStrings escapes literal newlines and tabs that appear
// inside string literals, which JSON forbids but models emit freely when a
// summary contains formatting.
func escapeRawNewlinesInStrings(s string) string {
	var b strings.Builder
	b.Grow(len(s) + 16)
	inStr, esc := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
				b.WriteByte(c)
			case c == '\\':
				esc = true
				b.WriteByte(c)
			case c == '"':
				inStr = false
				b.WriteByte(c)
			case c == '\n':
				b.WriteString(`\n`)
			case c == '\r':
				b.WriteString(`\r`)
			case c == '\t':
				b.WriteString(`\t`)
			default:
				b.WriteByte(c)
			}
			continue
		}
		if c == '"' {
			inStr = true
		}
		b.WriteByte(c)
	}
	return b.String()
}

func stringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func firstString(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if s := stringField(m, k); s != "" {
			return s
		}
	}
	return ""
}

func intField(m map[string]any, key string) *int {
	switch v := m[key].(type) {
	case float64:
		n := int(v)
		return &n
	case json.Number:
		if i, err := v.Int64(); err == nil {
			n := int(i)
			return &n
		}
	}
	return nil
}
