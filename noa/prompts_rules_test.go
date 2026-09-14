package noa

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// upstreamSource is the checked-out acp-kernel this port was taken from. The
// comparison is skipped when it is absent, so the suite still runs on a machine
// without it.
const upstreamSource = "/home/kali/acp-kernel/src/compression-rules.ts"

func upstreamConst(t *testing.T, name string) (string, bool) {
	t.Helper()
	raw, err := os.ReadFile(upstreamSource)
	if err != nil {
		return "", false
	}
	re := regexp.MustCompile("(?s)export const " + name + " = `(.*?)`;\n")
	m := re.FindSubmatch(raw)
	if m == nil {
		t.Fatalf("upstream constant %s not found in %s", name, upstreamSource)
	}
	return strings.ReplaceAll(string(m[1]), "\\`", "`"), true
}

// The four edits below are the ONLY divergence permitted from upstream. Each is
// a factual correction: upstream directs the model to a `decompress` tool that
// noa does not have, so leaving the text alone would send it after something
// that does not exist.
var allowedEdits = []struct{ upstream, ported string }{
	{"(or you, after decompressing)", "(or you, after reading the archive file)"},
	{"they let you or a later reader jump back via decompress to the exact original.",
		"they let you or a later reader locate the exact original inside the archive file."},
	{"This lets a later decompress target the right block by relevance, not by guessing locations.",
		"This lets a later reader grep the archive directory and find the right block by relevance, not by guessing locations."},
	{"they are ambiguous and cannot be grepped or decompressed-to later.",
		"they are ambiguous and cannot be grepped or located in the archive later."},
	{"you call `compress`", "you call `Compress`"},
}

func TestHowToCompressRulesMatchesUpstream(t *testing.T) {
	up, ok := upstreamConst(t, "HOW_TO_COMPRESS_RULES")
	if !ok {
		t.Skip("upstream acp-kernel checkout not present")
	}
	want := up
	for _, e := range allowedEdits {
		if !strings.Contains(want, e.upstream) {
			t.Fatalf("upstream no longer contains %q — the port's edit list is stale and must be revisited", e.upstream)
		}
		want = strings.Replace(want, e.upstream, e.ported, 1)
	}
	// The archive section is appended, so compare only the leading portion.
	got := HowToCompressRules
	idx := strings.Index(got, "\n\nTHE ARCHIVE CHANGES THE COST OF FORGETTING")
	if idx < 0 {
		t.Fatal("HowToCompressRules is missing the archive addendum")
	}
	if got[:idx] != want {
		t.Fatalf("HowToCompressRules diverges from upstream beyond the permitted edits.\nFirst difference at byte %d",
			firstDiff(got[:idx], want))
	}
}

func TestTierRulesAreVerbatim(t *testing.T) {
	cases := []struct {
		name   string
		ported string
	}{
		{"TIER2_DISTILL_RULES", Tier2Rules},
		{"TIER3_CONDENSE_RULES", Tier3Rules},
		{"COMPRESS_PHILOSOPHY", CompressPhilosophy},
	}
	for _, c := range cases {
		up, ok := upstreamConst(t, c.name)
		if !ok {
			t.Skip("upstream acp-kernel checkout not present")
		}
		if c.ported != up {
			t.Errorf("%s diverges from upstream at byte %d — it must be verbatim",
				c.name, firstDiff(c.ported, up))
		}
	}
}

// Whatever else changes, the rules must not send the model after a tool that
// does not exist.
func TestRulesMentionNoMissingTools(t *testing.T) {
	for name, text := range map[string]string{
		"CompressPhilosophy": CompressPhilosophy,
		"HowToCompressRules": HowToCompressRules,
		"Tier2Rules":         Tier2Rules,
		"Tier3Rules":         Tier3Rules,
	} {
		for _, missing := range []string{"decompress", "search_context", "acp_status"} {
			if strings.Contains(text, missing) {
				t.Errorf("%s mentions %q, which noa does not provide", name, missing)
			}
		}
	}
}

func TestHowToCompressRulesCarriesArchiveGuidance(t *testing.T) {
	if !strings.Contains(HowToCompressRules, "THE ARCHIVE CHANGES THE COST OF FORGETTING") {
		t.Fatal("the archive addendum is missing — without it the model treats compression as deletion and under-compresses")
	}
	// The closing constraint is what keeps the archive reachable.
	if !strings.Contains(HowToCompressRules, "leaves the archive unreachable") {
		t.Fatal("the addendum must warn that a summary lacking anchors makes the archive unfindable")
	}
}

func firstDiff(a, b string) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
