package noa

import (
	"strings"
	"testing"
)

func TestFormatTokens(t *testing.T) {
	cases := map[int]string{
		0: "0", 42: "42", 999: "999",
		1000: "1.0K", 2100: "2.1K", 9999: "10.0K",
		10000: "10K", 12345: "12K", 120000: "120K",
	}
	for in, want := range cases {
		if got := FormatTokens(in); got != want {
			t.Errorf("FormatTokens(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestFormatPanelSuccess(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: longSummary, Topic: "auth",
	})
	if len(res.BlocksCreated) == 0 {
		t.Fatalf("no block: %v", res.Errors)
	}
	panel := FormatPanel(PanelInput{Result: res, BeforeTokens: 12000, AfterTokens: 3100})

	if !strings.HasPrefix(panel, "▣ noa | 12K → 3.1K tokens (~8.9K reclaimed)") {
		t.Fatalf("headline = %q", strings.SplitN(panel, "\n", 2)[0])
	}
	b := res.BlocksCreated[0]
	if !strings.Contains(panel, b.BlockID+"(T1)=") {
		t.Fatalf("panel is missing the block line:\n%s", panel)
	}
	if !strings.Contains(panel, b.ArchivePath) {
		t.Fatalf("panel is missing the archive path:\n%s", panel)
	}
	if PanelBlockCount(panel) != 1 {
		t.Fatalf("PanelBlockCount = %d, want 1", PanelBlockCount(panel))
	}
}

// A panel reporting no blocks must be recognisable as such: it is not an error,
// but nothing was reclaimed, and the failure counter depends on telling the
// difference.
func TestFormatPanelZeroBlocks(t *testing.T) {
	panel := FormatPanel(PanelInput{Result: ApplyResult{Errors: []string{"Summary too short (5 chars, min 50)."}}})
	if !strings.Contains(panel, "0 blocks created") {
		t.Fatalf("panel = %q, want it to state that nothing was created", panel)
	}
	if PanelBlockCount(panel) != 0 {
		t.Fatalf("PanelBlockCount = %d, want 0", PanelBlockCount(panel))
	}
	if !strings.Contains(panel, "Summary too short") {
		t.Fatalf("panel lost the error text: %q", panel)
	}
}

// The tier label contains parentheses; a counter that scanned to a closing
// paren would miscount.
func TestPanelBlockCountHandlesTierLabels(t *testing.T) {
	panel := "▣ noa | 12K → 3K tokens (~9K reclaimed)\n" +
		"  b6(T2)=m00012–m00160* → /a/b6.md\n" +
		"  b7(T1)=m00161–m00204 → /a/b7.md\n" +
		"  note: b3(T2) was inside the requested range but was not consumed\n"
	// b3(T2) in the note has no "=", so it must not be counted as a created block.
	if got := PanelBlockCount(panel); got != 2 {
		t.Fatalf("PanelBlockCount = %d, want 2 — the note's b3(T2) carries no '=' and is not a created block", got)
	}
}

func TestFormatPanelReportsSurvivingBlocks(t *testing.T) {
	res := ApplyResult{
		BlocksCreated: []CompressionBlock{{
			BlockID: "b5", Tier: 2, StartRef: "m00012", EndRef: "m00160", ArchivePath: "/a/b5.md",
		}},
		SurvivingBlockIDs: []string{"b3"},
		State: CompressionState{
			Blocks:      []CompressionBlock{{BlockID: "b3", Tier: 2, Active: true}},
			MessageRefs: MessageRefMap{ByRef: map[string]string{}},
		},
	}
	panel := FormatPanel(PanelInput{Result: res, BeforeTokens: 12000, AfterTokens: 3000})
	if !strings.Contains(panel, "b3(T2) was inside the requested range but was not consumed") {
		t.Fatalf("panel must explain the surviving block:\n%s", panel)
	}
}

func TestFormatPanelIncludesWarnings(t *testing.T) {
	res := ApplyResult{
		BlocksCreated: []CompressionBlock{{BlockID: "b1", Tier: 1, StartRef: "m1", EndRef: "m9", ArchivePath: "/a.md"}},
		Warnings:      []string{"Excluded 2 protected message(s) m00203, m00204 (recent/last-user zone)."},
		State:         CompressionState{MessageRefs: MessageRefMap{ByRef: map[string]string{}}},
	}
	panel := FormatPanel(PanelInput{Result: res})
	if !strings.Contains(panel, "warn: Excluded 2 protected") {
		t.Fatalf("panel lost the warning:\n%s", panel)
	}
}

func TestCompressToolSchemaContentHasNoArrayType(t *testing.T) {
	s := CompressToolSchema()
	props := s["properties"].(map[string]any)
	content := props["content"].(map[string]any)
	if _, has := content["type"]; has {
		t.Fatal(`content must NOT declare "type": "array" — providers without strict tool support stringify nested arrays, ` +
			`and a typed schema rejects that shape before the lenient parser can repair it`)
	}
	if _, has := content["items"]; !has {
		t.Fatal("content must still describe its items")
	}
	req := s["required"].([]any)
	if len(req) != 1 || req[0] != "content" {
		t.Fatalf("required = %v, want [content]", req)
	}
}

// The description ships with every request; without the brake the model may
// call Compress unprompted and write a summary with no rules to follow.
func TestCompressToolDescriptionCarriesBrake(t *testing.T) {
	if !strings.Contains(CompressToolDescription, "Do NOT call this on your own initiative") {
		t.Fatalf("description is missing the brake: %q", CompressToolDescription)
	}
}
