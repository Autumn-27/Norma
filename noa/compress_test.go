package noa

import (
	"strings"
	"testing"
)

const longSummary = "Explored the authentication design and settled on JWT with refresh tokens. " +
	"Key files: auth/jwt.go (issue/verify), auth/middleware.go (context injection). " +
	"Refresh tokens live in Redis with a 7-day TTL; access tokens are 15 minutes and are not persisted."

// fixture builds a view with `filler` compressible assistant messages between a
// leading user message and a tail large enough to absorb the protected zone.
//
// The tail must satisfy BOTH protections or the filler is not compressible:
// PreserveRecentMessages (5 messages) and PreserveRecentTokens (5000 tokens).
// Six tail messages of 4000 chars each carry 1000 tokens apiece, so the token
// accumulator stops inside the tail and never reaches the filler.
func fixture(t *testing.T, filler int, fillerChars int) ([]CoreMessage, CompressionState) {
	t.Helper()
	body := strings.Repeat("x", fillerChars)
	msgs := []CoreMessage{msg("u0", RoleUser, CTText, "the task")}
	for i := range filler {
		msgs = append(msgs, msg(idFor(i), RoleAssistant, CTText, body))
	}
	tailBody := strings.Repeat("t", 4000)
	for i := range 6 {
		msgs = append(msgs, msg("tail"+idFor(i), RoleUser, CTText, tailBody))
	}
	st := CreateInitialState("sess", "/archive")
	res := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})
	return res.Messages, res.State
}

func idFor(i int) string { return string(rune('a'+i%26)) + strings.Repeat("'", i/26) }

func applyOne(t *testing.T, msgs []CoreMessage, st CompressionState, a Archiver, r CompressRange) ApplyResult {
	t.Helper()
	return ApplyCompression(ApplyInput{
		Ranges: []CompressRange{r}, Messages: msgs, State: st,
		Config: DefaultConfig(200000), CallID: "toolu_1", Archiver: a,
		CreatedAt: "2026-09-14T10:00:00+08:00", Now: 1789000000,
	})
}

func TestApplyCompressionCreatesBlockAndArchive(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	arch := newMemArchiver()
	res := applyOne(t, msgs, st, arch, CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: longSummary, Topic: "JWT 认证方案",
	})
	if len(res.Errors) != 0 {
		t.Fatalf("errors = %v", res.Errors)
	}
	if len(res.BlocksCreated) != 1 {
		t.Fatalf("BlocksCreated = %d, want 1", len(res.BlocksCreated))
	}
	b := res.BlocksCreated[0]
	if b.Tier != 1 {
		t.Fatalf("Tier = %d, want 1 for a pure message range", b.Tier)
	}
	if b.ArchivePath == "" || b.ArchiveRel == "" {
		t.Fatalf("block has no archive path: %+v", b)
	}
	if arch.count() != 1 {
		t.Fatalf("archiver holds %d files, want 1", arch.count())
	}
	body := arch.content(b.ArchivePath)
	if !strings.Contains(body, "block: "+b.BlockID) || !strings.Contains(body, "tier: 1") {
		t.Fatalf("archive front matter missing:\n%s", body[:min(400, len(body))])
	}
	if !strings.Contains(body, "topic: JWT 认证方案") {
		t.Fatal("archive front matter lost the topic")
	}
	if res.State.Stats.CompressionCount != 1 {
		t.Fatalf("CompressionCount = %d, want 1", res.State.Stats.CompressionCount)
	}
}

// The archive must hold the originals verbatim — it is the only path back.
func TestApplyCompressionArchivesOriginalText(t *testing.T) {
	msgs := []CoreMessage{
		msg("u0", RoleUser, CTText, "the task"),
		msg("big", RoleAssistant, CTText, strings.Repeat("unique-content ", 500)),
	}
	tailBody := strings.Repeat("t", 4000)
	for i := range 6 {
		msgs = append(msgs, msg("t"+idFor(i), RoleUser, CTText, tailBody))
	}
	st := CreateInitialState("sess", "/archive")
	pt := ProcessTurn(ProcessTurnInput{Messages: msgs, State: st, Config: DefaultConfig(200000)})

	arch := newMemArchiver()
	res := applyOne(t, pt.Messages, pt.State, arch, CompressRange{
		StartRef: "m00002", EndRef: "m00002", Summary: longSummary,
	})
	if len(res.Errors) != 0 {
		t.Fatalf("errors = %v", res.Errors)
	}
	body := arch.content(res.BlocksCreated[0].ArchivePath)
	if strings.Count(body, "unique-content") != 500 {
		t.Fatalf("archive holds %d occurrences of the original text, want all 500 — archives are never truncated",
			strings.Count(body, "unique-content"))
	}
}

// A failed archive write must leave no trace: no block, no state change, and
// the archives already written this batch rolled back.
func TestApplyCompressionRollsBackOnArchiveFailure(t *testing.T) {
	msgs, st := fixture(t, 8, 2000)
	arch := newMemArchiver()
	arch.failOn = "b2"

	res := ApplyCompression(ApplyInput{
		Ranges: []CompressRange{
			{StartRef: "m00002", EndRef: "m00004", Summary: longSummary},
			{StartRef: "m00005", EndRef: "m00007", Summary: longSummary},
		},
		Messages: msgs, State: st, Config: DefaultConfig(200000),
		CallID: "toolu_1", Archiver: arch,
	})
	if len(res.BlocksCreated) != 0 {
		t.Fatalf("BlocksCreated = %d, want 0 — a failed archive must abort the batch", len(res.BlocksCreated))
	}
	if len(res.Errors) == 0 || !strings.Contains(strings.Join(res.Errors, " "), "Failed to archive") {
		t.Fatalf("errors = %v, want one reporting the archive failure", res.Errors)
	}
	if arch.count() != 0 {
		t.Fatalf("archiver still holds %d files, want the batch rolled back", arch.count())
	}
	if len(res.State.Blocks) != 0 {
		t.Fatalf("state gained %d blocks despite the failure", len(res.State.Blocks))
	}
}

func TestApplyCompressionRejectsSmallBatchAtomically(t *testing.T) {
	msgs, st := fixture(t, 4, 50) // far below MinCompressRange
	arch := newMemArchiver()
	res := applyOne(t, msgs, st, arch, CompressRange{
		StartRef: "m00002", EndRef: "m00003", Summary: longSummary,
	})
	if len(res.BlocksCreated) != 0 {
		t.Fatalf("BlocksCreated = %d, want 0", len(res.BlocksCreated))
	}
	if len(res.Errors) != 1 || !strings.Contains(res.Errors[0], "too small") {
		t.Fatalf("errors = %v, want the size-gate rejection", res.Errors)
	}
	if arch.count() != 0 {
		t.Fatal("a rejected batch must not write archives")
	}
}

func TestApplyCompressionUnknownRefDiagnostics(t *testing.T) {
	msgs, st := fixture(t, 4, 50)
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m09000", EndRef: "m09001", Summary: longSummary,
	})
	joined := strings.Join(res.Errors, "\n")
	if !strings.Contains(joined, "unknown to this session") {
		t.Fatalf("errors = %v, want the all-unknown rejection", res.Errors)
	}
	// The model must be told refs are never reassigned, or it will assume a
	// previous compression invalidated them and stop using refs it holds.
	if !strings.Contains(joined, "reassign") {
		t.Fatalf("rejection text must state that refs are never reassigned: %q", joined)
	}
	if !strings.Contains(joined, "diagnostics:") {
		t.Fatalf("rejection text is missing the diagnostics block: %q", joined)
	}
}

func TestApplyCompressionSkipsOverlappingRanges(t *testing.T) {
	msgs, st := fixture(t, 10, 2000)
	arch := newMemArchiver()
	res := ApplyCompression(ApplyInput{
		Ranges: []CompressRange{
			{StartRef: "m00002", EndRef: "m00006", Summary: longSummary},
			{StartRef: "m00004", EndRef: "m00008", Summary: longSummary}, // overlaps
		},
		Messages: msgs, State: st, Config: DefaultConfig(200000),
		CallID: "toolu_1", Archiver: arch,
	})
	if len(res.BlocksCreated) != 1 {
		t.Fatalf("BlocksCreated = %d, want 1 — the overlapping range must be skipped", len(res.BlocksCreated))
	}
	if !strings.Contains(strings.Join(res.Warnings, "\n"), "overlaps an earlier range") {
		t.Fatalf("warnings = %v, want the overlap warning", res.Warnings)
	}
}

func TestApplyCompressionSummaryValidation(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	cases := []struct {
		name    string
		summary string
		maxChar *int
		want    string
	}{
		{"empty", "", nil, "Summary is empty"},
		{"too short", "brief", nil, "too short"},
		{"too long", strings.Repeat("x", 25000), nil, "too long"},
	}
	for _, c := range cases {
		res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
			StartRef: "m00002", EndRef: "m00005", Summary: c.summary, SummaryMaxChars: c.maxChar,
		})
		if len(res.Errors) == 0 || !strings.Contains(strings.Join(res.Errors, " "), c.want) {
			t.Errorf("%s: errors = %v, want one containing %q", c.name, res.Errors, c.want)
		}
	}
}

func TestApplyCompressionSummaryMaxCharsOverride(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	big := 30000
	long := strings.Repeat("y", 25000)
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: long, SummaryMaxChars: &big,
	})
	if len(res.Errors) != 0 {
		t.Fatalf("errors = %v, want summaryMaxChars to raise the ceiling", res.Errors)
	}
}

// Compressing into the recent tail must be refused, not silently trimmed to
// nothing: the model needs to know the zone exists.
func TestApplyCompressionRejectsProtectedZoneOnlyRange(t *testing.T) {
	msgs, st := fixture(t, 2, 3000)
	arch := newMemArchiver()
	// The last refs are inside the protected tail.
	last := HighestUsedIndex(st.MessageRefs)
	res := applyOne(t, msgs, st, arch, CompressRange{
		StartRef: IndexToRef(last - 1), EndRef: IndexToRef(last), Summary: longSummary,
	})
	joined := strings.Join(res.Errors, "\n")
	if !strings.Contains(joined, "protected zone") && !strings.Contains(joined, "too small") {
		t.Fatalf("errors = %v, want the range refused", res.Errors)
	}
	if len(res.BlocksCreated) != 0 {
		t.Fatal("a protected-zone-only range must create no block")
	}
}

func TestApplyCompressionResetsNudgeBaselines(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	st.Nudge.LastPerMessageNudgeTokens = 120000
	st.Nudge.LastNudgeShownTokens = 118000
	st.Nudge.LastShownByTier = map[Tier]int{2: 100000}

	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: longSummary,
	})
	if len(res.BlocksCreated) == 0 {
		t.Fatalf("no block created: %v", res.Errors)
	}
	n := res.State.Nudge
	if n.LastPerMessageNudgeTokens != 0 || n.LastNudgeShownTokens != 0 || len(n.LastShownByTier) != 0 {
		t.Fatalf("nudge baselines = %+v, want all cleared — otherwise the next turn replays the nudge the model just acted on", n)
	}
}

// The block reports what it actually covers, not what was asked for.
func TestApplyCompressionSpanReflectsActualCoverage(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	res := applyOne(t, msgs, st, newMemArchiver(), CompressRange{
		StartRef: "m00002", EndRef: "m00005", Summary: longSummary,
	})
	if len(res.BlocksCreated) == 0 {
		t.Fatalf("no block: %v", res.Errors)
	}
	b := res.BlocksCreated[0]
	lo, hi := refSpan(b.EffectiveMessageIDs, res.State.MessageRefs)
	if b.StartRef != lo || b.EndRef != hi {
		t.Fatalf("span = %s..%s, want it derived from coverage (%s..%s)", b.StartRef, b.EndRef, lo, hi)
	}
}

func TestApplyCompressionNoArchiverIsRefused(t *testing.T) {
	msgs, st := fixture(t, 6, 2000)
	res := ApplyCompression(ApplyInput{
		Ranges:   []CompressRange{{StartRef: "m00002", EndRef: "m00005", Summary: longSummary}},
		Messages: msgs, State: st, Config: DefaultConfig(200000),
	})
	if len(res.Errors) == 0 || !strings.Contains(res.Errors[0], "archiver") {
		t.Fatalf("errors = %v, want a refusal to compress without an archiver", res.Errors)
	}
}

func TestApplyCompressionEmptyRangesIsNoop(t *testing.T) {
	msgs, st := fixture(t, 4, 2000)
	res := ApplyCompression(ApplyInput{Messages: msgs, State: st, Config: DefaultConfig(200000), Archiver: newMemArchiver()})
	if len(res.Errors) != 0 || len(res.BlocksCreated) != 0 {
		t.Fatalf("empty ranges: errors=%v blocks=%d, want a clean no-op", res.Errors, len(res.BlocksCreated))
	}
}
