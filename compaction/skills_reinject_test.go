package compaction

import (
	"strings"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

// With many skills behind the boundary whose combined size exceeds the total
// re-injection budget (25000 tokens), the newest fit and the oldest are dropped —
// exercising the candidate/kept sort paths and the over-budget branch.
func TestReinjectSkillsBudgetDropsOldest(t *testing.T) {
	const n = 8
	var msgs []llm.Message
	for i := range n {
		// Each body is large enough that truncation caps it near the per-skill cap,
		// so several skills together blow the total budget.
		body := strings.Repeat("x", 24000)
		msgs = append(msgs, llm.SkillInvocationMessage("skill"+string(rune('A'+i)), "", body))
	}
	msgs = append(msgs, llm.UserText("tail")) // preserved tail
	tailStart := n                            // all skills are behind the boundary

	got := reinjectSkills(msgs, tailStart)
	if len(got) < 2 {
		t.Fatalf("expected multiple skills re-injected, got %d", len(got))
	}
	if len(got) >= n {
		t.Fatalf("total budget should drop at least one skill, kept %d of %d", len(got), n)
	}
	// Kept skills stay in chronological (idx-ascending) order.
	for i := 1; i < len(got); i++ {
		if pos(got[i]) < pos(got[i-1]) {
			t.Fatal("re-injected skills must be in chronological order")
		}
	}
	// The oldest skill (skillA) is the one dropped when over budget.
	for _, m := range got {
		if name, _ := llm.SkillInvocationName(m); name == "skillA" {
			t.Fatal("oldest skill should be dropped first when over budget")
		}
	}
}

// pos returns a stable ordering key from a skill message's body letter (the tests
// build bodies that keep their name letter), used only to check ordering.
func pos(m llm.Message) int {
	name, _ := llm.SkillInvocationName(m)
	if name == "" {
		return 0
	}
	return int(name[len(name)-1])
}

// A single skill under budget is re-injected as-is (baseline for the budget path).
func TestReinjectSkillsSingleFits(t *testing.T) {
	msgs := []llm.Message{
		llm.SkillInvocationMessage("solo", "/s", "1. do a thing"),
		llm.UserText("tail"),
	}
	got := reinjectSkills(msgs, 1)
	if len(got) != 1 {
		t.Fatalf("want 1 re-injected skill, got %d", len(got))
	}
	if name, _ := llm.SkillInvocationName(got[0]); name != "solo" {
		t.Fatalf("want solo, got %q", name)
	}
}
