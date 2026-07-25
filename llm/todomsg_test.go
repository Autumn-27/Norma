package llm

import (
	"strings"
	"testing"
)

func TestTodoReminderMessageRoundTrip(t *testing.T) {
	m := TodoReminderMessage("1. [in_progress] build\n2. [pending] test")
	if m.Role != RoleUser {
		t.Fatalf("todo reminder must be a user message, got %q", m.Role)
	}
	if !strings.HasPrefix(m.Text(), "<system-reminder>") {
		t.Fatalf("must be wrapped in <system-reminder>: %q", m.Text())
	}
	if !strings.Contains(m.Text(), "build") || !strings.Contains(m.Text(), "test") {
		t.Fatalf("list content missing: %q", m.Text())
	}
	if !IsTodoReminder(m) {
		t.Fatal("IsTodoReminder should detect its own message")
	}
}

func TestIsTodoReminderNoFalsePositive(t *testing.T) {
	// Ordinary user text, a plain system-reminder, and an assistant message must
	// not be mistaken for a todo reminder.
	cases := []Message{
		UserText("please update the todo list"),
		UserText("<system-reminder>\n# currentDate\nToday is 2026-01-01.\n</system-reminder>"),
		{Role: RoleAssistant, Content: []ContentBlock{TextBlock("<system-reminder>...</system-reminder>")}},
	}
	for i, m := range cases {
		if IsTodoReminder(m) {
			t.Fatalf("case %d falsely detected as todo reminder: %q", i, m.Text())
		}
	}
}
