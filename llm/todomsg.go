package llm

import "strings"

// Todo-reminder messages re-surface the current todo list as standing guidance
// when the model hasn't touched TodoWrite in a while. Like the skill-invocation
// message, it is a self-identifying <system-reminder> user message: the harness
// derives its throttle ("turns since last reminder") by scanning history for it,
// and it is regenerated from live state each eligible turn so it survives
// compaction. Mirrors claude-code's todo_reminder attachment.

// todoReminderLead is the fixed nag text — also the detection sentinel (a normal
// user message is exceedingly unlikely to contain this exact system phrasing).
const todoReminderLead = "The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if it has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user."

// TodoReminderMessage builds the <system-reminder> user message carrying the
// current todo list. listText is the rendered list (see tool.RenderTodoReminder).
func TodoReminderMessage(listText string) Message {
	var b strings.Builder
	b.WriteString("<system-reminder>\n")
	b.WriteString(todoReminderLead)
	b.WriteString("\n\nHere are the existing contents of your todo list:\n\n")
	b.WriteString(listText)
	b.WriteString("\n</system-reminder>")
	return Message{Role: RoleUser, Content: []ContentBlock{TextBlock(b.String())}}
}

// IsTodoReminder reports whether m is a todo-reminder message (used to derive the
// throttle "turns since last reminder").
func IsTodoReminder(m Message) bool {
	if m.Role != RoleUser || len(m.Content) != 1 || m.Content[0].Type != BlockText {
		return false
	}
	t := m.Content[0].Text
	return strings.HasPrefix(t, "<system-reminder>") && strings.Contains(t, todoReminderLead)
}
