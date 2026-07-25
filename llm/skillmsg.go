package llm

import "strings"

// Skill-invocation messages carry an invoked skill's full instructions in the
// conversation as an independent user message (not a tool_result), wrapped in a
// <system-reminder> so the model reads them as standing guidance. They are
// self-identifying via the <invoked-skill …> envelope so compaction can find the
// latest invocation of each skill in the retained (non-destructive) history and
// re-inject it verbatim after a summary boundary — keeping skill instructions
// alive across compactions even though the transcript copy gets summarized.

const (
	skillReminderLead = "The following skill was invoked in this session. Continue to follow these guidelines:"
	skillOpenPrefix   = "<invoked-skill name=\""
	skillFooter       = "\n</invoked-skill>\n</system-reminder>"
	skillTruncNote    = "\n…[skill instructions truncated to save context — re-read the skill's base directory for the full text]"
)

// SkillInvocationMessage builds the independent user message that carries a
// skill's instructions. name is required; path (the skill's base directory) is
// optional and, when present, is surfaced both as an attribute (for detection)
// and as a human-readable "Base directory:" line. body is the instructions plus
// any caller-supplied context.
func SkillInvocationMessage(name, path, body string) Message {
	var b strings.Builder
	b.WriteString("<system-reminder>\n")
	b.WriteString(skillReminderLead)
	b.WriteString("\n\n")
	b.WriteString(skillOpenPrefix)
	b.WriteString(name)
	b.WriteString("\"")
	if path != "" {
		b.WriteString(" path=\"")
		b.WriteString(path)
		b.WriteString("\"")
	}
	b.WriteString(">\n")
	if path != "" {
		b.WriteString("Base directory: ")
		b.WriteString(path)
		b.WriteString("\n\n")
	}
	b.WriteString(body)
	b.WriteString(skillFooter)
	return Message{Role: RoleUser, Content: []ContentBlock{TextBlock(b.String())}}
}

// SkillInvocationName reports whether m is a skill-invocation message and, if so,
// returns the invoked skill's name (used by compaction to dedupe to the latest
// invocation per skill).
func SkillInvocationName(m Message) (string, bool) {
	if m.Role != RoleUser || len(m.Content) != 1 || m.Content[0].Type != BlockText {
		return "", false
	}
	_, rest, ok := strings.Cut(m.Content[0].Text, skillOpenPrefix)
	if !ok {
		return "", false
	}
	name, _, ok := strings.Cut(rest, "\"")
	if !ok || name == "" {
		return "", false
	}
	return name, true
}

// TruncateSkillMessage returns m with its instructions body clipped so the whole
// message stays within roughly maxChars, preserving the envelope tags and
// appending a truncation note. Messages already within budget are returned
// unchanged.
func TruncateSkillMessage(m Message, maxChars int) Message {
	if maxChars <= 0 || len(m.Content) != 1 || m.Content[0].Type != BlockText {
		return m
	}
	text := m.Content[0].Text
	if len(text) <= maxChars {
		return m
	}
	head := text
	if k := strings.LastIndex(text, skillFooter); k >= 0 {
		head = text[:k] // header + open tag + body, without the footer
	}
	if len(head) > maxChars {
		head = head[:maxChars]
	}
	return Message{Role: m.Role, Content: []ContentBlock{TextBlock(head + skillTruncNote + skillFooter)}}
}
