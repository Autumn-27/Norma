package noa

import (
	"fmt"
	"strings"
)

// Archiver writes a block's original content to durable storage.
//
// The interface keeps this package pure: noa decides WHAT to archive and
// renders it, the host decides WHERE and performs the I/O. Unit tests use an
// in-memory implementation.
type Archiver interface {
	// Write stores content for a block and returns its absolute path and the
	// path relative to the archive root. It must be atomic: a partially written
	// archive is worse than none, because the block that points at it will claim
	// the originals are recoverable.
	Write(tier Tier, blockID, startRef, endRef string, content []byte) (abs, rel string, err error)
	// Remove deletes an archive, used to roll back a failed batch.
	Remove(abs string) error
}

// ArchiveEntry is one item in a tier-2/3 archive: either an absorbed block's
// summary, or a loose message that was swept up with it.
type ArchiveEntry struct {
	// Block is set for an absorbed lower-tier block.
	Block *CompressionBlock
	// Message is set for an uncompressed message inside the range.
	Message *CoreMessage
	// Ref is the message's ref, when it has one.
	Ref string
}

// ArchiveRenderInput is everything needed to render one archive file.
type ArchiveRenderInput struct {
	BlockID   string
	Tier      Tier
	SessionID string
	// CreatedAt is an RFC3339 timestamp supplied by the caller — this package
	// does not read the clock.
	CreatedAt string
	StartRef  string
	EndRef    string
	Topic     string
	Summary   string
	// Entries are in VIEW ORDER: absorbed block summaries and loose messages
	// interleaved exactly as the model saw them, so a reader sees the shape of
	// the conversation rather than a regrouped digest.
	Entries []ArchiveEntry
	// OriginalTokens is the cost of what this archive replaces.
	OriginalTokens int
}

// RenderArchive produces the markdown for one archive file.
//
// Content is written verbatim — no truncation, ever. Tool output is already
// bounded upstream by the host's capture layer, and the archive is the only
// path back to the originals: shortening it here would make the guarantee a
// lie.
func RenderArchive(in ArchiveRenderInput) []byte {
	var b strings.Builder

	b.WriteString("---\n")
	fmt.Fprintf(&b, "block: %s\n", in.BlockID)
	fmt.Fprintf(&b, "tier: %d\n", in.Tier)
	if in.SessionID != "" {
		fmt.Fprintf(&b, "session: %s\n", in.SessionID)
	}
	if in.CreatedAt != "" {
		fmt.Fprintf(&b, "created: %s\n", in.CreatedAt)
	}
	fmt.Fprintf(&b, "range: %s-%s\n", in.StartRef, in.EndRef)

	msgCount, blockCount := 0, 0
	var consumed []string
	for _, e := range in.Entries {
		switch {
		case e.Block != nil:
			blockCount++
			consumed = append(consumed, e.Block.BlockID)
		case e.Message != nil:
			msgCount++
		}
	}
	if blockCount > 0 {
		fmt.Fprintf(&b, "consumed_blocks: [%s]\n", strings.Join(consumed, ", "))
		fmt.Fprintf(&b, "loose_messages: %d\n", msgCount)
	} else {
		fmt.Fprintf(&b, "message_count: %d\n", msgCount)
	}
	fmt.Fprintf(&b, "original_tokens: %d\n", in.OriginalTokens)
	if in.Topic != "" {
		fmt.Fprintf(&b, "topic: %s\n", in.Topic)
	}
	b.WriteString("---\n\n")

	fmt.Fprintf(&b, "# 归档 %s · Tier %d · %s–%s\n\n", in.BlockID, in.Tier, in.StartRef, in.EndRef)
	if s := strings.TrimSpace(in.Summary); s != "" {
		b.WriteString("**摘要**（当前在上下文中生效的版本）：\n\n")
		for _, line := range strings.Split(s, "\n") {
			b.WriteString("> ")
			b.WriteString(line)
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	b.WriteString("---\n\n")

	// Tier 1 archives hold raw messages at level 2; tier 2/3 archives hold block
	// entries at level 2 and demote loose messages to level 3, so the hierarchy
	// stays readable.
	msgLevel := "##"
	if blockCount > 0 {
		msgLevel = "###"
	}
	for _, e := range in.Entries {
		switch {
		case e.Block != nil:
			writeBlockEntry(&b, *e.Block)
		case e.Message != nil:
			writeMessageEntry(&b, *e.Message, e.Ref, msgLevel)
		}
	}
	return []byte(b.String())
}

// writeBlockEntry renders an absorbed block: its summary plus both forms of the
// path to its own archive. The relative one is for a human following links; the
// absolute one is what a Read tool needs.
func writeBlockEntry(b *strings.Builder, blk CompressionBlock) {
	fmt.Fprintf(b, "## %s · Tier %d · %s–%s\n\n", blk.BlockID, blk.Tier, blk.StartRef, blk.EndRef)
	if blk.Topic != "" {
		fmt.Fprintf(b, "**%s**\n\n", blk.Topic)
	}
	if s := strings.TrimSpace(blk.Summary); s != "" {
		b.WriteString(s)
		b.WriteString("\n\n")
	}
	if blk.ArchiveRel != "" {
		fmt.Fprintf(b, "> 下级归档：[`%s`](%s)\n", blk.ArchiveRel, relLink(blk.ArchiveRel))
	}
	if blk.ArchivePath != "" {
		fmt.Fprintf(b, "> 绝对路径：`%s`\n", blk.ArchivePath)
	}
	b.WriteString("\n")
}

// relLink turns "tier1/b1_x.md" into a link that works from a tier2/ file.
func relLink(rel string) string {
	if strings.Contains(rel, "/") {
		return "../" + rel
	}
	return rel
}

// writeMessageEntry renders one original message.
func writeMessageEntry(b *strings.Builder, m CoreMessage, ref, level string) {
	label := ref
	if label == "" {
		label = "(unaddressed)"
	}
	switch m.ContentType {
	case CTReasoning:
		fmt.Fprintf(b, "%s %s · %s · thinking\n\n", level, label, m.Role)
		b.WriteString(m.Text)
		b.WriteString("\n\n")
	case CTToolCall:
		fmt.Fprintf(b, "%s %s · %s · tool_use · %s\n\n", level, label, m.Role, m.ToolName)
		writeFenced(b, m.Text, "json")
	case CTToolResult:
		name := m.ToolName
		if name == "" {
			name = "tool"
		}
		fmt.Fprintf(b, "%s %s · tool_result · %s\n\n", level, label, name)
		writeFenced(b, m.Text, "")
	default:
		fmt.Fprintf(b, "%s %s · %s\n\n", level, label, m.Role)
		b.WriteString(m.Text)
		b.WriteString("\n\n")
	}
}

// writeFenced wraps body in a code fence long enough to survive any fence the
// body itself contains — tool output frequently includes markdown.
func writeFenced(b *strings.Builder, body, lang string) {
	fence := "```"
	for strings.Contains(body, fence) {
		fence += "`"
	}
	b.WriteString(fence)
	b.WriteString(lang)
	b.WriteString("\n")
	b.WriteString(body)
	if !strings.HasSuffix(body, "\n") {
		b.WriteString("\n")
	}
	b.WriteString(fence)
	b.WriteString("\n\n")
}

// ArchiveFileName is the archive's basename: block id first so it is unique,
// then the ref span so a human can locate it by eye.
func ArchiveFileName(blockID, startRef, endRef string) string {
	return fmt.Sprintf("%s_%s-%s.md", blockID, startRef, endRef)
}

// ArchiveTierDir is the subdirectory a tier's archives live in.
func ArchiveTierDir(t Tier) string { return fmt.Sprintf("tier%d", t) }
