package noa

import (
	"slices"
	"strings"
)

// IsMessageProtected reports whether a message is HARD-protected: excluded from
// every compression range, and never assigned a ref.
//
// Distinct from the soft protected zone (the most recent N messages / last user
// message), which is a recency heuristic that shifts as the conversation grows.
func IsMessageProtected(m CoreMessage, cfg Config) bool {
	if m.ToolName == "" {
		return false
	}
	if slices.Contains(AlwaysProtectedTools, m.ToolName) {
		return true
	}
	for _, pat := range cfg.ProtectedTools {
		if MatchToolPattern(pat, m.ToolName) {
			return true
		}
	}
	if cfg.IsToolProtected != nil && cfg.IsToolProtected(m.ToolName) {
		return true
	}
	return false
}

// MatchToolPattern matches a tool name against a configured pattern: exact
// (case-insensitive), or a single trailing "*" prefix wildcard.
func MatchToolPattern(pattern, toolName string) bool {
	if pattern == "" || toolName == "" {
		return false
	}
	p, n := strings.ToLower(pattern), strings.ToLower(toolName)
	if prefix, ok := strings.CutSuffix(p, "*"); ok {
		return strings.HasPrefix(n, prefix)
	}
	return p == n
}

// IsNeverPreserveRecentTool reports whether a message's tool is excluded from
// the "most recent N" protected-zone accounting. Such messages remain fully
// compressible — they simply do not occupy a slot.
func IsNeverPreserveRecentTool(m CoreMessage) bool {
	if m.ToolName == "" {
		return false
	}
	for _, name := range NeverPreserveRecentTools {
		if strings.EqualFold(m.ToolName, name) {
			return true
		}
	}
	return false
}

// collectProtectedToolCallIDs gathers the tool_call ids of every hard-protected
// message, so the paired half can be excluded too.
func collectProtectedToolCallIDs(messages []CoreMessage, cfg Config) map[string]bool {
	ids := map[string]bool{}
	for _, m := range messages {
		if m.ToolCallID != "" && IsMessageProtected(m, cfg) {
			ids[m.ToolCallID] = true
		}
	}
	return ids
}

// isMessageProtectedWithPairing extends hard protection to the other half of a
// protected tool exchange: a tool_result whose tool_call is protected must be
// excluded too, or the request would be rebuilt with an unpaired block.
func isMessageProtectedWithPairing(m CoreMessage, cfg Config, protectedCallIDs map[string]bool) bool {
	if IsMessageProtected(m, cfg) {
		return true
	}
	return m.ToolCallID != "" && protectedCallIDs[m.ToolCallID]
}
