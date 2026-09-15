// Package noaadapter wires the pure noa engine into Norma: it projects
// llm.Message into noa's flat representation and back, persists state and
// archives, and exposes the Compress tool.
//
// See docs/noa-实施方案.md for the specification.
package noaadapter

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/Autumn-27/norma/noa"
)

// DeriveMessageID computes a message's stable identity.
//
// llm.Message has no id field, so noa derives one from the content. A content
// hash rather than an array index is not a stylistic choice: indices shift
// whenever a message is pruned, a nudge is injected, or a session resumes, and
// every ref in the map would break at once. A hash stays put.
//
// text must already have any ref tag stripped. A tag inside the hash would make
// the id change every time the tag is re-rendered, which is every turn.
func DeriveMessageID(role noa.Role, ct noa.ContentType, text, toolCallID, toolName string) string {
	seed := string(role) + "|" + string(ct) + "|" + toolCallID + "|" + toolName + "|" + text
	sum := sha256.Sum256([]byte(seed))
	return "h_" + hex.EncodeToString(sum[:])[:16]
}

// ClusterCounter disambiguates messages that hash identically.
//
// Repeating a command produces byte-identical output, and two messages sharing
// an id would collapse into one in the ref map. The counter appends an
// occurrence suffix so each keeps its own identity, while the first occurrence
// keeps the bare hash — so an unchanged history projects to unchanged ids.
type ClusterCounter struct {
	counts map[string]int
}

// NewClusterCounter returns a counter for one projection pass. It must be fresh
// per pass: reusing one across passes would give the same message a different
// suffix each time.
func NewClusterCounter() *ClusterCounter {
	return &ClusterCounter{counts: map[string]int{}}
}

// Next returns the id to use for the next occurrence of base.
func (c *ClusterCounter) Next(base string) string {
	n := c.counts[base]
	c.counts[base] = n + 1
	if n == 0 {
		return base
	}
	return fmt.Sprintf("%s_%d", base, n)
}
