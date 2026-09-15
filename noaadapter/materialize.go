package noaadapter

import (
	"time"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
)

// Materialize applies a session's compression state to a message array so the
// session can continue with noa switched off.
//
// Why this is needed: l.messages is an in-memory working copy rebuilt from the
// transcript on every resume, and the transcript holds the FULL history. With
// noa off, View no longer runs, so llm.MessagesForAPI would send all of it and
// overflow immediately.
//
// It is a pure function — no transcript write, no state mutation, no trace —
// which buys three properties:
//
//   - Idempotent: same transcript + same state, same output, every resume.
//   - Reversible: turning noa back on costs nothing, because the block ledger
//     was never touched.
//   - The transcript stays exactly what it is: a record of what happened, with
//     no derived content mixed in for later readers to have to distinguish.
//
// The archive pointers inside the summaries keep working: they are plain paths
// to files that are still on disk, readable with the ordinary Read tool. Being
// able to follow a compression back to its originals does not depend on any
// noa code still running.
//
// No state and no active blocks means nothing to do, so calling this
// unconditionally is safe.
func Materialize(msgs []llm.Message, o Options) ([]llm.Message, error) {
	if o.ArchiveBaseDir == "" {
		return msgs, ErrNoArchiveDir
	}
	sess, err := newSession(o)
	if err != nil {
		return msgs, err
	}
	state := sess.State()
	if len(noa.ActiveBlocks(state)) == 0 {
		return msgs, nil
	}

	cores, sc := Project(msgs)
	pruned, _ := noa.RunPruneOnly(cores, state, sess.Config())

	// No ref tags: they address a scheme that is no longer running, and would
	// only invite the model to use ids nothing will resolve.
	return Reassemble(pruned, msgs, sc, ReassembleOptions{Tag: false}), nil
}

// RebuildFromHistory reconstructs compression state by replaying the Compress
// calls recorded in a conversation.
//
// It is the fallback when state.json is missing — imported, corrupted, or never
// written. No model is involved: the summaries live in the recorded call
// arguments, so replaying returns the byte-identical text rather than a fresh
// approximation of it.
//
// Existing archives are adopted rather than rewritten. They are historical fact;
// a re-render would be a different file describing the same moment.
func RebuildFromHistory(msgs []llm.Message, o Options) (*Session, error) {
	sess, err := newSession(o)
	if err != nil {
		return nil, err
	}
	cores, _ := Project(msgs)
	now := sess.now()

	res := noa.RebuildStateFromLog(noa.RebuildInput{
		Messages:    cores,
		Config:      sess.Config(),
		Archiver:    NewReuseArchiver(sess.ArchiveRoot()),
		SessionID:   o.SessionID,
		ArchiveRoot: sess.ArchiveRoot(),
		CreatedAt:   now.Format(time.RFC3339),
		Now:         now.Unix(),
	})
	for _, w := range res.Warnings {
		sess.warn("rebuild: " + w)
	}

	sess.mu.Lock()
	sess.state = res.State
	sess.mu.Unlock()
	if err := sess.store.Save(res.State); err != nil {
		sess.warn("rebuild: state save failed: " + err.Error())
	}
	return sess, nil
}
