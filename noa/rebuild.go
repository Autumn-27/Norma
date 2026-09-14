package noa

// RebuildInput replays a session's compressions from its recorded history.
type RebuildInput struct {
	// Messages is the full projected history, oldest first.
	Messages []CoreMessage
	// Config and Archiver mirror the live ones.
	Config   Config
	Archiver Archiver
	// SessionID and ArchiveRoot seed the rebuilt state.
	SessionID   string
	ArchiveRoot string
	CountTokens TokenCountFn
	// CreatedAt and Now stamp rebuilt blocks; this package does not read the
	// clock.
	CreatedAt string
	Now       int64
}

// RebuildResult reports the replay.
type RebuildResult struct {
	State CompressionState
	// Replayed is how many compressions were reapplied.
	Replayed int
	// Skipped is how many recorded calls were passed over (they failed then, and
	// replaying them would fail identically).
	Skipped  int
	Warnings []string
}

// RebuildStateFromLog reconstructs compression state from the conversation
// itself.
//
// This works — and needs no model — because the summaries were never derived
// from anything: the model wrote them as arguments to its Compress calls, and
// the host recorded those calls verbatim. Replaying the calls returns the
// BYTE-IDENTICAL summaries, not an approximation of them.
//
// That is the practical difference from summarise-on-demand compaction, whose
// summary is a model response: lose it and it must be regenerated, and the
// regenerated text differs.
//
// Each call is replayed against the view AS IT WAS at that point, so refs
// resolve the way they did originally.
func RebuildStateFromLog(in RebuildInput) RebuildResult {
	res := RebuildResult{State: CreateInitialState(in.SessionID, in.ArchiveRoot)}

	// A call whose result reported failure is skipped: it created nothing then
	// and would create nothing now.
	succeeded := map[string]bool{}
	for _, m := range in.Messages {
		if m.ContentType != CTToolResult || m.ToolName != CompressToolName || m.ToolCallID == "" {
			continue
		}
		succeeded[m.ToolCallID] = PanelBlockCount(m.Text) > 0
	}

	for i, m := range in.Messages {
		if m.ContentType != CTToolCall || m.ToolName != CompressToolName || m.ToolCallID == "" {
			continue
		}
		if !succeeded[m.ToolCallID] {
			res.Skipped++
			continue
		}
		parsed := ParseCompressArgs([]byte(m.Text))
		if len(parsed.Ranges) == 0 {
			res.Skipped++
			continue
		}

		// Replay against the history up to and including this call. The refs in
		// the arguments were resolved against that view, so anything later would
		// shift what they point at.
		prefix := in.Messages[:i+1]
		turn := ProcessTurn(ProcessTurnInput{
			Messages: prefix, State: res.State, Config: in.Config, CountTokens: in.CountTokens,
		})

		applied := ApplyCompression(ApplyInput{
			Ranges: parsed.Ranges, Messages: turn.Messages, State: turn.State,
			Config: in.Config, CallID: m.ToolCallID, Archiver: in.Archiver,
			CreatedAt: in.CreatedAt, Now: in.Now, CountTokens: in.CountTokens,
		})
		if len(applied.Errors) > 0 || len(applied.BlocksCreated) == 0 {
			// A batch is atomic: a rejected one changed nothing then either.
			res.Skipped++
			res.Warnings = append(res.Warnings, applied.Errors...)
			continue
		}
		res.State = applied.State
		res.Replayed++
	}
	return res
}

// RunPruneOnly projects the view without any of the pressure machinery.
//
// Used when noa is switched off: the compression state still has to be applied
// so the history stays the size it was compressed to, but nudging, truncation
// and hide-compress-calls all belong to a system that is no longer running.
func RunPruneOnly(msgs []CoreMessage, state CompressionState, cfg Config) ([]CoreMessage, CompressionState) {
	ensureMaps(&state)
	io := NodeIO{Messages: append([]CoreMessage(nil), msgs...), State: state}
	io = RunPipeline([]PipelineNode{
		assignRefsNode(),
		syncBlocksNode(),
		pruneNode(),
	}, io, PipelineContext{Config: cfg})
	return io.Messages, io.State
}
