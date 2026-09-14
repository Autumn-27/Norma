package noaadapter

import (
	"errors"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/noa"
)

// Options configures one noa-managed session.
type Options struct {
	// ArchiveBaseDir is where archives and state live. REQUIRED: noa refuses to
	// guess. The archive is the only route back to compressed originals, so
	// where it lands must be the host's explicit decision — never a silent
	// fallback to a temp directory that may be swept away.
	ArchiveBaseDir string
	// SessionID names the subdirectory under ArchiveBaseDir.
	SessionID string
	// Config overrides the defaults; the zero value means DefaultConfig.
	Config *noa.Config
	// ModelContextLimit sizes the default config when Config is nil.
	ModelContextLimit int
	// ParentSessionIDs are ancestors to inherit a ledger from when this session
	// has none of its own, nearest first. A fork, a subagent, or a transcript
	// copied without its sidecar would otherwise start from zero and re-compress
	// everything the ancestor already did. At most maxInheritDepth are tried.
	ParentSessionIDs []string
	// OnWarn receives non-fatal diagnostics. noa writes nothing to stdout or
	// stderr on its own.
	OnWarn func(string)
	// Now overrides the clock, for tests.
	Now func() time.Time
}

// ErrNoArchiveDir is returned when ArchiveBaseDir is empty.
var ErrNoArchiveDir = errors.New("noaadapter: ArchiveBaseDir is required — compression must not run without a durable place for the originals")

// Session is the state the compactor and the Compress tool share.
//
// They are two entry points into one system and must see the same data:
//
//   - state: the tool creates blocks, the view hides them. If they diverge, the
//     model compresses into a void — the panel reports success and the next
//     request still carries the full history.
//   - lastView: the refs the model used were resolved against the array the
//     view produced. Re-deriving it in the tool could yield a different array
//     and therefore a misaligned range.
//   - attempts: the tool counts failures, the view reads the count to decide
//     whether to keep nudging.
type Session struct {
	mu sync.Mutex

	state    noa.CompressionState
	lastView []noa.CoreMessage
	sidecar  *Sidecar

	// lastTokenCount is the measured size of the context this turn.
	lastTokenCount int
	// attempts counts consecutive failed or ignored compression prompts.
	attempts int
	// suppressedAtTokens records where suppression began, so it can lift once
	// the context has grown enough for the situation to have changed.
	suppressedAtTokens int
	// nudgedLastTurn lets the view notice a nudge that was ignored outright.
	nudgedLastTurn bool
	// lastTruncatedCount is how many tool results were mechanically shortened.
	lastTruncatedCount int
	// providerTokens is the input size the provider last reported. It is
	// authoritative where available: a local estimate can drift, and the whole
	// pressure ladder keys off this number.
	providerTokens int
	// emergencyFloor, when non-zero, forces the view to build under this ceiling
	// after a prompt-too-long error taught us the real window.
	emergencyFloor int
	// deadRange refuses a range set the model keeps resubmitting.
	deadRange *DeadRangeTracker
	// lastCompressAt is Stats.CompressionCount at the last successful
	// compression, used to spot a stale provider token anchor.
	lastCompressAt int
	// providerTokensAt is the compression count when providerTokens was
	// reported, so a figure that predates a compression can be recognised.
	providerTokensAt int

	cfg      noa.Config
	store    *StateStore
	archiver noa.Archiver
	root     string
	now      func() time.Time
	onWarn   func(string)
}

func newSession(o Options) (*Session, error) {
	if o.ArchiveBaseDir == "" {
		return nil, ErrNoArchiveDir
	}
	cfg := noa.DefaultConfig(o.ModelContextLimit)
	if o.Config != nil {
		cfg = *o.Config
	}
	if cfg.ModelContextLimit <= 0 {
		cfg.ModelContextLimit = 200000
	}
	now := o.Now
	if now == nil {
		now = time.Now
	}

	root := filepath.Join(o.ArchiveBaseDir, o.SessionID)
	s := &Session{
		cfg:       cfg,
		store:     NewStateStore(root),
		archiver:  NewFileArchiver(root),
		root:      root,
		now:       now,
		onWarn:    o.OnWarn,
		deadRange: NewDeadRangeTracker(),
	}
	for _, w := range noa.ValidateConfig(cfg) {
		s.warn("config: " + w)
	}
	st, err := s.store.Load(o.SessionID, root)
	if err != nil {
		// A damaged state file is recoverable: blocks can be replayed from the
		// transcript. Starting fresh beats refusing to run.
		s.warn("state: " + err.Error() + " — starting from an empty state")
		st = noa.CreateInitialState(o.SessionID, root)
	}
	if len(st.Blocks) == 0 && len(o.ParentSessionIDs) > 0 {
		if inherited, from, ok := InheritState(InheritOptions{
			ArchiveBaseDir: o.ArchiveBaseDir, SessionID: o.SessionID, ParentIDs: o.ParentSessionIDs,
		}); ok {
			s.warn("inherited " + itoa(len(inherited.Blocks)) + " block(s) from session " + from)
			st = inherited
		}
	}
	s.state = st
	return s, nil
}

func itoa(n int) string { return strconv.Itoa(n) }

func (s *Session) warn(msg string) {
	if s.onWarn != nil {
		s.onWarn("noa: " + msg)
	}
}

// ArchiveRoot is where this session's archives and state live.
func (s *Session) ArchiveRoot() string { return s.root }

// Config returns the resolved configuration.
func (s *Session) Config() noa.Config { return s.cfg }

// State returns a copy of the current compression state.
func (s *Session) State() noa.CompressionState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return noa.CloneState(s.state)
}

// View projects the history for one request.
//
// Pure with respect to msgs: the returned slice is for this request only and is
// never written back. Message identity is a content hash, so writing a tagged
// or truncated body back would change every id derived from it.
func (s *Session) View(msgs []llm.Message) []llm.Message {
	s.mu.Lock()
	defer s.mu.Unlock()

	cores, sc := Project(msgs)
	tokens := s.resolveTokenCount(cores)
	s.lastTokenCount = tokens

	// A new user turn means the situation changed; give the model a clean slate
	// before deciding whether to nudge.
	s.notePriorTurnOutcome(msgs)

	cfg := s.cfg
	if s.emergencyFloor > 0 {
		// A prompt-too-long error taught us the real window. Build to it, rather
		// than to the configured one that just failed.
		cfg.ModelContextLimit = s.emergencyFloor
	}

	res := noa.ProcessTurn(noa.ProcessTurnInput{
		Messages:   cores,
		State:      s.state,
		Config:     cfg,
		TokenCount: tokens,
	})
	s.state = res.State
	s.sidecar = sc
	s.lastTruncatedCount = res.TruncatedCount

	view := res.Messages
	s.nudgedLastTurn = false
	if res.Nudge != nil && s.nudgeAllowed(tokens) {
		voice, text := noa.RenderNudgeText(*res.Nudge, noa.NudgeSections{})
		_ = voice
		view = append(view, noa.CoreMessage{
			ID: noa.NudgeMessageID, Role: noa.RoleUser, ContentType: noa.CTText, Text: text,
		})
		s.nudgedLastTurn = true
	}
	// lastView is what the model's refs will be resolved against, so it must be
	// the array the model actually sees — nudge included.
	s.lastView = view

	return Reassemble(view, msgs, sc, ReassembleOptions{State: &s.state, Tag: true})
}

// nudgeAllowed gates injection on the failure ladder.
//
// After MaxCompressAttempts consecutive failures or outright refusals, nudging
// stops: each one costs a couple of thousand tokens, and replaying it into a
// context that is already too big makes the problem it describes worse.
//
// Suppression is not permanent. It lifts once the context has grown by a full
// cadence step, because by then the situation genuinely differs from the one
// the model declined to act on. Without that release the first three failures
// in a session would silence nudging for good. Callers hold the lock.
func (s *Session) nudgeAllowed(tokenCount int) bool {
	if s.attempts < s.cfg.MaxCompressAttempts {
		return true
	}
	floor := noa.NudgeGrowthFloor(s.cfg)
	if tokenCount-s.suppressedAtTokens >= floor {
		s.attempts = 0
		s.suppressedAtTokens = 0
		return true
	}
	return false
}

// notePriorTurnOutcome reacts to what happened since the last view.
//
// Two things reset or advance the ladder here:
//
//   - A genuine new user message means new instructions and a new situation, so
//     the model deserves a fresh start.
//   - A nudge that drew no Compress call at all counts as a failure. Upstream
//     only counts calls that FAILED, which leaves the cheapest way to ignore a
//     nudge — not calling the tool — entirely uncounted, and an emergency nudge
//     then repeats on every request forever.
//
// Callers hold the lock.
func (s *Session) notePriorTurnOutcome(msgs []llm.Message) {
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	if isRealUserTurn(last) {
		s.attempts = 0
		s.suppressedAtTokens = 0
		s.nudgedLastTurn = false
		return
	}
	if s.nudgedLastTurn && !lastAssistantCalledCompress(msgs) {
		s.noteFailedAttempt()
		s.nudgedLastTurn = false
	}
}

// isRealUserTurn distinguishes a person speaking from a packaged tool result.
//
// Norma carries tool results in Role: user messages, so the role alone says
// nothing; the absence of any tool_result block is what marks a real turn.
func isRealUserTurn(m llm.Message) bool {
	if m.Role != llm.RoleUser {
		return false
	}
	for _, b := range m.Content {
		if b.Type == llm.BlockToolResult {
			return false
		}
	}
	return len(m.Content) > 0
}

// lastAssistantCalledCompress reports whether the most recent assistant message
// invoked Compress.
func lastAssistantCalledCompress(msgs []llm.Message) bool {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != llm.RoleAssistant {
			continue
		}
		for _, b := range msgs[i].Content {
			if b.Type == llm.BlockToolUse && b.Name == noa.CompressToolName {
				return true
			}
		}
		return false
	}
	return false
}

// resolveTokenCount decides what this turn's context size is.
//
// The provider's reported input size wins when it is close to the local
// estimate; a large divergence means the reported figure describes a different
// array than the one being built (it lags a compression, or counts content
// prune removes), and the estimate of the SENT view is the honest number.
// Callers hold the lock.
func (s *Session) resolveTokenCount(cores []noa.CoreMessage) int {
	est := estimateCoreTokens(cores)
	p := s.providerTokens
	if p <= 0 {
		return est
	}
	// Floor-stale: the provider figure was reported before the most recent
	// compression, so it describes a context that no longer exists. Trusting it
	// would keep the pressure ladder pinned at the pre-compression size and
	// re-trigger an emergency the model already resolved.
	if s.providerTokensAt < s.lastCompressAt {
		return est
	}
	drift := p - est
	if drift < 0 {
		drift = -drift
	}
	if drift > max(1000, est/10) {
		return est
	}
	return p
}

// estimateCoreTokens sizes the projected view.
//
// Estimating the SENT view rather than raw history matters: the raw array still
// contains everything prune will hide, so counting it would pin the figure high
// forever and drive compression that reclaims nothing.
func estimateCoreTokens(cores []noa.CoreMessage) int {
	total := 0
	for _, c := range cores {
		total += noa.DefaultCountTokens(c.Text)
	}
	return total
}

// applyCompression runs one Compress call against the shared state.
func (s *Session) applyCompression(ranges []noa.CompressRange, callID string) noa.ApplyResult {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now()
	res := noa.ApplyCompression(noa.ApplyInput{
		Ranges:    ranges,
		Messages:  s.lastView,
		State:     s.state,
		Config:    s.cfg,
		CallID:    callID,
		Archiver:  s.archiver,
		CreatedAt: now.Format(time.RFC3339),
		Now:       now.Unix(),
	})
	if len(res.BlocksCreated) > 0 {
		s.state = res.State
		s.attempts = 0
		s.suppressedAtTokens = 0
		s.lastCompressAt = s.state.Stats.CompressionCount
		// The view changed, so a range that was dead before may not be now.
		s.deadRange.Reset()
		if err := s.store.Save(s.state); err != nil {
			// The compression itself succeeded and the archives are on disk;
			// losing the state file costs a rebuild, not data.
			s.warn("state save failed: " + err.Error())
		}
	} else {
		s.noteFailedAttempt()
		s.deadRange.Record(ranges)
	}
	return res
}

// checkDeadRange refuses a range set the model keeps resubmitting.
func (s *Session) checkDeadRange(ranges []noa.CompressRange) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deadRange.Check(ranges, s.state)
}

// recordParseFailure counts arguments that could not be parsed at all.
//
// Malformed arguments count against the same ladder as an empty result: from
// the context's point of view nothing was reclaimed either way, and a model
// stuck producing unusable JSON loops just as expensively as one producing
// unusable ranges.
func (s *Session) recordParseFailure() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.noteFailedAttempt()
}

// noteFailedAttempt records an attempt that reclaimed nothing. Callers hold the
// lock.
func (s *Session) noteFailedAttempt() {
	s.attempts++
	if s.attempts == s.cfg.MaxCompressAttempts {
		s.suppressedAtTokens = s.lastTokenCount
	}
}

// noteProviderTokens records the input size the provider reported.
func (s *Session) noteProviderTokens(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.providerTokens = n
	s.providerTokensAt = s.state.Stats.CompressionCount
}

// setEmergencyFloor arms or clears the post-overflow ceiling.
func (s *Session) setEmergencyFloor(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.emergencyFloor = n
}

// tokensBefore reports the current view's size, for the panel headline.
func (s *Session) tokensBefore() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastTokenCount
}
