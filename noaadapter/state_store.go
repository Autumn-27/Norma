package noaadapter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/Autumn-27/norma/noa"
)

// stateFileVersion guards against silently misreading a future format.
const stateFileVersion = 1

// stateFile is the on-disk shape of the compression state.
type stateFile struct {
	Version       int            `json:"version"`
	SessionID     string         `json:"sessionId"`
	NextBlockID   int            `json:"nextBlockId"`
	Blocks        []blockJSON    `json:"blocks"`
	MessageRefs   refMapJSON     `json:"messageRefs"`
	TokenSnapshot map[string]int `json:"tokenSnapshot,omitempty"`
	Nudge         nudgeJSON      `json:"nudge"`
	Stats         statsJSON      `json:"stats"`
	DerivedFrom   string         `json:"derivedFrom,omitempty"`
}

type blockJSON struct {
	BlockID             string   `json:"blockId"`
	Tier                int      `json:"tier"`
	Topic               string   `json:"topic,omitempty"`
	Summary             string   `json:"summary"`
	DirectMessageIDs    []string `json:"directMessageIds,omitempty"`
	EffectiveMessageIDs []string `json:"effectiveMessageIds,omitempty"`
	DirectBlockIDs      []string `json:"directBlockIds,omitempty"`
	ArchivePath         string   `json:"archivePath,omitempty"`
	ArchiveRel          string   `json:"archiveRel,omitempty"`
	CompressedTokens    int      `json:"compressedTokens"`
	StartRef            string   `json:"startRef,omitempty"`
	EndRef              string   `json:"endRef,omitempty"`
	CreatedAt           int64    `json:"createdAt,omitempty"`
	Active              bool     `json:"active"`
	CompressCallID      string   `json:"compressCallId,omitempty"`
}

type refMapJSON struct {
	ByRaw map[string]string `json:"byRaw"`
	ByRef map[string]string `json:"byRef"`
}

type nudgeJSON struct {
	LastPerMessageNudgeTokens int            `json:"lastPerMessageNudgeTokens"`
	LastNudgeShownTokens      int            `json:"lastNudgeShownTokens"`
	LastShownByTier           map[string]int `json:"lastShownByTier,omitempty"`
}

type statsJSON struct {
	TokensCompressed int `json:"tokensCompressed"`
	CompressionCount int `json:"compressionCount"`
}

// StateStore persists compression state beside the archives.
type StateStore struct {
	mu sync.Mutex
	// path is "" for a session with no durable location.
	path string
	// cache is the authoritative in-process copy.
	cache noa.CompressionState
	// loaded marks the cache as populated.
	loaded bool
}

// NewStateStore returns a store writing to <root>/state.json. An empty root
// yields a memory-only store.
func NewStateStore(root string) *StateStore {
	if root == "" {
		return &StateStore{}
	}
	return &StateStore{path: filepath.Join(root, "state.json")}
}

// Save records the state.
//
// The cache is updated BEFORE the early return for a location-less store. A
// store that skipped the cache when it had no file would hand back a zero state
// on the next load, and the model would re-compress the same range every turn.
func (s *StateStore) Save(st noa.CompressionState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cache = noa.CloneState(st)
	s.loaded = true
	if s.path == "" {
		return nil
	}
	return s.writeAtomic(st)
}

// Load returns the state, preferring the in-process cache.
func (s *StateStore) Load(sessionID, archiveRoot string) (noa.CompressionState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loaded {
		return noa.CloneState(s.cache), nil
	}
	if s.path == "" {
		st := noa.CreateInitialState(sessionID, archiveRoot)
		s.cache, s.loaded = noa.CloneState(st), true
		return st, nil
	}
	st, err := s.readFile(sessionID, archiveRoot)
	if err != nil {
		return noa.CreateInitialState(sessionID, archiveRoot), err
	}
	s.cache, s.loaded = noa.CloneState(st), true
	return st, nil
}

// Exists reports whether a persisted state file is present.
func (s *StateStore) Exists() bool {
	if s.path == "" {
		return false
	}
	_, err := os.Stat(s.path)
	return err == nil
}

func (s *StateStore) readFile(sessionID, archiveRoot string) (noa.CompressionState, error) {
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return noa.CreateInitialState(sessionID, archiveRoot), nil
		}
		return noa.CreateInitialState(sessionID, archiveRoot), err
	}
	var f stateFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return noa.CreateInitialState(sessionID, archiveRoot), fmt.Errorf("parse state file: %w", err)
	}
	if f.Version != stateFileVersion {
		return noa.CreateInitialState(sessionID, archiveRoot),
			fmt.Errorf("state file version %d, want %d", f.Version, stateFileVersion)
	}
	return fromFile(f, sessionID, archiveRoot), nil
}

// writeAtomic replaces the state file via temp + rename, so a crash mid-write
// cannot leave a truncated file where a valid one was.
func (s *StateStore) writeAtomic(st noa.CompressionState) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("create state dir: %w", err)
	}
	data, err := json.MarshalIndent(toFile(st), "", "  ")
	if err != nil {
		return fmt.Errorf("encode state: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".noa-state-*.tmp")
	if err != nil {
		return fmt.Errorf("create temp state: %w", err)
	}
	name := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(name)
		return fmt.Errorf("write state: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return fmt.Errorf("close state: %w", err)
	}
	if err := os.Chmod(name, 0o644); err != nil {
		os.Remove(name)
		return err
	}
	if err := os.Rename(name, s.path); err != nil {
		os.Remove(name)
		return fmt.Errorf("install state: %w", err)
	}
	return nil
}

func toFile(st noa.CompressionState) stateFile {
	f := stateFile{
		Version:       stateFileVersion,
		SessionID:     st.SessionID,
		NextBlockID:   st.NextBlockID,
		MessageRefs:   refMapJSON{ByRaw: st.MessageRefs.ByRaw, ByRef: st.MessageRefs.ByRef},
		TokenSnapshot: st.TokenSnapshot,
		Nudge: nudgeJSON{
			LastPerMessageNudgeTokens: st.Nudge.LastPerMessageNudgeTokens,
			LastNudgeShownTokens:      st.Nudge.LastNudgeShownTokens,
			LastShownByTier:           map[string]int{},
		},
		Stats: statsJSON{TokensCompressed: st.Stats.TokensCompressed, CompressionCount: st.Stats.CompressionCount},
	}
	for tier, n := range st.Nudge.LastShownByTier {
		f.Nudge.LastShownByTier[fmt.Sprintf("%d", tier)] = n
	}
	for _, b := range st.Blocks {
		f.Blocks = append(f.Blocks, blockJSON{
			BlockID: b.BlockID, Tier: int(b.Tier), Topic: b.Topic, Summary: b.Summary,
			DirectMessageIDs: b.DirectMessageIDs, EffectiveMessageIDs: b.EffectiveMessageIDs,
			DirectBlockIDs: b.DirectBlockIDs, ArchivePath: b.ArchivePath, ArchiveRel: b.ArchiveRel,
			CompressedTokens: b.CompressedTokens, StartRef: b.StartRef, EndRef: b.EndRef,
			CreatedAt: b.CreatedAt, Active: b.Active, CompressCallID: b.CompressCallID,
		})
	}
	return f
}

func fromFile(f stateFile, sessionID, archiveRoot string) noa.CompressionState {
	st := noa.CreateInitialState(sessionID, archiveRoot)
	if f.SessionID != "" {
		st.SessionID = f.SessionID
	}
	st.NextBlockID = max(f.NextBlockID, 1)
	if f.MessageRefs.ByRaw != nil {
		st.MessageRefs.ByRaw = f.MessageRefs.ByRaw
	}
	if f.MessageRefs.ByRef != nil {
		st.MessageRefs.ByRef = f.MessageRefs.ByRef
	}
	if f.TokenSnapshot != nil {
		st.TokenSnapshot = f.TokenSnapshot
	}
	st.Nudge.LastPerMessageNudgeTokens = f.Nudge.LastPerMessageNudgeTokens
	st.Nudge.LastNudgeShownTokens = f.Nudge.LastNudgeShownTokens
	for k, v := range f.Nudge.LastShownByTier {
		var tier int
		if _, err := fmt.Sscanf(k, "%d", &tier); err == nil {
			st.Nudge.LastShownByTier[noa.Tier(tier)] = v
		}
	}
	st.Stats.TokensCompressed = f.Stats.TokensCompressed
	st.Stats.CompressionCount = f.Stats.CompressionCount
	for _, b := range f.Blocks {
		st.Blocks = append(st.Blocks, noa.CompressionBlock{
			BlockID: b.BlockID, Tier: noa.Tier(b.Tier), Topic: b.Topic, Summary: b.Summary,
			DirectMessageIDs: b.DirectMessageIDs, EffectiveMessageIDs: b.EffectiveMessageIDs,
			DirectBlockIDs: b.DirectBlockIDs, ArchivePath: b.ArchivePath, ArchiveRel: b.ArchiveRel,
			CompressedTokens: b.CompressedTokens, StartRef: b.StartRef, EndRef: b.EndRef,
			CreatedAt: b.CreatedAt, Active: b.Active, CompressCallID: b.CompressCallID,
		})
	}
	return st
}
