package noa

import (
	"fmt"
	"sync"
)

// memArchiver is an in-memory Archiver for tests: it records what was written
// so assertions can inspect the archive without touching a filesystem.
type memArchiver struct {
	mu      sync.Mutex
	files   map[string][]byte
	order   []string
	failOn  string // block id whose Write should fail
	removed []string
}

func newMemArchiver() *memArchiver { return &memArchiver{files: map[string][]byte{}} }

func (a *memArchiver) Write(tier Tier, blockID, startRef, endRef string, content []byte) (string, string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.failOn == blockID {
		return "", "", fmt.Errorf("simulated write failure for %s", blockID)
	}
	rel := ArchiveTierDir(tier) + "/" + ArchiveFileName(blockID, startRef, endRef)
	abs := "/archive/" + rel
	a.files[abs] = append([]byte(nil), content...)
	a.order = append(a.order, abs)
	return abs, rel, nil
}

func (a *memArchiver) Remove(abs string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.files, abs)
	a.removed = append(a.removed, abs)
	return nil
}

func (a *memArchiver) content(abs string) string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return string(a.files[abs])
}

func (a *memArchiver) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.files)
}
