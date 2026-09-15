package noaadapter

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/Autumn-27/norma/noa"
)

// fileArchiver writes archives under <root>/tier<N>/.
type fileArchiver struct {
	mu   sync.Mutex
	root string
}

// NewFileArchiver returns an Archiver rooted at dir.
func NewFileArchiver(root string) noa.Archiver { return &fileArchiver{root: root} }

// Write persists one archive atomically.
//
// Atomicity matters more here than almost anywhere else in the system: a block
// exists only if its archive does, and a half-written archive would be a block
// promising originals it cannot produce. Writing to a temp file and renaming
// means a reader sees either the whole file or none of it.
func (a *fileArchiver) Write(tier noa.Tier, blockID, startRef, endRef string, content []byte) (string, string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	rel := filepath.Join(noa.ArchiveTierDir(tier), noa.ArchiveFileName(blockID, startRef, endRef))
	abs := filepath.Join(a.root, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return "", "", fmt.Errorf("create archive dir: %w", err)
	}

	tmp, err := os.CreateTemp(filepath.Dir(abs), ".noa-*.tmp")
	if err != nil {
		return "", "", fmt.Errorf("create temp archive: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return "", "", fmt.Errorf("write archive: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return "", "", fmt.Errorf("close archive: %w", err)
	}
	if err := os.Chmod(tmpName, 0o644); err != nil {
		os.Remove(tmpName)
		return "", "", fmt.Errorf("chmod archive: %w", err)
	}
	if err := os.Rename(tmpName, abs); err != nil {
		os.Remove(tmpName)
		return "", "", fmt.Errorf("install archive: %w", err)
	}
	return abs, filepath.ToSlash(rel), nil
}

// Remove deletes an archive, used to roll back a failed batch. A missing file
// is not an error: rollback may run over archives that never landed.
func (a *fileArchiver) Remove(abs string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := os.Remove(abs); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// reuseArchiver wraps an Archiver so an existing file is adopted rather than
// rewritten.
//
// State rebuilt from the transcript replays past compressions; their archives
// are historical fact and must not be overwritten by a re-render. Only a
// genuinely missing archive is regenerated.
type reuseArchiver struct {
	inner noa.Archiver
	root  string
}

// NewReuseArchiver returns an Archiver that adopts archives already on disk.
func NewReuseArchiver(root string) noa.Archiver {
	return &reuseArchiver{inner: NewFileArchiver(root), root: root}
}

func (a *reuseArchiver) Write(tier noa.Tier, blockID, startRef, endRef string, content []byte) (string, string, error) {
	rel := filepath.Join(noa.ArchiveTierDir(tier), noa.ArchiveFileName(blockID, startRef, endRef))
	abs := filepath.Join(a.root, rel)
	if _, err := os.Stat(abs); err == nil {
		return abs, filepath.ToSlash(rel), nil
	}
	return a.inner.Write(tier, blockID, startRef, endRef, content)
}

// Remove is a no-op: a replay must not delete archives it did not create.
func (a *reuseArchiver) Remove(string) error { return nil }
