package tool

import (
	"os"
	"path/filepath"
)

// atomicReplace stages an existing file's replacement in the same directory.
// A failed write, sync, close or rename leaves the original intact. Readers see
// the old or new file on platforms with atomic same-directory rename (Unix).
// Follow symlinks as Edit does, and retain the target's permission bits.
// This is not a transaction against concurrent external writers, and replacing
// an inode does not preserve hard-link identity, ownership or extended metadata.
func atomicReplace(path string, write func(*os.File) error) error {
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		return err
	}
	// Retain the old requirement that the target itself be writable; directory
	// write permission alone must not allow edits of a read-only file.
	original, err := os.OpenFile(target, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	info, err := original.Stat()
	closeErr := original.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	f, err := os.CreateTemp(filepath.Dir(target), ".norma-edit-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	if err := write(f); err != nil {
		return err
	}
	if err := f.Chmod(info.Mode().Perm()); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), target)
}
