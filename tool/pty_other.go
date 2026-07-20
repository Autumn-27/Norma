//go:build windows

package tool

import (
	"errors"
	"os"
	"os/exec"
)

// startPTY is unsupported on Windows (no ConPTY integration yet). Interactive
// shell sessions degrade to an explicit error there.
func startPTY(_ *exec.Cmd, _, _ int) (*os.File, error) {
	return nil, errors.New("interactive shell (PTY) is not supported on this platform")
}

func ptySupported() bool { return false }
