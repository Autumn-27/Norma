//go:build !linux

package tool

import (
	"os"
)

// procTreeRSS is unimplemented off linux: there is no portable pure-Go way to
// read another process tree's RSS without cgo, and the deploys this guard
// protects (containerized, pod memory limits) are all linux. Reporting
// unavailable stands the watchdog down instead of guessing.
func procTreeRSS(root *os.Process) (int64, bool) {
	return 0, false
}
