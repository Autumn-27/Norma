//go:build !linux

package tool

// memGuardSupported is false off linux: there is no portable pure-Go way to
// read another process group's memory use without cgo, and the deploys this
// guard protects (containers under a pod memory limit) are all linux. Standing
// the watchdog down beats guessing — and keeps the Bash tool description from
// advertising a cap that would never fire.
const memGuardSupported = false

func pgrpRSS(_ []int) map[int]int64 { return nil }

func pgrpPSS(_ int) (int64, bool) { return 0, false }

func hostMemBudget() (int64, bool) { return 0, false }
