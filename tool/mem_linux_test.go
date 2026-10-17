//go:build linux

package tool

import (
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// holdOrphan backgrounds a memory holder from a subshell that exits at once:
// the holder is reparented to pid 1 and so leaves the task's ppid tree, but
// stays in its process group — the set treeKill actually kills. The trailing
// ":" in each branch stops the shell from exec-ing the final sleep in place,
// which would drop the allocation the test depends on.
const holdOrphan = `( { x=a; while [ ${#x} -lt 200000000 ]; do x="$x$x"; done; sleep 30; :; } & ) ; sleep 30; :`

// forkShared allocates once and then forks five subshells that hold the same
// pages copy-on-write. Summing per-process RSS counts those pages six times
// (~880 MiB measured); the real footprint is ~145 MiB.
const forkShared = `x=a; while [ ${#x} -lt 100000000 ]; do x="$x$x"; done; for i in 1 2 3 4 5; do { sleep 25; :; } & done; sleep 25; :`

// A leaker whose parent exits must still be caught. Measuring the ppid tree
// instead of the process group loses it the moment init adopts it: the walk
// reports the few MiB of the surviving shell while the orphan holds hundreds,
// and the guard never fires on the shape a shell script produces every time it
// starts its real work with "cmd &".
func TestMemGuardCatchesOrphanedDescendant(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "64m")
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: holdOrphan})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminalIn(t, m, task.ID, 30*time.Second)
	if info.Status != TaskFailed || info.Reason != terminateMemory {
		t.Fatalf("status=%s reason=%q, want failed/memory — the orphan left the ppid tree but not the process group", info.Status, info.Reason)
	}
}

// The inverse: a parent that forks workers shares its pages, and summing RSS
// over the group multiplies them by the number of holders. Killing on that sum
// would take down ordinary multiprocessing work and tell the model to rewrite a
// command that never leaked, so a group over the cheap bound must be confirmed
// against Pss before it dies.
func TestMemGuardSparesSharedPages(t *testing.T) {
	// Between the two measures: sum(RSS) ~880 MiB is over, Pss ~145 MiB is not.
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "400m")
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: forkShared})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)

	// Wait for the two measures to actually diverge: while the parent is still
	// growing its string alone there is nothing shared yet and RSS == Pss, so
	// polling on RSS alone would sample before the forks exist.
	pgid := task.cmd.Process.Pid
	var rss, pss int64
	var ok bool
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		rss = pgrpRSS([]int{pgid})[pgid]
		pss, ok = pgrpPSS(pgid)
		if ok && rss > 400<<20 && pss < 300<<20 {
			break
		}
		if info, ok := m.Get(task.ID); ok && info.Status.Terminal() {
			t.Fatalf("task died at %s/%s before the shared allocation was resident (rss=%dMiB pss=%dMiB)",
				info.Status, info.Reason, rss>>20, pss>>20)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !ok || rss <= 400<<20 || pss >= 400<<20 {
		t.Skipf("workload did not land between the two measures on this host (rss=%dMiB pss=%dMiB ok=%v)", rss>>20, pss>>20, ok)
	}

	// Several guard passes with the group sitting over the cheap bound.
	time.Sleep(3 * memSampleInterval)
	info, _ := m.Get(task.ID)
	if info.Status.Terminal() {
		t.Fatalf("killed a group whose real footprint (%d MiB Pss) is under the %s cap; only its double-counted RSS (%d MiB) was over",
			pss>>20, humanBytes(memCapBytes()), rss>>20)
	}
	m.Kill(task.ID)
}

// pgrpRSS must see every member of the group, including one the task's own
// parent-child links no longer reach.
func TestPgrpRSSCoversWholeGroup(t *testing.T) {
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: holdOrphan})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer m.Kill(task.ID)
	pgid := task.cmd.Process.Pid

	var measured, want int64
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		measured = pgrpRSS([]int{pgid})[pgid]
		want = groupRSSViaProc(t, pgid)
		if want > 100<<20 {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	if want <= 100<<20 {
		t.Skipf("holder never became resident on this host (%d MiB)", want>>20)
	}
	// Both read /proc a moment apart, so allow drift; the failure this guards
	// against is an order-of-magnitude miss, not a few MiB.
	if diff := want - measured; diff > 32<<20 || diff < -(32<<20) {
		t.Fatalf("pgrpRSS = %d MiB, independent group sum = %d MiB", measured>>20, want>>20)
	}
}

// groupRSSViaProc sums VmRSS over the process group, reading /proc/<pid>/status
// rather than the stat field pgrpRSS parses, so the two do not share a bug.
func groupRSSViaProc(t *testing.T, pgid int) int64 {
	t.Helper()
	entries, err := os.ReadDir("/proc")
	if err != nil {
		t.Fatalf("read /proc: %v", err)
	}
	var total int64
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		g, err := syscall.Getpgid(pid)
		if err != nil || g != pgid {
			continue
		}
		b, err := os.ReadFile(procFile(pid, "status"))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(line, "VmRSS:") {
				continue
			}
			f := strings.Fields(line)
			if len(f) >= 2 {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					total += kb << 10
				}
			}
			break
		}
	}
	return total
}

// The derived default must be a usable cap, not a constant and not something
// that would kill ordinary work on the machine running the tests.
func TestDefaultMemCapFollowsHostBudget(t *testing.T) {
	budget, ok := hostMemBudget()
	if !ok {
		t.Skip("host memory budget unreadable")
	}
	if budget < minMemCap {
		t.Skipf("host budget %s is below the floor", humanBytes(budget))
	}
	got := defaultMemCap()
	if got > budget {
		t.Fatalf("cap %s exceeds the budget %s it is meant to protect", humanBytes(got), humanBytes(budget))
	}
	if got < minMemCap {
		t.Fatalf("cap %s is below the floor %s", humanBytes(got), humanBytes(minMemCap))
	}
}
