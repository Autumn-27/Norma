//go:build linux

package tool

import (
	"bytes"
	"os"
	"strconv"
	"strings"
)

// memGuardSupported reports whether this platform can measure a task's memory
// use. Only linux can, through /proc; elsewhere the guard stands down and the
// Bash tool description must not advertise a cap it will never enforce.
const memGuardSupported = true

// Indices into the fields of /proc/<pid>/stat that FOLLOW the comm field, as
// returned by statFields: index 0 is stat field 3. See proc(5).
const (
	statPgrp = 2  // field 5: process group id
	statRSS  = 21 // field 24: resident set size, in pages
)

// pgrpRSS sums, per requested process group, the resident memory of every
// process in that group. One pass over /proc answers for every running task at
// once, so the cost is O(processes on the host) no matter how many tasks are
// being watched.
//
// The process GROUP, not the ppid tree, is the unit on purpose: treeKill kills
// the group, so measuring the same set keeps "what we count" and "what we kill"
// identical. A ppid walk diverges the moment a descendant is orphaned — its
// parent exits, init adopts it — because the orphan stays in the group and
// still dies with it, but has left the tree and would stop being counted. A
// shell script that starts its worker with `cmd &` and returns produces exactly
// that shape, so the walk would miss the common leak outright.
//
// A group with no live member is simply absent from the result, which reads as
// zero: a task whose process already exited is never over the cap, and the
// normal exit path finishes it.
//
// RSS charges a copy-on-write page to every process holding it, so a parent
// that forks workers over-reports its shared pages several times over. That
// over-report is deliberate at this tier — it is the cheap, never-miss sample;
// pgrpPSS settles the question before anything is killed.
func pgrpRSS(pgids []int) map[int]int64 {
	want := make(map[int]struct{}, len(pgids))
	for _, g := range pgids {
		want[g] = struct{}{}
	}
	page := int64(os.Getpagesize())
	total := make(map[int]int64, len(pgids))
	forEachPid(func(pid int) {
		f, ok := statFields(pid)
		if !ok || len(f) <= statRSS {
			return
		}
		pgrp, err := strconv.Atoi(f[statPgrp])
		if err != nil {
			return
		}
		if _, ok := want[pgrp]; !ok {
			return
		}
		pages, err := strconv.ParseInt(f[statRSS], 10, 64)
		if err != nil {
			return
		}
		total[pgrp] += pages * page
	})
	return total
}

// pgrpPSS sums the proportional set size of one process group: a page shared by
// n processes is charged 1/n to each, so a parent that forks workers is counted
// once instead of n+1 times. This is the accurate answer and the expensive one
// — smaps_rollup walks every mapping of the target while holding its mmap_lock,
// which costs roughly 0.4µs per VMA (measured: ~13ms for a process with 30k
// mappings, against ~45µs to read its VmRSS) and contends with the target's own
// mmap traffic, which is heaviest in exactly the memory-churning processes this
// guard watches.
//
// So it runs only on the kill path, to confirm a verdict pgrpRSS has already
// reached. ok is false when not one member of the group could be measured —
// the group is gone, and there is nothing to confirm. A single member whose
// smaps_rollup is unreadable (kernel older than 4.14, hardened /proc) falls
// back to that member's RSS, which can only over-report and so cannot turn a
// real overrun into a reprieve.
func pgrpPSS(pgid int) (int64, bool) {
	var total int64
	seen := 0
	forEachPid(func(pid int) {
		f, ok := statFields(pid)
		if !ok || len(f) <= statRSS {
			return
		}
		if pgrp, err := strconv.Atoi(f[statPgrp]); err != nil || pgrp != pgid {
			return
		}
		seen++
		if pss, ok := procPSS(pid); ok {
			total += pss
			return
		}
		if pages, err := strconv.ParseInt(f[statRSS], 10, 64); err == nil {
			total += pages * int64(os.Getpagesize())
		}
	})
	return total, seen > 0
}

// hostMemBudget reports the memory this process can actually exhaust: the
// cgroup limit when one is set — that is the number the kernel OOM-kills
// against, and the one a container deploy cares about — otherwise total host
// RAM. ok is false when neither can be read.
func hostMemBudget() (int64, bool) {
	// cgroup v2: a plain byte count, or "max" for unlimited.
	if b, err := os.ReadFile("/sys/fs/cgroup/memory.max"); err == nil {
		if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil && v > 0 {
			return v, true
		}
	}
	// cgroup v1: unlimited is reported as a near-maxint sentinel, not a word.
	if b, err := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes"); err == nil {
		if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil && v > 0 && v < 1<<62 {
			return v, true
		}
	}
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range bytes.Split(b, []byte("\n")) {
			if !bytes.HasPrefix(line, []byte("MemTotal:")) {
				continue
			}
			// "MemTotal:   16316412 kB"
			f := strings.Fields(string(line))
			if len(f) < 2 {
				break
			}
			if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil && kb > 0 {
				return kb << 10, true
			}
			break
		}
	}
	return 0, false
}

// forEachPid calls fn for every numeric entry under /proc. Entries are not
// filtered by IsDir: on procfs that costs a stat per entry and a non-directory
// named by digits does not exist.
func forEachPid(fn func(pid int)) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return
	}
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		fn(pid)
	}
}

// statFields reads /proc/<pid>/stat and returns the whitespace-separated fields
// that follow comm, so index 0 is stat field 3 (see the stat* constants). comm
// is parenthesized and may itself contain spaces and ')', so the fields start
// after the LAST ')'.
func statFields(pid int) ([]string, bool) {
	b, err := os.ReadFile(procFile(pid, "stat"))
	if err != nil {
		return nil, false
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return nil, false
	}
	return strings.Fields(string(b[i+1:])), true
}

// procPSS reads Pss (kB) from /proc/<pid>/smaps_rollup and returns it in bytes.
func procPSS(pid int) (int64, bool) {
	// ReadFile, not one f.Read: /proc files can answer with a short read, and
	// a partial buffer would miss the line.
	b, err := os.ReadFile(procFile(pid, "smaps_rollup"))
	if err != nil {
		return 0, false
	}
	for _, line := range bytes.Split(b, []byte("\n")) {
		if !bytes.HasPrefix(line, []byte("Pss:")) {
			continue
		}
		// The line is "Pss:\t<kB> kB" — three fields, the unit last.
		f := strings.Fields(string(line))
		if len(f) < 2 {
			return 0, false
		}
		kb, err := strconv.ParseInt(f[1], 10, 64)
		if err != nil {
			return 0, false
		}
		return kb << 10, true
	}
	return 0, false
}

func procFile(pid int, name string) string {
	return "/proc/" + strconv.Itoa(pid) + "/" + name
}
