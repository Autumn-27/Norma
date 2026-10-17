//go:build linux

package tool

import (
	"bytes"
	"os"
	"strconv"
	"strings"
)

// procTreeRSS returns the total RSS in bytes of root and every process below it,
// read from /proc. ok is false when /proc is unreadable or the root has already
// exited — the caller treats that as "nothing to enforce" and lets the exit
// path finish the task.
//
// The walk is a snapshot: processes born or dying mid-walk are missed, which
// can only under-report for one tick. A runaway stays over the cap on every
// later sample, so the guard still lands (see watchMemory).
func procTreeRSS(root *os.Process) (total int64, ok bool) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0, false
	}
	// One pass over /proc/<pid>/stat builds the pid -> ppid map.
	ppid := make(map[int]int, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		if p, ok := procPpid(pid); ok {
			ppid[pid] = p
		}
	}
	children := make(map[int][]int, len(ppid))
	for pid, p := range ppid {
		children[p] = append(children[p], pid)
	}
	// Breadth-first sum from the root; a tree that vanished between the /proc
	// listing and the BFS reports seen == 0.
	seen := 0
	queue := []int{root.Pid}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if r, ok := procRSS(pid); ok {
			total += r
			seen++
		}
		queue = append(queue, children[pid]...)
	}
	return total, seen > 0
}

// procPpid reads the parent pid from /proc/<pid>/stat. The comm field (2nd) is
// parenthesized and may itself contain spaces and ')', so the numeric fields
// start after the LAST ')'; ppid is the second of them.
func procPpid(pid int) (int, bool) {
	b, err := os.ReadFile(procFile(pid, "stat"))
	if err != nil {
		return 0, false
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return 0, false
	}
	fields := strings.Fields(string(b[i+1:]))
	if len(fields) < 2 {
		return 0, false
	}
	v, err := strconv.Atoi(fields[1])
	if err != nil {
		return 0, false
	}
	return v, true
}

// procRSS reads VmRSS (kB, whole process including all threads) from
// /proc/<pid>/status and returns it in bytes. Kernel threads carry no VmRSS
// and are skipped; they are never part of a spawned command's tree.
func procRSS(pid int) (int64, bool) {
	// ReadFile, not one f.Read: /proc files can answer with a short read, and
	// VmRSS sits mid-file — a partial buffer would miss the line.
	b, err := os.ReadFile(procFile(pid, "status"))
	if err != nil {
		return 0, false
	}
	for _, line := range bytes.Split(b, []byte("\n")) {
		if !bytes.HasPrefix(line, []byte("VmRSS:")) {
			continue
		}
		// The line is "VmRSS:\t<kB> kB" — three fields, the unit last.
		fields := strings.Fields(string(line))
		if len(fields) < 2 {
			return 0, false
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
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
