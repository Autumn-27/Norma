package tool

import (
	"testing"
)

func TestParseMemCap(t *testing.T) {
	cases := []struct {
		in   string
		want int64
		ok   bool
	}{
		{"0", 0, true},
		{"off", 0, true},
		{"OFF", 0, true},
		{"  0  ", 0, true},
		{"512", 512, true},
		{"512b", 512, true},
		{"1k", 1 << 10, true},
		{"2kb", 2 << 10, true},
		{"3KiB", 3 << 10, true},
		{"512m", 512 << 20, true},
		{"1MB", 1 << 20, true},
		{"2g", 2 << 30, true},
		{"2gb", 2 << 30, true},
		{"1GiB", 1 << 30, true},
		{"10g", 10 << 30, true},
		{"1t", 1 << 40, true},
		{"0b", 0, true},
		{" 3m ", 3 << 20, true},
		// rejects
		{"", 0, false},
		{"-5", 0, false},
		{"abc", 0, false},
		{"5x", 0, false},
		{"1.5g", 0, false},
		{"g", 0, false},
		{"m b", 0, false},
	}
	for _, c := range cases {
		got, ok := parseMemCap(c.in)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("parseMemCap(%q) = (%d, %v), want (%d, %v)", c.in, got, ok, c.want, c.ok)
		}
	}
}

// An overflowing product must be rejected, not wrapped: a wrapped value lands
// on a negative or zero cap, which memCapBytes reads as "guard disabled" — so a
// mistyped unit would silently switch the guard off, the one thing the
// fall-back-to-default rule exists to prevent.
func TestParseMemCapRejectsOverflow(t *testing.T) {
	for _, in := range []string{"8388608t", "16777216t", "9223372036854775807g", "99999999999999999999"} {
		if got, ok := parseMemCap(in); ok {
			t.Errorf("parseMemCap(%q) = (%d, true), want rejected", in, got)
		}
		t.Setenv("AGENT_CORE_TASK_MEM_CAP", in)
		if got := memCapBytes(); got != defaultMemCap() {
			t.Errorf("AGENT_CORE_TASK_MEM_CAP=%q yielded %d, want the default %d", in, got, defaultMemCap())
		}
	}
}

func TestMemCapEnvOverride(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "64m")
	if got := memCapBytes(); got != 64<<20 {
		t.Fatalf("memCapBytes() = %d, want %d", got, 64<<20)
	}
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "off")
	if got := memCapBytes(); got != 0 {
		t.Fatalf("off must disable the guard, got %d", got)
	}
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "bogus")
	if got := memCapBytes(); got != defaultMemCap() {
		t.Fatalf("unparseable value must fall back to the default, got %d", got)
	}
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "")
	if got := memCapBytes(); got != defaultMemCap() {
		t.Fatalf("unset value must yield the default, got %d", got)
	}
}

func TestHumanBytes(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{2 << 30, "2GiB"},
		{1 << 30, "1GiB"},
		{512 << 20, "512MiB"},
		{1 << 20, "1MiB"},
		{3 << 10, "3KiB"},
		{300, "300 B"},
		{0, "0 B"},
		{3 << 30, "3GiB"},
		{1 << 40, "1TiB"},
		{2<<30 + 512<<20, "2.5GiB"},
	}
	for _, c := range cases {
		if got := humanBytes(c.in); got != c.want {
			t.Errorf("humanBytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}
