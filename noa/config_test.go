package noa

import (
	"strings"
	"testing"
)

func TestDefaultConfigMatchesSpec(t *testing.T) {
	c := DefaultConfig(200000)
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"MaxTier", c.Tiers.MaxTier, Tier(3)},
		{"Tier2Trigger", c.Tiers.Tier2Trigger, 5},
		{"Tier3Trigger", c.Tiers.Tier3Trigger, 10},
		{"MaxContextLimitPct", c.Nudge.MaxContextLimitPct, 0.75},
		{"MinContextLimitPct", c.Nudge.MinContextLimitPct, 0.45},
		{"EmergencyThresholdPct", c.Nudge.EmergencyThresholdPct, 0.95},
		{"GrowthFloor", c.Nudge.GrowthFloor, 50000},
		{"GrowthCap", c.Nudge.GrowthCap, 50000},
		{"MinGrowthFloor", c.Nudge.MinGrowthFloor, 20000},
		{"MinGrowthRatio", c.Nudge.MinGrowthRatio, 0.45},
		{"Tier2GrowthMultiplier", c.Nudge.Tier2GrowthMultiplier, 1.5},
		{"Truncate.Threshold", c.Truncate.Threshold, 0.95},
		{"MinCompressRange", c.Compress.MinCompressRange, 5000},
		{"MinSummaryLength", c.Compress.MinSummaryLength, 50},
		{"MaxSummaryLength", c.Compress.MaxSummaryLength, 20000},
		{"PreserveRecentMessages", c.PreserveRecentMessages, 5},
		{"PreserveRecentTokens", c.PreserveRecentTokens, 5000},
		{"MaxCompressAttempts", c.MaxCompressAttempts, 3},
	}
	for _, ch := range checks {
		if ch.got != ch.want {
			t.Errorf("%s = %v, want %v", ch.name, ch.got, ch.want)
		}
	}
}

func TestValidateConfigAcceptsDefaults(t *testing.T) {
	if w := ValidateConfig(DefaultConfig(200000)); len(w) != 0 {
		t.Fatalf("ValidateConfig(defaults) = %v, want no warnings", w)
	}
}

func TestValidateConfigWarnings(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
		want   string
	}{
		{"zero limit", func(c *Config) { c.ModelContextLimit = 0 }, "ModelContextLimit"},
		{"min above max", func(c *Config) { c.Nudge.MinContextLimitPct = 0.9 }, "MinContextLimitPct"},
		{"max above emergency", func(c *Config) { c.Nudge.MaxContextLimitPct = 0.99 }, "MaxContextLimitPct"},
		{"threshold zero", func(c *Config) { c.Truncate.Threshold = 0 }, "Truncate.Threshold"},
		{"threshold above one", func(c *Config) { c.Truncate.Threshold = 1.5 }, "Truncate.Threshold"},
		{"tier2 zero", func(c *Config) { c.Tiers.Tier2Trigger = 0 }, "Tier2Trigger"},
		{"tier3 not above tier2", func(c *Config) { c.Tiers.Tier3Trigger = 5 }, "Tier3Trigger"},
		{"maxtier too high", func(c *Config) { c.Tiers.MaxTier = 5 }, "MaxTier"},
		{"maxtier too low", func(c *Config) { c.Tiers.MaxTier = 1 }, "MaxTier"},
		{"summary bounds inverted", func(c *Config) { c.Compress.MinSummaryLength = 30000 }, "MinSummaryLength"},
		{"attempts zero", func(c *Config) { c.MaxCompressAttempts = 0 }, "MaxCompressAttempts"},
	}
	for _, tc := range cases {
		c := DefaultConfig(200000)
		tc.mutate(&c)
		w := ValidateConfig(c)
		if len(w) == 0 {
			t.Errorf("%s: ValidateConfig returned no warning", tc.name)
			continue
		}
		if !strings.Contains(strings.Join(w, "\n"), tc.want) {
			t.Errorf("%s: warnings %v, want one mentioning %q", tc.name, w, tc.want)
		}
	}
}

func TestValidateConfigNegativeMinPressureBenefit(t *testing.T) {
	c := DefaultConfig(200000)
	neg := -1
	c.Nudge.MinPressureBenefitTokens = &neg
	if w := ValidateConfig(c); len(w) == 0 {
		t.Fatal("ValidateConfig accepted a negative MinPressureBenefitTokens")
	}
}

// MaxTier=2 is the supported test lever: it makes the terminal-tier guard
// reachable with a fraction of the fixture.
func TestValidateConfigAcceptsMaxTierTwo(t *testing.T) {
	c := DefaultConfig(200000)
	c.Tiers.MaxTier = 2
	if w := ValidateConfig(c); len(w) != 0 {
		t.Fatalf("ValidateConfig with MaxTier=2 = %v, want no warnings", w)
	}
}

func TestMinPressureBenefitDefault(t *testing.T) {
	// max(5000, round(200000*0.01)) = max(5000, 2000) = 5000
	if got := DefaultConfig(200000).minPressureBenefit(); got != 5000 {
		t.Fatalf("minPressureBenefit(200k window) = %d, want 5000", got)
	}
	// max(5000, round(1000000*0.01)) = max(5000, 10000) = 10000
	if got := DefaultConfig(1000000).minPressureBenefit(); got != 10000 {
		t.Fatalf("minPressureBenefit(1M window) = %d, want 10000", got)
	}
}

func TestMinPressureBenefitExplicitZeroDisablesGate(t *testing.T) {
	c := DefaultConfig(200000)
	zero := 0
	c.Nudge.MinPressureBenefitTokens = &zero
	if got := c.minPressureBenefit(); got != 0 {
		t.Fatalf("minPressureBenefit with explicit 0 = %d, want 0 — an explicit zero restores the old ungated behaviour", got)
	}
}

func TestIsMessageProtected(t *testing.T) {
	c := DefaultConfig(200000)
	compress := CoreMessage{ContentType: CTToolCall, ToolName: CompressToolName}
	if !IsMessageProtected(compress, c) {
		t.Fatal("the Compress tool must always be hard-protected")
	}
	if IsMessageProtected(CoreMessage{ContentType: CTToolCall, ToolName: "Read"}, c) {
		t.Fatal("Read is not protected by default")
	}
	c.ProtectedTools = []string{"Read"}
	if !IsMessageProtected(CoreMessage{ContentType: CTToolCall, ToolName: "Read"}, c) {
		t.Fatal("a configured ProtectedTools entry was ignored")
	}
	c.ProtectedTools = nil
	c.IsToolProtected = func(n string) bool { return n == "Custom" }
	if !IsMessageProtected(CoreMessage{ContentType: CTToolCall, ToolName: "Custom"}, c) {
		t.Fatal("the IsToolProtected predicate was ignored")
	}
}

func TestMatchToolPattern(t *testing.T) {
	cases := []struct {
		pattern, tool string
		want          bool
	}{
		{"Read", "Read", true},
		{"read", "Read", true},
		{"mcp__*", "mcp__github__list", true},
		{"mcp__*", "Read", false},
		{"Read", "ReadFile", false},
		{"", "Read", false},
		{"Read", "", false},
	}
	for _, c := range cases {
		if got := MatchToolPattern(c.pattern, c.tool); got != c.want {
			t.Errorf("MatchToolPattern(%q,%q) = %v, want %v", c.pattern, c.tool, got, c.want)
		}
	}
}

// These tools do not occupy a protected-zone slot, but remain compressible —
// a distinct concept from ProtectedTools.
func TestIsNeverPreserveRecentTool(t *testing.T) {
	for _, n := range []string{"Read", "Grep", "Glob", "Bash", "LS", "read", "bash"} {
		if !IsNeverPreserveRecentTool(CoreMessage{ToolName: n}) {
			t.Errorf("IsNeverPreserveRecentTool(%q) = false, want true", n)
		}
	}
	for _, n := range []string{"Write", "Edit", CompressToolName, ""} {
		if IsNeverPreserveRecentTool(CoreMessage{ToolName: n}) {
			t.Errorf("IsNeverPreserveRecentTool(%q) = true, want false", n)
		}
	}
}
