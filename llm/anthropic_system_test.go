package llm

import "testing"

func TestBuildAnthropicSystemCacheBoundary(t *testing.T) {
	sys := []string{"a", "b", "c"}

	cases := []struct {
		name       string
		boundary   int
		wantBlocks int
		wantCached []bool // per-block: has cache_control
	}{
		{"no boundary (0) → single uncached", 0, 1, []bool{false}},
		{"negative → single uncached", -1, 1, []bool{false}},
		{"mid split → cached prefix + dynamic tail", 2, 2, []bool{true, false}},
		{"boundary == len → whole thing cached", 3, 1, []bool{true}},
		{"boundary > len → whole thing cached", 5, 1, []bool{true}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			blocks := buildAnthropicSystem(CompletionRequest{System: sys, DynamicBoundary: c.boundary})
			if len(blocks) != c.wantBlocks {
				t.Fatalf("blocks=%d want %d", len(blocks), c.wantBlocks)
			}
			for i, wantCached := range c.wantCached {
				gotCached := blocks[i].CacheControl != nil
				if gotCached != wantCached {
					t.Fatalf("block %d cached=%v want %v", i, gotCached, wantCached)
				}
			}
		})
	}
}
