package llm

import (
	"encoding/json"
	"testing"
)

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

// TestBuildAnthropicToolCache asserts the tool-schema breakpoint: opted-in
// callers (DynamicBoundary > 0) get cache_control on the LAST tool only, so
// Anthropic caches the whole tools section; boundary=0 stays marker-free.
func TestBuildAnthropicToolCache(t *testing.T) {
	tools := []ToolSchema{
		{Name: "read_file", Description: "read", InputSchema: map[string]any{"type": "object"}},
		{Name: "search_web", Description: "search", InputSchema: map[string]any{"type": "object"}},
		{Name: "bash", Description: "run", InputSchema: map[string]any{"type": "object"}},
	}
	cases := []struct {
		name       string
		boundary   int
		wantCached []bool // per tool: has cache_control
	}{
		{"no boundary → no tool breakpoint", 0, []bool{false, false, false}},
		{"opted in → last tool carries the breakpoint", 1, []bool{false, false, true}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := &anthropicProvider{cfg: Config{Model: "claude-x"}}
			raw, err := p.buildBody(CompletionRequest{
				System:          []string{"s"},
				DynamicBoundary: c.boundary,
				Messages:        []Message{UserText("hi")},
				Tools:           tools,
			}, false)
			if err != nil {
				t.Fatal(err)
			}
			var body struct {
				Tools []anthropicTool `json:"tools"`
			}
			if err := json.Unmarshal(raw, &body); err != nil {
				t.Fatal(err)
			}
			if len(body.Tools) != len(tools) {
				t.Fatalf("tools=%d want %d", len(body.Tools), len(tools))
			}
			for i, wantCached := range c.wantCached {
				got := body.Tools[i].CacheControl != nil
				if got != wantCached {
					t.Fatalf("tool %d cached=%v want %v", i, got, wantCached)
				}
			}
		})
	}
	// No tools → must not panic / must not inject anything.
	p := &anthropicProvider{cfg: Config{Model: "claude-x"}}
	if _, err := p.buildBody(CompletionRequest{
		System: []string{"s"}, DynamicBoundary: 1, Messages: []Message{UserText("hi")},
	}, false); err != nil {
		t.Fatal(err)
	}
}
