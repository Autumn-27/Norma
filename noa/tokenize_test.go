package noa

import "testing"

func TestDefaultCountTokens(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want int
	}{
		{"empty", "", 0},
		{"ascii exact multiple", "abcd", 1},
		{"ascii rounds up", "abcde", 2},
		{"ascii single", "a", 1},
		{"cjk one per rune", "中文", 2},
		{"cjk longer", "你好世界", 4},
		{"hiragana", "ひらがな", 4},
		{"hangul", "한국어", 3},
		// 4 CJK + 8 ASCII -> 4 + ceil(8/4) = 6
		{"mixed", "认证方案 auth.go", 4 + 2},
	}
	for _, c := range cases {
		if got := DefaultCountTokens(c.in); got != c.want {
			t.Errorf("%s: DefaultCountTokens(%q) = %d, want %d", c.name, c.in, got, c.want)
		}
	}
}

// Deliberate deviation from upstream: counting is per rune, so an astral-plane
// character counts once where a UTF-16 implementation would count it twice.
func TestDefaultCountTokensAstralPlane(t *testing.T) {
	if got := DefaultCountTokens("🎉🎉🎉🎉"); got != 1 {
		t.Fatalf("DefaultCountTokens(4 emoji) = %d, want 1 (4 runes / 4)", got)
	}
}

func TestEstimateTokensFast(t *testing.T) {
	cases := map[string]int{"": 0, "abcd": 1, "abcde": 2, "中文测试": 1}
	for in, want := range cases {
		if got := EstimateTokensFast(in); got != want {
			t.Errorf("EstimateTokensFast(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestCountMessageTokens(t *testing.T) {
	m := CoreMessage{Text: "abcdefgh"}
	if got := CountMessageTokens(m, nil); got != 2 {
		t.Fatalf("CountMessageTokens with default counter = %d, want 2", got)
	}
	if got := CountMessageTokens(m, func(string) int { return 99 }); got != 99 {
		t.Fatalf("CountMessageTokens with injected counter = %d, want 99", got)
	}
}

func TestIsCJK(t *testing.T) {
	for _, r := range []rune{'中', '文', 'あ', 'ア', '한'} {
		if !isCJK(r) {
			t.Errorf("isCJK(%q) = false, want true", r)
		}
	}
	for _, r := range []rune{'a', 'Z', '0', ' ', '\n', '·', '—'} {
		if isCJK(r) {
			t.Errorf("isCJK(%q) = true, want false", r)
		}
	}
}

func TestCeilDiv(t *testing.T) {
	cases := []struct{ n, d, want int }{
		{0, 4, 0}, {1, 4, 1}, {4, 4, 1}, {5, 4, 2}, {8, 4, 2}, {-3, 4, 0},
	}
	for _, c := range cases {
		if got := ceilDiv(c.n, c.d); got != c.want {
			t.Errorf("ceilDiv(%d,%d) = %d, want %d", c.n, c.d, got, c.want)
		}
	}
}
