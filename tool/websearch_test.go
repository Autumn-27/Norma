package tool

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestNewWebSearchConfig(t *testing.T) {
	if _, err := NewWebSearch(WebSearchConfig{Backend: "ddgs"}); err != nil {
		t.Fatalf("ddgs backend should build: %v", err)
	}
	if _, err := NewWebSearch(WebSearchConfig{}); err != nil {
		t.Fatalf("empty backend should default to ddgs: %v", err)
	}
	if _, err := NewWebSearch(WebSearchConfig{Backend: "brave-free"}); err == nil {
		t.Fatal("brave-free without key should error")
	}
	if _, err := NewWebSearch(WebSearchConfig{Backend: "brave-free", BraveAPIKey: "k"}); err != nil {
		t.Fatalf("brave-free with key should build: %v", err)
	}
	if _, err := NewWebSearch(WebSearchConfig{Backend: "nope"}); err == nil {
		t.Fatal("unknown backend should error")
	}
}

func TestDDGSSearchParse(t *testing.T) {
	// Minimal fixture mimicking the html.duckduckgo.com/html/ result markup,
	// including the //duckduckgo.com/l/?uddg= redirect wrapper.
	page := `<html><body>
	<div class="result">
	  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x">First Result</a>
	  <a class="result__snippet">First snippet text.</a>
	</div>
	<div class="result">
	  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Second Result</a>
	  <a class="result__snippet">Second snippet.</a>
	</div>
	</body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil || r.Form.Get("q") == "" {
			t.Errorf("expected form q param, got err=%v form=%v", err, r.Form)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(page))
	}))
	defer srv.Close()

	p := &ddgsProvider{client: srv.Client(), endpoint: srv.URL}
	res, err := p.Search(context.Background(), "test query", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("want 2 results, got %d: %+v", len(res), res)
	}
	if res[0].URL != "https://example.com/a" {
		t.Errorf("uddg redirect not unwrapped: %q", res[0].URL)
	}
	if res[0].Title != "First Result" || res[0].Description != "First snippet text." {
		t.Errorf("bad result[0]: %+v", res[0])
	}
	if res[0].Position != 1 || res[1].Position != 2 {
		t.Errorf("positions not 1-indexed: %d %d", res[0].Position, res[1].Position)
	}

	// limit is honored.
	res, _ = p.Search(context.Background(), "test", 1)
	if len(res) != 1 {
		t.Errorf("limit=1 should yield 1 result, got %d", len(res))
	}
}

func TestBraveSearchParse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Subscription-Token"); got != "secret-key" {
			t.Errorf("missing/incorrect auth header: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"web":{"results":[
			{"title":"T1","url":"https://a.example","description":"D1"},
			{"title":"T2","url":"https://b.example","description":"D2"}
		]}}`))
	}))
	defer srv.Close()

	p := &braveProvider{apiKey: "secret-key", client: srv.Client(), endpoint: srv.URL}
	res, err := p.Search(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != 2 || res[0].Title != "T1" || res[1].URL != "https://b.example" {
		t.Fatalf("bad parse: %+v", res)
	}
	if res[0].Position != 1 {
		t.Errorf("position not 1-indexed: %d", res[0].Position)
	}
}

func TestBraveHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	p := &braveProvider{apiKey: "k", client: srv.Client(), endpoint: srv.URL}
	if _, err := p.Search(context.Background(), "q", 5); err == nil {
		t.Fatal("expected error on HTTP 401")
	}
}

func TestUnwrapDDGHref(t *testing.T) {
	cases := map[string]string{
		"//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.com%2Fy&rut=z": "https://x.com/y",
		"https://plain.example/path":                            "https://plain.example/path",
		"":                                                      "",
	}
	for in, want := range cases {
		if got := unwrapDDGHref(in); got != want {
			t.Errorf("unwrapDDGHref(%q) = %q, want %q", in, got, want)
		}
	}
	// Sanity: url encoding round-trips as expected.
	if _, err := url.Parse("//duckduckgo.com/l/?uddg=x"); err != nil {
		t.Fatal(err)
	}
}
