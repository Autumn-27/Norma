package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Autumn-27/norma/permission"
	"golang.org/x/net/html"
)

// SearchResult is one web-search hit, normalized across backends.
type SearchResult struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	Description string `json:"description"`
	Position    int    `json:"position"`
}

// searchProvider is a pluggable web-search backend. Search-only by design —
// content extraction is handled elsewhere (WebFetch). Implementations must not
// panic; network/parse failures come back as an error.
type searchProvider interface {
	Name() string
	Search(ctx context.Context, query string, limit int) ([]SearchResult, error)
}

// WebSearchConfig selects and configures the active search backend.
type WebSearchConfig struct {
	// Backend chooses the provider: "ddgs" (DuckDuckGo scrape, no key),
	// "brave-free" (Brave Search API, needs BraveAPIKey), or "tavily" (Tavily
	// Search API, needs TavilyAPIKey). Empty defaults to "ddgs".
	Backend string
	// BraveAPIKey authenticates the "brave-free" backend. Required (and only
	// used) when Backend == "brave-free".
	BraveAPIKey string
	// TavilyAPIKey authenticates the "tavily" backend. Required (and only
	// used) when Backend == "tavily".
	TavilyAPIKey string
	// Proxy, when set, routes search requests through this http(s) proxy URL —
	// useful when the search endpoint is only reachable via a proxy. Empty = direct.
	Proxy string
	// CACert / InsecureTLS mirror WebFetchConfig: trust a proxy's CA in addition
	// to the system roots (CACert), or skip TLS verification entirely (InsecureTLS).
	// Prefer CACert. See WebFetchConfig for the full semantics.
	CACert      string
	InsecureTLS bool
}

// NewWebSearch builds the web_search tool for the configured backend. It returns
// an error when the config is unusable (unknown backend, or brave-free without a
// key) so a session fails fast at construction instead of at first call. The tool
// is network-facing, so it is not part of DefaultTools and requires permission.
func NewWebSearch(cfg WebSearchConfig) (CoreTool, error) {
	prov, err := newSearchProvider(cfg)
	if err != nil {
		return nil, err
	}
	return Build(Spec{
		Name:        "web_search",
		Description: "Search the web and return a ranked list of results (title, URL, description) — metadata only, up to 5 by default. Use WebFetch to read the full content of a specific URL. This makes an external network request.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string", "description": "The search query."},
				"limit": map[string]any{"type": "integer", "description": "Maximum number of results to return (1-20). Default 5.", "minimum": 1, "maximum": 20},
			},
			"required": []any{"query"},
		},
		// Read-only and safe to run concurrently, but network-facing: require approval.
		ReadOnly:   func(json.RawMessage) bool { return true },
		Concurrent: func(json.RawMessage) bool { return true },
		Permissions: func(_ context.Context, input json.RawMessage, _ permission.Context) permission.Decision {
			var in struct {
				Query string `json:"query"`
			}
			_ = json.Unmarshal(input, &in)
			return permission.AskUser("web search: " + in.Query)
		},
		Run: webSearchRunner(prov),
	}), nil
}

// newSearchProvider builds the search backend for cfg (shared by NewWebSearch and
// WebSearchProbe). One http.Client honors the proxy / TLS config. Returns an error
// on an unknown backend, or brave-free without a key.
func newSearchProvider(cfg WebSearchConfig) (searchProvider, error) {
	backend := strings.ToLower(strings.TrimSpace(cfg.Backend))
	if backend == "" {
		backend = "ddgs"
	}
	client := newFetchClient(WebFetchConfig{Proxy: cfg.Proxy, CACert: cfg.CACert, InsecureTLS: cfg.InsecureTLS})
	switch backend {
	case "ddgs":
		return &ddgsProvider{client: client, endpoint: ddgsEndpoint}, nil
	case "brave-free":
		key := strings.TrimSpace(cfg.BraveAPIKey)
		if key == "" {
			return nil, fmt.Errorf("web_search: backend %q requires a Brave Search API key", backend)
		}
		return &braveProvider{apiKey: key, client: client, endpoint: braveEndpoint}, nil
	case "tavily":
		key := strings.TrimSpace(cfg.TavilyAPIKey)
		if key == "" {
			return nil, fmt.Errorf("web_search: backend %q requires a Tavily API key", backend)
		}
		return &tavilyProvider{apiKey: key, client: client, endpoint: tavilyEndpoint}, nil
	default:
		return nil, fmt.Errorf("web_search: unknown backend %q (want \"ddgs\", \"brave-free\", or \"tavily\")", backend)
	}
}

// WebSearchProbe runs a one-off search with cfg and returns the results, bypassing
// the tool/permission layer. It's the entry point for a connectivity/config check
// (e.g. a "test" button): same backend selection + proxy/TLS handling as the
// web_search tool, so a success means the live tool would reach the backend too.
func WebSearchProbe(ctx context.Context, cfg WebSearchConfig, query string, limit int) ([]SearchResult, error) {
	prov, err := newSearchProvider(cfg)
	if err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 3
	}
	return prov.Search(ctx, query, limit)
}

func webSearchRunner(prov searchProvider) func(context.Context, json.RawMessage, *ToolContext) (Result, error) {
	return func(ctx context.Context, input json.RawMessage, _ *ToolContext) (Result, error) {
		var in struct {
			Query string `json:"query"`
			Limit int    `json:"limit"`
		}
		if err := json.Unmarshal(input, &in); err != nil {
			return Result{}, err
		}
		query := strings.TrimSpace(in.Query)
		if query == "" {
			return Errorf("Error: query must not be empty"), nil
		}
		limit := in.Limit
		if limit <= 0 {
			limit = 5
		}
		if limit > 20 {
			limit = 20
		}
		// Hard wall-clock cap so a slow/rate-limited backend can't hang the agent loop.
		cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		results, err := prov.Search(cctx, query, limit)
		if err != nil {
			return Errorf(fmt.Sprintf("Error searching web via %s: %s", prov.Name(), err.Error())), nil
		}
		out, _ := json.MarshalIndent(map[string]any{
			"backend": prov.Name(),
			"query":   query,
			"results": results,
		}, "", "  ")
		return Text(truncate(string(out), 50000)), nil
	}
}

// ─── DuckDuckGo (ddgs) ───────────────────────────────────────────────────────
// No API key. Scrapes the DuckDuckGo HTML endpoint (the same approach as the
// Python `ddgs` package) and parses results out of the returned markup. This is
// an unofficial endpoint: DuckDuckGo rate-limits by IP and may change the markup,
// so failures are surfaced as errors rather than treated as fatal.

const ddgsEndpoint = "https://html.duckduckgo.com/html/"

type ddgsProvider struct {
	client   *http.Client
	endpoint string
}

func (p *ddgsProvider) Name() string { return "ddgs" }

func (p *ddgsProvider) Search(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	form := url.Values{"q": {query}, "kl": {"wt-wt"}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	// A browser-like UA — the endpoint blocks obviously-automated clients.
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0")
	req.Header.Set("Referer", "https://html.duckduckgo.com/")
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusAccepted || resp.StatusCode == http.StatusTooManyRequests {
		return nil, fmt.Errorf("DuckDuckGo is rate-limiting (HTTP %d); try again later or switch backend", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DuckDuckGo returned HTTP %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	doc, err := html.Parse(strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("could not parse DuckDuckGo response: %w", err)
	}
	return parseDDGS(doc, limit), nil
}

// parseDDGS walks the DuckDuckGo HTML result page. Each result carries a title
// anchor with class "result__a" (its href is a //duckduckgo.com/l/?uddg=<target>
// redirect) and a snippet element with class "result__snippet". We collect titles
// and snippets in document order and pair them by index.
func parseDDGS(doc *html.Node, limit int) []SearchResult {
	type hit struct{ title, href string }
	var hits []hit
	var snippets []string

	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" && hasClass(n, "result__a") {
			hits = append(hits, hit{title: nodeText(n), href: unwrapDDGHref(attr(n, "href"))})
		}
		if n.Type == html.ElementNode && hasClass(n, "result__snippet") {
			snippets = append(snippets, nodeText(n))
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	var out []SearchResult
	for i, h := range hits {
		if len(out) >= limit {
			break
		}
		if h.href == "" {
			continue
		}
		desc := ""
		if i < len(snippets) {
			desc = snippets[i]
		}
		out = append(out, SearchResult{
			Title:       strings.TrimSpace(h.title),
			URL:         h.href,
			Description: strings.TrimSpace(desc),
			Position:    len(out) + 1,
		})
	}
	return out
}

// unwrapDDGHref resolves DuckDuckGo's redirect wrapper. Result links come back as
// "//duckduckgo.com/l/?uddg=<url-encoded target>&rut=…"; we return the decoded
// target. A plain http(s) href is returned unchanged.
func unwrapDDGHref(href string) string {
	if href == "" {
		return ""
	}
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	u, err := url.Parse(href)
	if err != nil {
		return href
	}
	if target := u.Query().Get("uddg"); target != "" {
		return target
	}
	if strings.HasPrefix(href, "http://") || strings.HasPrefix(href, "https://") {
		return href
	}
	return ""
}

// ─── Brave Search (free tier) ────────────────────────────────────────────────
// Official Brave Search API. Free tier is 2,000 queries/month (1 qps). Auth via
// the X-Subscription-Token header.

const braveEndpoint = "https://api.search.brave.com/res/v1/web/search"

type braveProvider struct {
	apiKey   string
	client   *http.Client
	endpoint string
}

func (p *braveProvider) Name() string { return "brave-free" }

func (p *braveProvider) Search(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	count := min(limit, 20) // Brave caps `count` at 20.
	q := url.Values{"q": {query}, "count": {fmt.Sprintf("%d", count)}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.endpoint+"?"+q.Encode(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", p.apiKey)
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Brave Search returned HTTP %d", resp.StatusCode)
	}
	var payload struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("could not parse Brave Search response: %w", err)
	}
	var out []SearchResult
	for _, r := range payload.Web.Results {
		if len(out) >= limit {
			break
		}
		out = append(out, SearchResult{
			Title:       r.Title,
			URL:         r.URL,
			Description: r.Description,
			Position:    len(out) + 1,
		})
	}
	return out, nil
}

// ─── Tavily Search ───────────────────────────────────────────────────────────
// Official Tavily Search API. Requires an API key (tvly-…). Returns results
// with extracted content summaries, making it a good complement to WebFetch.

const tavilyEndpoint = "https://api.tavily.com/search"

type tavilyProvider struct {
	apiKey   string
	client   *http.Client
	endpoint string
}

func (p *tavilyProvider) Name() string { return "tavily" }

func (p *tavilyProvider) Search(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	body, err := json.Marshal(map[string]any{
		"api_key":      p.apiKey,
		"query":        query,
		"max_results":  min(limit, 20),
		"search_depth": "basic",
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Tavily Search returned HTTP %d", resp.StatusCode)
	}
	var payload struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("could not parse Tavily Search response: %w", err)
	}
	var out []SearchResult
	for _, r := range payload.Results {
		if len(out) >= limit {
			break
		}
		out = append(out, SearchResult{
			Title:       r.Title,
			URL:         r.URL,
			Description: r.Content,
			Position:    len(out) + 1,
		})
	}
	return out, nil
}

// ─── small helpers ───────────────────────────────────────────────────────────

// hasClass reports whether an element node's class attribute contains class
// (whitespace-delimited match).
func hasClass(n *html.Node, class string) bool {
	for _, c := range strings.Fields(attr(n, "class")) {
		if c == class {
			return true
		}
	}
	return false
}

// nodeText returns the concatenated, whitespace-collapsed text of a node's subtree.
func nodeText(n *html.Node) string {
	var b strings.Builder
	textOnly(&b, n)
	return strings.Join(strings.Fields(b.String()), " ")
}
