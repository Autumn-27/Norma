package tool

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/Autumn-27/norma/permission"
	"golang.org/x/net/html"
)

// WebFetchConfig tunes the network behavior of a WebFetch tool.
type WebFetchConfig struct {
	// Proxy, when set, routes requests through this http(s) proxy URL. Empty = direct.
	Proxy string
	// CACert, when set, is a path to a PEM CA certificate added to the system
	// trust pool. This lets WebFetch *verify* HTTPS fetched through a MITM/recording
	// proxy that re-signs certificates with its own CA — the proper alternative to
	// InsecureTLS, keeping verification on for every other connection. Preferred
	// over InsecureTLS; ignored when InsecureTLS is set.
	CACert string
	// InsecureTLS skips TLS certificate verification entirely (equivalent to
	// curl -k). A blunt fallback when no CA cert is available; prefer CACert.
	InsecureTLS bool
}

// NewWebFetch builds the WebFetch tool: fetches a URL over HTTP(S) and returns
// its content as Markdown (HTML is parsed and rendered; non-HTML is returned
// as-is). With extract=true it also appends a RECON section that preserves the
// signal HTML→Markdown normally discards — comments, <script> sources, inline
// endpoints, forms/inputs (including hidden) and meta tags — useful for security
// reconnaissance. cfg optionally routes requests through a proxy and/or skips
// TLS verification (see WebFetchConfig). It is network-facing, so it is not part
// of DefaultTools and requires permission to run.
func NewWebFetch(cfg WebFetchConfig) CoreTool {
	return Build(Spec{
		Name:        "WebFetch",
		Description: "Fetches a URL over HTTP(S) and returns its content as Markdown (HTML is converted; non-HTML returned as-is). Set extract=true to also surface HTML comments, <script> sources, inline endpoints, forms/inputs and meta tags. Use for documentation, API responses, and web pages the user references.",
		Prompt:      "Provide an absolute http(s) URL. Set extract=true to also append a RECON section (HTML comments, script sources, inline endpoints, forms/inputs incl. hidden, links, meta) — useful for security recon. Large responses are truncated. This tool makes an external network request.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"url":     map[string]any{"type": "string", "description": "The absolute http(s) URL to fetch."},
				"extract": map[string]any{"type": "boolean", "description": "When true, append a RECON section preserving HTML comments, <script src>, inline-script endpoints, forms/inputs (including hidden), links and meta tags. Default false."},
			},
			"required": []any{"url"},
		},
		// Network access is open-world: require approval rather than auto-allow.
		Permissions: func(_ context.Context, input json.RawMessage, _ permission.Context) permission.Decision {
			var in struct {
				URL string `json:"url"`
			}
			_ = json.Unmarshal(input, &in)
			return permission.AskUser("fetch external URL: " + in.URL)
		},
		Run: webFetchRunner(cfg),
	})
}

const maxFetchBytes = 2 << 20 // 2 MiB

// newFetchClient returns an HTTP client honoring the proxy / TLS config.
func newFetchClient(cfg WebFetchConfig) *http.Client {
	if cfg.Proxy == "" && !cfg.InsecureTLS && cfg.CACert == "" {
		return &http.Client{}
	}
	tr := &http.Transport{}
	if cfg.Proxy != "" {
		if pu, err := url.Parse(cfg.Proxy); err == nil {
			tr.Proxy = http.ProxyURL(pu)
		}
	}
	switch {
	case cfg.InsecureTLS:
		tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	case cfg.CACert != "":
		// Trust the proxy's CA in addition to the system roots, so certs it
		// re-signs verify while every other connection stays fully checked.
		if pem, err := os.ReadFile(cfg.CACert); err == nil {
			pool, _ := x509.SystemCertPool()
			if pool == nil {
				pool = x509.NewCertPool()
			}
			if pool.AppendCertsFromPEM(pem) {
				tr.TLSClientConfig = &tls.Config{RootCAs: pool}
			}
		}
	}
	return &http.Client{Transport: tr}
}

// proxyLikelyErr reports whether a fetch error is the kind a recording/MITM proxy
// introduces — a closed tunnel (EOF / connection reset), a failed CONNECT, or a
// TLS/HTTP2 mismatch — so a direct (no-proxy) retry may succeed. Name-resolution
// failures and caller cancellation/timeout are target- or caller-side problems a
// direct retry won't fix, so those return false (no retry).
func proxyLikelyErr(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	for _, n := range []string{"no such host", "context canceled", "context deadline", "timeout"} {
		if strings.Contains(s, n) {
			return false
		}
	}
	for _, p := range []string{
		"eof", "connection reset", "connection refused", "proxyconnect",
		"broken pipe", "forcibly closed", "malformed", "http2", "http/2",
		"tls:", "protocol error",
	} {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

func webFetchRunner(cfg WebFetchConfig) func(context.Context, json.RawMessage, *ToolContext) (Result, error) {
	client := newFetchClient(cfg)
	// direct is the no-proxy fallback used when a proxied fetch fails for a
	// proxy-caused reason (closed tunnel / failed CONNECT / TLS or HTTP2 mismatch):
	// it bypasses the configured proxy so a recording/MITM hiccup doesn't fail an
	// otherwise-reachable URL. nil when no proxy is configured (nothing to bypass).
	var direct *http.Client
	if cfg.Proxy != "" {
		direct = newFetchClient(WebFetchConfig{InsecureTLS: cfg.InsecureTLS})
	}
	return func(ctx context.Context, input json.RawMessage, _ *ToolContext) (Result, error) {
		var in struct {
			URL     string `json:"url"`
			Extract bool   `json:"extract"`
		}
		if err := json.Unmarshal(input, &in); err != nil {
			return Result{}, err
		}
		if !strings.HasPrefix(in.URL, "http://") && !strings.HasPrefix(in.URL, "https://") {
			return Errorf("Error: url must start with http:// or https://"), nil
		}
		cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(cctx, http.MethodGet, in.URL, nil)
		if err != nil {
			return Errorf("Error: " + err.Error()), nil
		}
		req.Header.Set("user-agent", "norma/0.4")
		resp, err := client.Do(req)
		// One direct retry when the proxied attempt failed for a proxy-caused reason.
		// A non-proxy error (bad host, caller timeout/cancel) is returned as-is — a
		// direct retry wouldn't fix it. The direct response is NOT captured by the proxy.
		if err != nil && direct != nil && proxyLikelyErr(err) {
			if r2, e2 := http.NewRequestWithContext(cctx, http.MethodGet, in.URL, nil); e2 == nil {
				r2.Header.Set("user-agent", "norma/0.4")
				if resp2, err2 := direct.Do(r2); err2 == nil {
					resp, err = resp2, nil
				} else {
					err = err2 // surface the direct failure (more representative of the target)
				}
			}
		}
		if err != nil {
			return Errorf("Error fetching URL: " + err.Error()), nil
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))

		header := fmt.Sprintf("[%d %s] %s\n", resp.StatusCode, http.StatusText(resp.StatusCode), in.URL)
		// Cheap fingerprint signal that HTML→Markdown would otherwise hide.
		if sv := resp.Header.Get("Server"); sv != "" {
			header += "Server: " + sv + "\n"
		}
		if xp := resp.Header.Get("X-Powered-By"); xp != "" {
			header += "X-Powered-By: " + xp + "\n"
		}

		var out string
		if strings.Contains(strings.ToLower(resp.Header.Get("content-type")), "html") {
			doc, perr := html.Parse(strings.NewReader(string(body)))
			if perr != nil { // html.Parse is lenient; this is effectively unreachable
				out = string(body)
			} else {
				out = htmlToMarkdown(doc)
				if in.Extract {
					if recon := extractRecon(doc); recon != "" {
						out += "\n\n" + recon
					}
				}
			}
		} else {
			out = string(body)
		}
		return Text(truncate(header+strings.TrimSpace(out), 50000)), nil
	}
}

var (
	reWS         = regexp.MustCompile(`[ \t]+`)
	reLineTrim   = regexp.MustCompile(` *\n *`) // strip spaces hugging a newline
	reBlankLines = regexp.MustCompile(`\n{2,}`) // collapse blank lines → one newline
	// reEndpoint matches quoted absolute URLs and root-relative paths in inline JS.
	reEndpoint = regexp.MustCompile("[\"'`](https?://[^\"'`\\s]+|/[A-Za-z0-9._~$&'()*+,;=:@%/?#\\[\\]-]{2,})[\"'`]")
)

// skipMarkdown are elements whose subtree carries no readable prose.
var skipMarkdown = map[string]bool{"script": true, "style": true, "noscript": true, "svg": true, "template": true}

// htmlToMarkdown renders the visible content of a parsed document as Markdown.
// Text nodes are already entity-decoded by the parser, so no unescaping is needed.
func htmlToMarkdown(doc *html.Node) string {
	var b strings.Builder
	renderNode(&b, doc)
	s := reWS.ReplaceAllString(b.String(), " ")
	s = reLineTrim.ReplaceAllString(s, "\n")   // drop indentation / trailing spaces per line
	s = reBlankLines.ReplaceAllString(s, "\n") // no blank lines — compact, token-lean output
	return strings.TrimSpace(s)
}

func renderNode(b *strings.Builder, n *html.Node) {
	switch n.Type {
	case html.TextNode:
		b.WriteString(strings.ReplaceAll(n.Data, "\n", " "))
		return
	case html.ElementNode:
		if skipMarkdown[n.Data] {
			return
		}
	default:
		if n.Type != html.DocumentNode {
			return
		}
	}
	tag := ""
	if n.Type == html.ElementNode {
		tag = n.Data
	}
	switch tag {
	case "h1", "h2", "h3", "h4", "h5", "h6":
		b.WriteString("\n\n" + strings.Repeat("#", int(tag[1]-'0')) + " ")
		renderChildren(b, n)
		b.WriteString("\n\n")
	case "title":
		b.WriteString("\n\n# ")
		renderChildren(b, n)
		b.WriteString("\n\n")
	case "p", "div", "section", "article", "header", "footer", "main", "ul", "ol", "table", "tr", "blockquote":
		b.WriteString("\n\n")
		renderChildren(b, n)
		b.WriteString("\n\n")
	case "br":
		b.WriteString("\n")
	case "hr":
		b.WriteString("\n\n---\n\n")
	case "li":
		b.WriteString("\n- ")
		renderChildren(b, n)
	case "a":
		var inner strings.Builder
		renderChildren(&inner, n)
		text := strings.TrimSpace(inner.String())
		if href := attr(n, "href"); href != "" && text != "" {
			fmt.Fprintf(b, "[%s](%s)", text, href)
		} else {
			b.WriteString(text)
		}
	case "img":
		if src := attr(n, "src"); src != "" {
			fmt.Fprintf(b, "![%s](%s)", attr(n, "alt"), src)
		}
	case "strong", "b":
		b.WriteString("**")
		renderChildren(b, n)
		b.WriteString("**")
	case "em", "i":
		b.WriteString("*")
		renderChildren(b, n)
		b.WriteString("*")
	case "code":
		b.WriteString("`")
		renderChildren(b, n)
		b.WriteString("`")
	case "pre":
		var inner strings.Builder
		textOnly(&inner, n)
		b.WriteString("\n\n```\n" + strings.TrimSpace(inner.String()) + "\n```\n\n")
	default:
		renderChildren(b, n)
	}
}

func renderChildren(b *strings.Builder, n *html.Node) {
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		renderNode(b, c)
	}
}

// extractRecon walks the document for security-relevant signal that Markdown
// rendering drops: comments, script sources, inline endpoints, forms, meta, links.
func extractRecon(doc *html.Node) string {
	var comments, scriptSrcs, inlineEndpoints, forms, metas, links []string
	seenLink := map[string]bool{}

	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		switch n.Type {
		case html.CommentNode:
			if c := oneLine(n.Data, 300); c != "" {
				comments = append(comments, c)
			}
		case html.ElementNode:
			switch n.Data {
			case "script":
				if src := attr(n, "src"); src != "" {
					scriptSrcs = append(scriptSrcs, src)
				} else {
					var body strings.Builder
					textOnly(&body, n)
					for _, m := range reEndpoint.FindAllStringSubmatch(body.String(), 40) {
						inlineEndpoints = append(inlineEndpoints, m[1])
					}
				}
			case "form":
				forms = append(forms, describeForm(n))
			case "meta":
				name := attr(n, "name")
				if name == "" {
					name = attr(n, "property")
				}
				if content := attr(n, "content"); name != "" && content != "" {
					metas = append(metas, name+"="+oneLine(content, 120))
				}
			case "a", "link":
				if h := attr(n, "href"); h != "" && !seenLink[h] {
					seenLink[h] = true
					links = append(links, h)
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	var b strings.Builder
	b.WriteString("=== RECON ===")
	reconSection(&b, "COMMENTS", comments)
	reconSection(&b, "SCRIPTS (src)", dedup(scriptSrcs))
	reconSection(&b, "INLINE ENDPOINTS", dedup(inlineEndpoints))
	reconSection(&b, "FORMS", forms)
	reconSection(&b, "META", metas)
	reconSection(&b, "LINKS", capSlice(links, 100))
	if b.Len() == len("=== RECON ===") {
		return "" // nothing worth reporting
	}
	return b.String()
}

// describeForm summarizes a <form> as "METHOD action → name(type[=value]) …".
// Hidden inputs keep their value (CSRF tokens, parameter defaults matter in recon).
func describeForm(form *html.Node) string {
	method := strings.ToUpper(attr(form, "method"))
	if method == "" {
		method = "GET"
	}
	var inputs []string
	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "input", "textarea", "select":
				typ := attr(n, "type")
				if typ == "" {
					typ = n.Data
				}
				desc := attr(n, "name") + "(" + typ
				if typ == "hidden" {
					if v := attr(n, "value"); v != "" {
						desc += "=" + oneLine(v, 60)
					}
				}
				inputs = append(inputs, desc+")")
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(form)
	return fmt.Sprintf("%s %s → %s", method, attr(form, "action"), strings.Join(inputs, " "))
}

// --- small helpers ---

func attr(n *html.Node, key string) string {
	for _, a := range n.Attr {
		if a.Key == key {
			return a.Val
		}
	}
	return ""
}

func textOnly(b *strings.Builder, n *html.Node) {
	if n.Type == html.TextNode {
		b.WriteString(n.Data)
		return
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		textOnly(b, c)
	}
}

func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}

func reconSection(b *strings.Builder, title string, items []string) {
	if len(items) == 0 {
		return
	}
	b.WriteString("\n\n## " + title + "\n")
	for _, it := range items {
		b.WriteString("- " + it + "\n")
	}
}

func dedup(in []string) []string {
	seen := map[string]bool{}
	out := in[:0:0]
	for _, s := range in {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

func capSlice(in []string, n int) []string {
	if len(in) <= n {
		return in
	}
	return append(in[:n:n], fmt.Sprintf("… (%d more)", len(in)-n))
}
