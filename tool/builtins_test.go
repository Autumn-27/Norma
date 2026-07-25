package tool

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadHardening(t *testing.T) {
	dir := t.TempDir()

	// Directory → error.
	if r := run(t, NewRead(), dir, map[string]any{"file_path": "."}); !r.IsError {
		t.Fatal("reading a directory should error")
	}
	// Binary file → notice, not garbage.
	os.WriteFile(filepath.Join(dir, "bin"), []byte{0x00, 0x01, 0x02, 'h', 'i'}, 0o644)
	if r := run(t, NewRead(), dir, map[string]any{"file_path": "bin"}); !strings.Contains(r.Flatten(), "binary file") {
		t.Fatalf("binary not detected: %q", r.Flatten())
	}
	// Long line truncation.
	os.WriteFile(filepath.Join(dir, "long.txt"), []byte(strings.Repeat("a", 5000)), 0o644)
	out := run(t, NewRead(), dir, map[string]any{"file_path": "long.txt"}).Flatten()
	if !strings.Contains(out, "line truncated") {
		t.Fatalf("long line not truncated: len=%d", len(out))
	}
}

func TestWriteCreatedVsOverwrote(t *testing.T) {
	dir := t.TempDir()
	if r := run(t, NewWrite(), dir, map[string]any{"file_path": "f.txt", "content": "a"}); !strings.Contains(r.Flatten(), "Created") {
		t.Fatalf("expected Created: %q", r.Flatten())
	}
	if r := run(t, NewWrite(), dir, map[string]any{"file_path": "f.txt", "content": "b"}); !strings.Contains(r.Flatten(), "Overwrote") {
		t.Fatalf("expected Overwrote: %q", r.Flatten())
	}
	if r := run(t, NewWrite(), dir, map[string]any{"file_path": ".", "content": "x"}); !r.IsError {
		t.Fatal("writing over a directory should error")
	}
}

func TestMultiEditAtomic(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "f.go"), []byte("alpha beta gamma"), 0o644)

	// Two good edits apply in order.
	run(t, NewMultiEdit(), dir, map[string]any{
		"file_path": "f.go",
		"edits": []any{
			map[string]any{"old_string": "alpha", "new_string": "ALPHA"},
			map[string]any{"old_string": "gamma", "new_string": "GAMMA"},
		},
	})
	if b, _ := os.ReadFile(filepath.Join(dir, "f.go")); string(b) != "ALPHA beta GAMMA" {
		t.Fatalf("multiedit result: %q", b)
	}

	// A failing edit aborts the whole batch (file unchanged).
	r := run(t, NewMultiEdit(), dir, map[string]any{
		"file_path": "f.go",
		"edits": []any{
			map[string]any{"old_string": "beta", "new_string": "BETA"},
			map[string]any{"old_string": "nonexistent", "new_string": "x"},
		},
	})
	if !r.IsError {
		t.Fatal("expected error for unmatched edit")
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "f.go")); string(b) != "ALPHA beta GAMMA" {
		t.Fatalf("file should be unchanged after failed multiedit: %q", b)
	}
}

func TestLS(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("hi"), 0o644)
	os.MkdirAll(filepath.Join(dir, "sub"), 0o755)
	out := run(t, NewLS(), dir, map[string]any{}).Flatten()
	if !strings.Contains(out, "sub/") || !strings.Contains(out, "b.txt") {
		t.Fatalf("ls output: %q", out)
	}
}

func TestTodoWrite(t *testing.T) {
	store := NewTodoStore()
	st := store.Tool()
	st.Call(context.Background(), []byte(`{"todos":[{"content":"build","status":"in_progress"},{"content":"test","status":"pending"}]}`), &ToolContext{})
	items := store.List()
	if len(items) != 2 || items[0].Status != TodoInProgress {
		t.Fatalf("todos: %+v", items)
	}
}

func TestWebFetch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/html")
		w.Write([]byte(`<html><body><script>secret()</script><h1>Title</h1><p>Hello &amp; bye</p></body></html>`))
	}))
	defer srv.Close()
	raw, _ := json.Marshal(map[string]any{"url": srv.URL})
	res, err := NewWebFetch(WebFetchConfig{}).Call(context.Background(), raw, &ToolContext{})
	if err != nil {
		t.Fatalf("webfetch: %v", err)
	}
	out := res.Flatten()
	if !strings.Contains(out, "Title") || !strings.Contains(out, "Hello & bye") {
		t.Fatalf("text extraction: %q", out)
	}
	if strings.Contains(out, "secret") {
		t.Fatalf("script should be stripped: %q", out)
	}
}

func TestWebFetchRejectsStaticAssets(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.Write([]byte("console.log(1)"))
	}))
	defer srv.Close()

	// .js/.css (and with a query string), images, fonts → rejected before any request.
	for _, path := range []string{"/app.js", "/main.css", "/bundle.min.js?v=3", "/logo.png", "/font.woff2"} {
		raw, _ := json.Marshal(map[string]any{"url": srv.URL + path})
		res, err := NewWebFetch(WebFetchConfig{}).Call(context.Background(), raw, &ToolContext{})
		if err != nil {
			t.Fatalf("webfetch %s: %v", path, err)
		}
		if !res.IsError || !strings.Contains(res.Flatten(), "static asset") {
			t.Fatalf("%s should be rejected as a static asset, got: %q (isErr=%v)", path, res.Flatten(), res.IsError)
		}
	}
	if hits != 0 {
		t.Fatalf("no network request should be made for static assets, got %d", hits)
	}

	// A normal page path still works.
	raw, _ := json.Marshal(map[string]any{"url": srv.URL + "/docs/guide"})
	res, _ := NewWebFetch(WebFetchConfig{}).Call(context.Background(), raw, &ToolContext{})
	if res.IsError {
		t.Fatalf("normal page path should not be rejected: %q", res.Flatten())
	}
}

func TestProxyLikelyErr(t *testing.T) {
	// Proxy/MITM-caused failures → a direct retry may help.
	for _, s := range []string{
		`Get "https://x/": EOF`,
		"proxyconnect tcp: dial tcp 127.0.0.1:8788: connect: connection refused",
		"read tcp: connection reset by peer",
		"http2: server sent GOAWAY",
		"remote error: tls: handshake failure",
		"malformed HTTP response",
	} {
		if !proxyLikelyErr(errors.New(s)) {
			t.Errorf("want retry (proxy-likely) for %q", s)
		}
	}
	// Target/caller problems → a direct retry won't fix them.
	for _, s := range []string{
		"dial tcp: lookup nope.invalid: no such host",
		"context deadline exceeded",
		"context canceled",
		"net/http: request canceled (Client.Timeout exceeded while awaiting headers)",
	} {
		if proxyLikelyErr(errors.New(s)) {
			t.Errorf("want NO retry for %q", s)
		}
	}
	if proxyLikelyErr(nil) {
		t.Error("nil error must be false")
	}
}

// TestWebFetchProxyFallback proves a proxied fetch that fails for a proxy-caused
// reason (here: the proxy port is dead → connection refused) retries once directly
// (no proxy) and reaches the target. A successful fetch is not annotated.
func TestWebFetchProxyFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/html")
		w.Write([]byte(`<html><body><h1>Direct OK</h1></body></html>`))
	}))
	defer srv.Close()
	raw, _ := json.Marshal(map[string]any{"url": srv.URL})
	res, err := NewWebFetch(WebFetchConfig{Proxy: "http://127.0.0.1:1"}).Call(context.Background(), raw, &ToolContext{})
	if err != nil {
		t.Fatalf("webfetch: %v", err)
	}
	out := res.Flatten()
	if !strings.Contains(out, "Direct OK") {
		t.Fatalf("direct fallback should have fetched the page: %q", out)
	}
}

// TestWebFetchCompact verifies the HTML→Markdown output is token-lean: nested
// blocks don't leave blank lines, and per-line indentation is stripped, while
// content and list structure are preserved.
func TestWebFetchCompact(t *testing.T) {
	const page = `<html><body>
		<div>   <div>  <p>Para one</p>  </div>   </div>
		<div><p>Para two</p></div>
		<ul><li>alpha</li><li>beta</li></ul>
	</body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/html")
		w.Write([]byte(page))
	}))
	defer srv.Close()
	raw, _ := json.Marshal(map[string]any{"url": srv.URL})
	out := mustFetch(t, raw)
	if strings.Contains(out, "\n\n") {
		t.Fatalf("output must have no blank lines (compact): %q", out)
	}
	if strings.Contains(out, "\n ") || strings.Contains(out, " \n") {
		t.Fatalf("output must not keep spaces hugging newlines: %q", out)
	}
	for _, w := range []string{"Para one", "Para two", "- alpha", "- beta"} {
		if !strings.Contains(out, w) {
			t.Fatalf("content %q missing in:\n%s", w, out)
		}
	}
}

func TestWebFetchExtract(t *testing.T) {
	const page = `<html><head><meta name="generator" content="WP 6.0"></head><body>
		<!-- TODO remove backdoor /admin.php -->
		<h1>Login</h1>
		<form method="post" action="/login">
			<input type="text" name="user"><input type="password" name="pass">
			<input type="hidden" name="csrf" value="abc123">
		</form>
		<script src="/js/app.js"></script>
		<script>fetch("/api/v1/users")</script>
		<a href="/admin">admin</a>
	</body></html>`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/html")
		w.Write([]byte(page))
	}))
	defer srv.Close()

	// Without extract: no RECON section, markdown body only.
	raw, _ := json.Marshal(map[string]any{"url": srv.URL})
	out := mustFetch(t, raw)
	if strings.Contains(out, "RECON") {
		t.Fatalf("extract=false must not emit RECON: %q", out)
	}

	// With extract: comments, script src, inline endpoint, hidden field, meta all preserved.
	raw, _ = json.Marshal(map[string]any{"url": srv.URL, "extract": true})
	out = mustFetch(t, raw)
	for _, want := range []string{"RECON", "backdoor /admin.php", "/js/app.js", "/api/v1/users", "csrf(hidden=abc123)", "generator=WP 6.0", "POST /login"} {
		if !strings.Contains(out, want) {
			t.Fatalf("recon missing %q in:\n%s", want, out)
		}
	}
}

// TestWebFetchCACert proves CACert makes HTTPS through a self-signed (MITM-style)
// server verify: it fails without the CA and succeeds once the CA is trusted —
// the proper alternative to InsecureTLS.
func TestWebFetchCACert(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/html")
		w.Write([]byte("<h1>secure</h1>"))
	}))
	defer srv.Close()
	caFile := filepath.Join(t.TempDir(), "ca.pem")
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: srv.Certificate().Raw})
	if err := os.WriteFile(caFile, certPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(map[string]any{"url": srv.URL})

	// No CA → the self-signed cert fails verification.
	res, _ := NewWebFetch(WebFetchConfig{}).Call(context.Background(), raw, &ToolContext{})
	if out := res.Flatten(); !strings.Contains(out, "Error fetching URL") {
		t.Fatalf("expected TLS failure without CA, got: %q", out)
	}
	// CA trusted → verification passes, content fetched.
	res, _ = NewWebFetch(WebFetchConfig{CACert: caFile}).Call(context.Background(), raw, &ToolContext{})
	if out := res.Flatten(); !strings.Contains(out, "secure") {
		t.Fatalf("expected success with CA, got: %q", out)
	}
}

func mustFetch(t *testing.T, raw []byte) string {
	t.Helper()
	res, err := NewWebFetch(WebFetchConfig{}).Call(context.Background(), raw, &ToolContext{})
	if err != nil {
		t.Fatalf("webfetch: %v", err)
	}
	return res.Flatten()
}

// TestBashEnvInjection proves ToolContext.Env is layered into the Bash
// subprocess environment (and overrides inherited values).
func TestBashEnvInjection(t *testing.T) {
	t.Setenv("HTTP_PROXY", "inherited-should-be-overridden")
	raw, _ := json.Marshal(map[string]any{"command": "echo proxy=$HTTP_PROXY ca=$SSL_CERT_FILE"})
	tc := &ToolContext{Env: []string{
		"HTTP_PROXY=http://127.0.0.1:8080",
		"SSL_CERT_FILE=/etc/ca.pem",
	}}
	res, err := NewBash().Call(context.Background(), raw, tc)
	if err != nil {
		t.Fatalf("bash: %v", err)
	}
	out := res.Flatten()
	if !strings.Contains(out, "proxy=http://127.0.0.1:8080") || !strings.Contains(out, "ca=/etc/ca.pem") {
		t.Fatalf("env not injected/overridden: %q", out)
	}
}
