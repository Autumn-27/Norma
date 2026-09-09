package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// An unset Config keeps the historical defaults: 3 establishment retries on the
// exponential ladder, 2 empty-response retries on the same ladder.
func TestRetryDefaultsUnchanged(t *testing.T) {
	var c Config
	if got := c.retries(); got != 3 {
		t.Fatalf("retries=%d, want 3", got)
	}
	if got := c.emptyRetries(); got != emptyResponseRetries {
		t.Fatalf("emptyRetries=%d, want %d", got, emptyResponseRetries)
	}
	for attempt, want := range []time.Duration{500 * time.Millisecond, time.Second, 2 * time.Second, 8 * time.Second} {
		a := attempt
		if a == 3 {
			a = 5 // past the cap
		}
		if got := c.retryDelay(a); got != want {
			t.Fatalf("retryDelay(%d)=%v, want %v", a, got, want)
		}
		if got := c.emptyRetryDelay(a); got != want {
			t.Fatalf("emptyRetryDelay(%d)=%v, want %v", a, got, want)
		}
	}
}

// A configured interval replaces the ladder with a FIXED wait, on both knobs,
// and negative counts disable the respective retry entirely.
func TestRetryFixedIntervalAndDisable(t *testing.T) {
	c := Config{
		MaxRetries: 5, RetryInterval: 250 * time.Millisecond,
		EmptyResponseRetries: -1, EmptyResponseInterval: 700 * time.Millisecond,
	}
	if got := c.retries(); got != 5 {
		t.Fatalf("retries=%d, want 5", got)
	}
	if got := c.emptyRetries(); got != 0 {
		t.Fatalf("emptyRetries=%d, want 0 (disabled)", got)
	}
	for _, attempt := range []int{0, 1, 7} {
		if got := c.retryDelay(attempt); got != 250*time.Millisecond {
			t.Fatalf("retryDelay(%d)=%v, want fixed 250ms", attempt, got)
		}
		if got := c.emptyRetryDelay(attempt); got != 700*time.Millisecond {
			t.Fatalf("emptyRetryDelay(%d)=%v, want fixed 700ms", attempt, got)
		}
	}
	if got := (Config{MaxRetries: -1}).retries(); got != 0 {
		t.Fatalf("retries=%d, want 0 (disabled)", got)
	}
}

// MaxRetries bounds the establishment attempts end to end: a server that always
// 503s is hit exactly MaxRetries+1 times, and the fixed interval keeps the test
// fast (the default ladder would spend 0.5+1+2s here).
func TestDoStreamHonorsConfiguredRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	cfg := Config{
		Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m",
		MaxRetries: 1, RetryInterval: time.Millisecond,
	}
	if _, err := doStream(context.Background(), cfg, srv.URL, []byte(`{}`), func(*http.Request) {}, "openai"); err == nil {
		t.Fatal("want error after retries are exhausted")
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d, want 2 (1 attempt + 1 retry)", calls.Load())
	}
}

// EmptyResponseRetries bounds the empty-completion re-requests; the fixed
// interval applies between them.
func TestOpenAIStreamHonorsConfiguredEmptyRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(emptyStop))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{
		Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m",
		EmptyResponseRetries: 4, EmptyResponseInterval: time.Millisecond,
	})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if len(msg.Content) != 0 {
		t.Fatalf("want empty message after exhaustion, got %+v", msg.Content)
	}
	if calls.Load() != 5 {
		t.Fatalf("calls=%d, want 5 (1 attempt + 4 retries)", calls.Load())
	}
}

// A disabled empty-response retry surfaces the first empty completion as-is.
func TestOpenAIStreamEmptyRetryDisabled(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(emptyStop))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{
		Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m",
		EmptyResponseRetries: -1,
	})
	collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if calls.Load() != 1 {
		t.Fatalf("calls=%d, want 1 (retry disabled)", calls.Load())
	}
}
