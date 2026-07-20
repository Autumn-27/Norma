package llm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// doStream sends the streaming request with retries on transient failures and
// returns a 200 response (caller reads + closes the body). Only the request
// establishment phase is retried — connection reset / timeout / 429 / 5xx before
// any stream data is consumed — so retrying is safe (it can't duplicate
// partially-streamed assistant output). A mid-stream drop is the caller's to
// surface. Context cancellation is never retried and interrupts the backoff.
func doStream(ctx context.Context, cfg Config, url string, body []byte, setHeaders func(*http.Request), errPrefix string) (*http.Response, error) {
	// Rate limit once per logical request (outside the retry loop, so retries do
	// not consume additional tokens). Shared across all callers of this provider.
	if err := cfg.limiter.Wait(ctx); err != nil {
		return nil, err
	}
	retries := cfg.retries()
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		setHeaders(req)

		resp, err := cfg.HTTPClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		retryable := false
		switch {
		case err != nil:
			retryable = ctx.Err() == nil // transient network error (not cancellation)
		case isRetryableStatus(resp.StatusCode):
			retryable = true
		}
		if attempt >= retries || !retryable {
			if err != nil {
				return nil, err
			}
			data, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return nil, fmt.Errorf("%s: status %d: %s", errPrefix, resp.StatusCode, strings.TrimSpace(string(data)))
		}
		if resp != nil {
			resp.Body.Close()
		}
		if !backoffSleep(ctx, attempt) {
			return nil, ctx.Err() // context cancelled during backoff
		}
	}
}

func isRetryableStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, // 408
		http.StatusTooManyRequests,     // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout:      // 504
		return true
	}
	return false
}

// backoffSleep waits an exponential backoff (0.5s, 1s, 2s, … capped at 8s),
// interruptible by ctx. Returns false if the context was cancelled.
func backoffSleep(ctx context.Context, attempt int) bool {
	d := 500 * time.Millisecond * (1 << attempt)
	if d > 8*time.Second {
		d = 8 * time.Second
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
