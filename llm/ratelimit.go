package llm

import (
	"context"
	"iter"
	"sync"
	"time"
)

// RateLimit configures a shared request-rate limit for a Provider. A window with
// a zero value is unlimited. The limit is enforced across ALL callers that share
// the same Provider instance (one limiter per provider), so any number of
// concurrent agents using one Provider are bounded by a single limit.
//
// Only the logical request is counted: retries inside a single Stream call do
// not consume additional tokens.
type RateLimit struct {
	PerSecond float64 // max requests per second (0 = unlimited)
	PerMinute float64 // max requests per minute (0 = unlimited)
	// Burst is the per-second bucket capacity (how many may go at once before
	// smoothing). 0 = default to PerSecond.
	Burst float64
	// OnWait, when set, is called once per acquired request with how long the
	// caller blocked on the limiter (0 if it passed immediately). Observability
	// hook; nil disables it.
	OnWait func(blocked time.Duration)
}

// rateLimiter enforces PerSecond AND PerMinute simultaneously. Safe for
// concurrent use by many goroutines. A nil *rateLimiter is a no-op (unlimited).
type rateLimiter struct {
	sec    *bucket
	min    *bucket
	onWait func(time.Duration)
}

func newRateLimiter(rl RateLimit) *rateLimiter {
	if rl.PerSecond <= 0 && rl.PerMinute <= 0 {
		return nil // unlimited
	}
	l := &rateLimiter{onWait: rl.OnWait}
	if rl.PerSecond > 0 {
		burst := rl.Burst
		if burst <= 0 {
			burst = rl.PerSecond
		}
		l.sec = newBucket(rl.PerSecond, burst)
	}
	if rl.PerMinute > 0 {
		l.min = newBucket(rl.PerMinute/60.0, rl.PerMinute) // burst = a full minute
	}
	return l
}

// Wait blocks until one request is permitted by both windows, or ctx is done.
// A nil receiver returns immediately (unlimited).
func (l *rateLimiter) Wait(ctx context.Context) error {
	if l == nil {
		return nil
	}
	var d time.Duration
	if l.sec != nil {
		d = l.sec.reserve()
	}
	if l.min != nil {
		if dm := l.min.reserve(); dm > d {
			d = dm
		}
	}
	if d <= 0 {
		l.report(0)
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		// The caller bailed; the reserved tokens are not returned (a minor
		// over-count under cancellation), which only makes the limiter slightly
		// more conservative — never less.
		return ctx.Err()
	case <-t.C:
		l.report(d)
		return nil
	}
}

func (l *rateLimiter) report(d time.Duration) {
	if l.onWait != nil {
		l.onWait(d)
	}
}

// bucket is a token bucket: tokens refill at rate/sec up to burst; reserve()
// takes one token (going negative to reserve future capacity) and returns how
// long the caller must wait for it.
type bucket struct {
	mu     sync.Mutex
	rate   float64 // tokens per second
	burst  float64
	tokens float64
	last   time.Time
}

func newBucket(rate, burst float64) *bucket {
	return &bucket{rate: rate, burst: burst, tokens: burst, last: time.Now()}
}

func (b *bucket) reserve() time.Duration {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	b.tokens += now.Sub(b.last).Seconds() * b.rate
	if b.tokens > b.burst {
		b.tokens = b.burst
	}
	b.last = now
	b.tokens--
	if b.tokens >= 0 {
		return 0
	}
	return time.Duration(-b.tokens / b.rate * float64(time.Second))
}

// RateLimited wraps any Provider so that all of its Stream calls share one rate
// limit. Use it for custom Provider implementations; the built-in providers take
// a RateLimit via Config. Returns p unchanged if rl imposes no limit.
func RateLimited(p Provider, rl RateLimit) Provider {
	l := newRateLimiter(rl)
	if l == nil {
		return p
	}
	return &rateLimitedProvider{p: p, l: l}
}

type rateLimitedProvider struct {
	p Provider
	l *rateLimiter
}

func (rp *rateLimitedProvider) Stream(ctx context.Context, req CompletionRequest) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		if err := rp.l.Wait(ctx); err != nil {
			yield(StreamEvent{}, err)
			return
		}
		for ev, e := range rp.p.Stream(ctx, req) {
			if !yield(ev, e) {
				return
			}
		}
	}
}

func (rp *rateLimitedProvider) Complete(ctx context.Context, req CompletionRequest) (Message, string, Usage, error) {
	if err := rp.l.Wait(ctx); err != nil {
		return Message{}, "", Usage{}, err
	}
	return rp.p.Complete(ctx, req)
}
