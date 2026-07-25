package tool

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func TestSleepReturns(t *testing.T) {
	tl := NewSleep()
	if tl.Name() != "sleep" || !tl.IsReadOnly(nil) {
		t.Fatalf("unexpected identity: name=%q readonly=%v", tl.Name(), tl.IsReadOnly(nil))
	}
	res, err := tl.Call(context.Background(), json.RawMessage(`{"seconds":1}`), nil)
	if err != nil || res.IsError {
		t.Fatalf("call: err=%v res=%v", err, res)
	}
	if got := res.Flatten(); got != "slept 1s" {
		t.Fatalf("got %q, want %q", got, "slept 1s")
	}
}

func TestSleepInterrupted(t *testing.T) {
	// A large requested duration must return promptly once ctx is cancelled, not
	// block out the (clamped) wait.
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(20 * time.Millisecond); cancel() }()
	start := time.Now()
	res, err := NewSleep().Call(ctx, json.RawMessage(`{"seconds":3600}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("interrupt too slow: %v", elapsed)
	}
	if got := res.Flatten(); got != "sleep interrupted" {
		t.Fatalf("got %q, want interrupted", got)
	}
}
