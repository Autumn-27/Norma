package tool

import (
	"context"
	"encoding/json"
	"net/netip"
	"testing"
)

func TestIsPublicAddr(t *testing.T) {
	public := []string{"1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"}
	for _, s := range public {
		if !isPublicAddr(netip.MustParseAddr(s)) {
			t.Errorf("%s should count as public", s)
		}
	}
	blocked := []string{
		"127.0.0.1", "::1", // loopback
		"10.0.0.5", "192.168.1.1", "172.16.0.1", // private
		"169.254.169.254", // cloud metadata
		"fe80::1",         // link-local
		"0.0.0.0", "::",   // unspecified
		"224.0.0.1",  // multicast
		"100.64.0.1", // CGNAT
	}
	for _, s := range blocked {
		if isPublicAddr(netip.MustParseAddr(s)) {
			t.Errorf("%s must be blocked", s)
		}
	}
}

// The dial guard has to refuse the address, not just the URL string.
func TestGuardRefusesLoopback(t *testing.T) {
	if _, err := guardDialContext(context.Background(), "tcp", "127.0.0.1:80"); err == nil {
		t.Fatal("dialing loopback should be refused")
	}
	if _, err := guardDialContext(context.Background(), "tcp", "169.254.169.254:80"); err == nil {
		t.Fatal("dialing the metadata address should be refused")
	}
}

// End to end: WebFetch on a local URL returns an error result, not a page.
func TestWebFetchRefusesPrivateTarget(t *testing.T) {
	fetch := NewWebFetch(WebFetchConfig{})
	input, _ := json.Marshal(map[string]any{"url": "http://127.0.0.1:9/"})
	res, err := fetch.Call(context.Background(), input, &ToolContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.IsError {
		t.Fatalf("a fetch of 127.0.0.1 should fail, got %q", res.Flatten())
	}
	if !containsStr(res.Flatten(), "public internet") {
		t.Errorf("error should say why: %q", res.Flatten())
	}
}
