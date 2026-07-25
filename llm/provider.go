package llm

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"strings"
)

// StreamEventType enumerates the normalized streaming events every adapter
// emits, regardless of wire format.
type StreamEventType string

const (
	SEMessageStart  StreamEventType = "message_start"
	SETextDelta     StreamEventType = "text_delta"
	SEThinkingDelta StreamEventType = "thinking_delta"
	SEToolUseStart  StreamEventType = "tool_use_start"
	SEToolInputJSON StreamEventType = "tool_input_delta"
	SEMessageDelta  StreamEventType = "message_delta" // carries stop_reason / usage
	SEMessageStop   StreamEventType = "message_stop"
)

// StreamEvent is one normalized model-level streaming event (FR-03.5).
type StreamEvent struct {
	Type       StreamEventType
	Text       string // text/thinking/tool_input JSON fragment
	ToolID     string // tool_use_start
	ToolName   string // tool_use_start
	StopReason string // message_delta
	Usage      Usage  // message_start / message_delta
}

// Format selects the wire protocol of a Provider.
type Format string

const (
	FormatAnthropic Format = "anthropic"
	FormatOpenAI    Format = "openai"
)

// Config configures the built-in providers (FR-03.4).
type Config struct {
	Format     Format
	BaseURL    string
	APIKey     string
	Model      string
	APIVersion string // Anthropic only; default 2023-06-01
	HTTPClient *http.Client
	// Proxy, when set, routes all model requests through the given proxy URL —
	// "http://", "https://" or "socks5://" (optionally with user:pass@). It only
	// affects the built-in client: when empty, the built-in client falls back to
	// the standard HTTP_PROXY / HTTPS_PROXY / NO_PROXY environment variables.
	// Ignored when the caller supplies its own HTTPClient.
	Proxy string
	// MaxRetries bounds transient-failure retries on the model request
	// establishment (connection reset / timeout / 429 / 5xx). 0 = default 3;
	// negative disables retrying. Only the pre-stream phase is retried (safe);
	// a mid-stream drop is surfaced as an error.
	MaxRetries int
	// RateLimit, when set, bounds the request rate shared across every caller of
	// the built Provider (per-second and/or per-minute). Retries do not count.
	RateLimit *RateLimit

	// ThinkingType toggles extended/reasoning mode via the request's
	// "thinking":{"type":...} field, for both formats. "" omits it (provider
	// default); "enabled"/"disabled" are sent verbatim.
	ThinkingType string
	// ReasoningEffort sets the reasoning strength. "" omits it. For OpenAI it maps
	// to "reasoning_effort"; for Anthropic to "output_config":{"effort":...}.
	// Typical values: low / medium / high / max.
	ReasoningEffort string

	limiter *rateLimiter // built from RateLimit by NewProvider; shared via the provider
}

// retries returns the effective retry count (0 → default 3; negative → 0).
func (c Config) retries() int {
	if c.MaxRetries == 0 {
		return 3
	}
	if c.MaxRetries < 0 {
		return 0
	}
	return c.MaxRetries
}

// Provider's sole responsibility is streaming a completion. Hosts may implement
// this interface to plug in private RPC backends (FR-03.7). The returned
// iterator yields normalized events; a terminal error is delivered as
// (zero, err) and ends iteration.
type Provider interface {
	Stream(ctx context.Context, req CompletionRequest) iter.Seq2[StreamEvent, error]
}

// NewProvider builds a built-in provider for cfg.Format. Missing APIKey/BaseURL
// fall back to standard environment variables and public endpoints.
func NewProvider(cfg Config) (Provider, error) {
	if cfg.HTTPClient == nil {
		client, err := defaultHTTPClient(cfg.Proxy)
		if err != nil {
			return nil, err
		}
		cfg.HTTPClient = client
	}
	if cfg.RateLimit != nil {
		cfg.limiter = newRateLimiter(*cfg.RateLimit) // shared by the provider (cfg copies the pointer)
	}
	switch cfg.Format {
	case FormatAnthropic:
		if cfg.APIKey == "" {
			cfg.APIKey = os.Getenv("ANTHROPIC_API_KEY")
		}
		if cfg.BaseURL == "" {
			cfg.BaseURL = envOr("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
		}
		if cfg.APIVersion == "" {
			cfg.APIVersion = "2023-06-01"
		}
		return &anthropicProvider{cfg: cfg}, nil
	case FormatOpenAI:
		if cfg.APIKey == "" {
			cfg.APIKey = os.Getenv("OPENAI_API_KEY")
		}
		if cfg.BaseURL == "" {
			cfg.BaseURL = envOr("OPENAI_BASE_URL", "https://api.openai.com/v1")
		}
		return &openaiProvider{cfg: cfg}, nil
	default:
		return nil, fmt.Errorf("llm: unknown format %q (use anthropic or openai)", cfg.Format)
	}
}

// defaultHTTPClient builds the built-in HTTP client. A non-empty proxy
// ("http://", "https://" or "socks5://") routes every request through it;
// an empty proxy falls back to the standard *_PROXY environment variables
// (net/http's ProxyFromEnvironment). The transport is cloned from
// http.DefaultTransport so it keeps the standard timeouts and connection pool.
func defaultHTTPClient(proxy string) (*http.Client, error) {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	if proxy == "" {
		tr.Proxy = http.ProxyFromEnvironment
		return &http.Client{Transport: tr}, nil
	}
	u, err := url.Parse(proxy)
	if err != nil {
		return nil, fmt.Errorf("llm: invalid proxy %q: %w", proxy, err)
	}
	switch u.Scheme {
	case "http", "https", "socks5": // supported by net/http's Transport.Proxy
	case "":
		return nil, fmt.Errorf("llm: proxy %q missing scheme (use http://, https:// or socks5://)", proxy)
	default:
		return nil, fmt.Errorf("llm: unsupported proxy scheme %q (use http, https or socks5)", u.Scheme)
	}
	tr.Proxy = http.ProxyURL(u)
	return &http.Client{Transport: tr}, nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// joinSystem renders the system segments into a single string.
func joinSystem(segs []string) string {
	parts := make([]string, 0, len(segs))
	for _, s := range segs {
		if s = strings.TrimSpace(s); s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n\n")
}
