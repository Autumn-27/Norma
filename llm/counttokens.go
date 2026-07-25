package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// TokenCounter returns the number of input tokens a set of messages would
// consume. It can use a real API call (Anthropic's count_tokens endpoint) and
// falls back to estimation when the API is unavailable.
type TokenCounter func(ctx context.Context, msgs []Message) (int, error)

// NewAnthropicTokenCounter returns a TokenCounter backed by Anthropic's
// /v1/messages/count_tokens endpoint, which gives exact input-token counts. Use
// a cheap model (e.g. Haiku) in cfg.Model for counting.
func NewAnthropicTokenCounter(cfg Config) TokenCounter {
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com"
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = "2023-06-01"
	}
	return func(ctx context.Context, msgs []Message) (int, error) {
		body, err := json.Marshal(map[string]any{
			"model":    cfg.Model,
			"messages": MessagesForAPI(msgs),
		})
		if err != nil {
			return 0, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+"/v1/messages/count_tokens", bytes.NewReader(body))
		if err != nil {
			return 0, err
		}
		req.Header.Set("content-type", "application/json")
		req.Header.Set("x-api-key", cfg.APIKey)
		req.Header.Set("anthropic-version", cfg.APIVersion)
		resp, err := cfg.HTTPClient.Do(req)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			return 0, fmt.Errorf("count_tokens: status %d: %s", resp.StatusCode, string(data))
		}
		var out struct {
			InputTokens int `json:"input_tokens"`
		}
		if err := json.Unmarshal(data, &out); err != nil {
			return 0, err
		}
		return out.InputTokens, nil
	}
}
