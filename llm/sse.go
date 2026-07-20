package llm

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// sseScanner reads Server-Sent Events from a response body, yielding the
// (event, data) pair of each message. Anthropic uses named events; OpenAI uses
// bare data lines. Both are handled.
type sseScanner struct {
	r *bufio.Reader
}

func newSSEScanner(body io.Reader) *sseScanner {
	return &sseScanner{r: bufio.NewReaderSize(body, 64*1024)}
}

// next returns the next event's (type, data). event is "" for bare-data streams.
// Returns io.EOF at end of stream.
func (s *sseScanner) next() (event, data string, err error) {
	var buf bytes.Buffer
	for {
		line, rerr := s.r.ReadString('\n')
		if rerr != nil {
			if rerr == io.EOF && (buf.Len() > 0 || event != "") {
				return event, strings.TrimRight(buf.String(), "\n"), nil
			}
			return "", "", rerr
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if buf.Len() > 0 || event != "" {
				return event, strings.TrimRight(buf.String(), "\n"), nil
			}
			continue
		}
		switch {
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(line[len("event:"):])
		case strings.HasPrefix(line, "data:"):
			d := line[len("data:"):]
			d = strings.TrimPrefix(d, " ")
			buf.WriteString(d)
			buf.WriteByte('\n')
		default:
			// comment / unknown field — ignore
		}
	}
}
