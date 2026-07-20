// Package mcp integrates Model Context Protocol tools (FR-10). It speaks
// JSON-RPC 2.0 over a newline-delimited stdio transport to external MCP servers
// and also supports in-process servers (Go functions). Every MCP tool is
// adapted to a tool.CoreTool named mcp__<server>__<tool> (FR-10.2/10.3).
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	"github.com/Autumn-27/norma/tool"
)

// ToolName builds the three-segment MCP tool name.
func ToolName(server, name string) string { return "mcp__" + server + "__" + name }

// --- in-process server ---

// InProcessTool is a Go-native tool exposed under the MCP naming convention.
type InProcessTool struct {
	Name        string
	Description string
	Schema      map[string]any
	Handler     func(ctx context.Context, args json.RawMessage) (string, error)
}

// InProcessServer adapts a set of in-process tools to CoreTools (FR-10.3).
func InProcessServer(server string, tools []InProcessTool) []tool.CoreTool {
	out := make([]tool.CoreTool, 0, len(tools))
	for _, it := range tools {
		it := it
		out = append(out, tool.Build(tool.Spec{
			Name:        ToolName(server, it.Name),
			Description: it.Description,
			Schema:      it.Schema,
			Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
				return permission.AskUser("call MCP tool " + ToolName(server, it.Name) + "?")
			},
			Run: func(ctx context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
				text, err := it.Handler(ctx, in)
				if err != nil {
					return tool.Errorf("Error: " + err.Error()), nil
				}
				return tool.Text(text), nil
			},
		}))
	}
	return out
}

// --- JSON-RPC transport ---

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int    `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string { return fmt.Sprintf("mcp rpc error %d: %s", e.Code, e.Message) }

// conn is a JSON-RPC 2.0 connection over a newline-delimited byte stream. It is
// transport-agnostic (stdio, pipe, socket) for testability.
type conn struct {
	w       io.Writer
	mu      sync.Mutex
	nextID  int
	pending map[int]chan rpcResponse
	closeCh chan struct{}
	once    sync.Once
}

func newConn(r io.Reader, w io.Writer) *conn {
	c := &conn{w: w, pending: map[int]chan rpcResponse{}, closeCh: make(chan struct{})}
	go c.readLoop(r)
	return c
}

func (c *conn) readLoop(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var resp rpcResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			continue // notifications / non-response frames are ignored
		}
		if resp.ID == 0 {
			continue
		}
		c.mu.Lock()
		ch := c.pending[resp.ID]
		delete(c.pending, resp.ID)
		c.mu.Unlock()
		if ch != nil {
			ch <- resp
		}
	}
	c.once.Do(func() { close(c.closeCh) })
}

func (c *conn) notify(method string, params any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.writeLocked(rpcRequest{JSONRPC: "2.0", Method: method, Params: params})
}

func (c *conn) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.mu.Lock()
	c.nextID++
	id := c.nextID
	ch := make(chan rpcResponse, 1)
	c.pending[id] = ch
	err := c.writeLocked(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	c.mu.Unlock()
	if err != nil {
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.closeCh:
		return nil, fmt.Errorf("mcp: connection closed")
	case resp := <-ch:
		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	}
}

func (c *conn) writeLocked(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = c.w.Write(data)
	return err
}

// --- client ---

// Client is a connection to one MCP server.
type Client struct {
	server string
	conn   *conn
	closer io.Closer
}

type remoteTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// initialize performs the MCP handshake.
func (c *Client) initialize(ctx context.Context) error {
	_, err := c.conn.call(ctx, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "norma", "version": "0.2"},
	})
	if err != nil {
		return err
	}
	return c.conn.notify("notifications/initialized", map[string]any{})
}

// Tools lists the server's tools and adapts them to CoreTools.
func (c *Client) Tools(ctx context.Context) ([]tool.CoreTool, error) {
	raw, err := c.conn.call(ctx, "tools/list", map[string]any{})
	if err != nil {
		return nil, err
	}
	var res struct {
		Tools []remoteTool `json:"tools"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, err
	}
	out := make([]tool.CoreTool, 0, len(res.Tools))
	for _, rt := range res.Tools {
		out = append(out, c.wrap(rt))
	}
	return out, nil
}

func (c *Client) wrap(rt remoteTool) tool.CoreTool {
	schema := rt.InputSchema
	if schema == nil {
		schema = map[string]any{"type": "object"}
	}
	full := ToolName(c.server, rt.Name)
	return tool.Build(tool.Spec{
		Name:        full,
		Description: rt.Description,
		Schema:      schema,
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.AskUser("call MCP tool " + full + "?")
		},
		Run: func(ctx context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var args any
			if len(in) > 0 {
				_ = json.Unmarshal(in, &args)
			}
			raw, err := c.conn.call(ctx, "tools/call", map[string]any{"name": rt.Name, "arguments": args})
			if err != nil {
				return tool.Errorf("Error: " + err.Error()), nil
			}
			var res struct {
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
				IsError bool `json:"isError"`
			}
			if err := json.Unmarshal(raw, &res); err != nil {
				return tool.Errorf("Error: bad MCP response: " + err.Error()), nil
			}
			var text string
			for _, blk := range res.Content {
				text += blk.Text
			}
			return tool.Result{Content: []llm.ContentBlock{llm.TextBlock(text)}, IsError: res.IsError}, nil
		},
	})
}

// Close shuts down the client's transport.
func (c *Client) Close() error {
	if c.closer != nil {
		return c.closer.Close()
	}
	return nil
}
