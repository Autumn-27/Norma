package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestInProcessServerNamingAndCall(t *testing.T) {
	tools := InProcessServer("calc", []InProcessTool{{
		Name:        "add",
		Description: "add",
		Schema:      map[string]any{"type": "object"},
		Handler: func(_ context.Context, args json.RawMessage) (string, error) {
			var a struct{ A, B float64 }
			json.Unmarshal(args, &a)
			return "sum", nil
		},
	}})
	if len(tools) != 1 || tools[0].Name() != "mcp__calc__add" {
		t.Fatalf("naming wrong: %v", tools[0].Name())
	}
	res, _ := tools[0].Call(context.Background(), []byte(`{"a":1,"b":2}`), nil)
	if res.Flatten() != "sum" {
		t.Fatalf("call result: %q", res.Flatten())
	}
}

// fakeServer speaks just enough MCP JSON-RPC to satisfy initialize, tools/list,
// and tools/call, over the provided pipes.
func fakeServer(in io.Reader, out io.Writer) {
	sc := bufio.NewScanner(in)
	enc := json.NewEncoder(out)
	for sc.Scan() {
		var req map[string]any
		if json.Unmarshal(sc.Bytes(), &req) != nil {
			continue
		}
		id, hasID := req["id"]
		method, _ := req["method"].(string)
		if !hasID {
			continue // notification
		}
		var result any
		switch method {
		case "initialize":
			result = map[string]any{"protocolVersion": "2024-11-05"}
		case "tools/list":
			result = map[string]any{"tools": []any{map[string]any{
				"name": "echo", "description": "echoes", "inputSchema": map[string]any{"type": "object"},
			}}}
		case "tools/call":
			result = map[string]any{"content": []any{map[string]any{"type": "text", "text": "echoed!"}}}
		}
		enc.Encode(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
	}
}

func TestStdioClientOverPipes(t *testing.T) {
	cReader, sWriter := io.Pipe() // server -> client
	sReader, cWriter := io.Pipe() // client -> server
	go fakeServer(sReader, sWriter)

	ctx := context.Background()
	client, err := newClientWithConn(ctx, "srv", cReader, cWriter, nil)
	if err != nil {
		t.Fatalf("init: %v", err)
	}
	tools, err := client.Tools(ctx)
	if err != nil {
		t.Fatalf("tools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name() != "mcp__srv__echo" {
		t.Fatalf("tool naming: %+v", tools)
	}
	res, err := tools[0].Call(ctx, []byte(`{"msg":"hi"}`), nil)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(res.Flatten(), "echoed!") {
		t.Fatalf("call result: %q", res.Flatten())
	}
}
