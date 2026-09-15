package noa

import (
	"encoding/json"
	"strings"
	"testing"
)

func compressCall(id, callID, summary string) CoreMessage {
	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m00001", "endId": "m00009", "summary": summary}},
	})
	return CoreMessage{
		ID: id, Role: RoleAssistant, ContentType: CTToolCall,
		ToolCallID: callID, ToolName: CompressToolName, Text: string(args),
	}
}

func compressResult(id, callID string) CoreMessage {
	return CoreMessage{
		ID: id, Role: RoleTool, ContentType: CTToolResult,
		ToolCallID: callID, ToolName: CompressToolName, Text: "▣ noa | 12K → 3K tokens",
	}
}

func hideWith(msgs []CoreMessage, st CompressionState) []CoreMessage {
	io := hideCompressCalls(NodeIO{Messages: msgs, State: st}, PipelineContext{Config: DefaultConfig(200000)})
	return io.Messages
}

// A live block's Compress call stays: prune exempts it from orphan stripping
// precisely because it anchors the summary in the conversation flow.
func TestHideKeepsActiveBlockCall(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	st.Blocks = []CompressionBlock{b}

	msgs := []CoreMessage{
		compressCall("c1", "t1", strings.Repeat("s", 1000)),
		compressResult("r1", "t1"),
	}
	got := hideWith(msgs, st)
	if len(got) != 2 {
		t.Fatalf("got %v, want the live block's call kept", ids(got))
	}
}

// Once the block is gone the arguments are pure waste — and so is the result.
func TestHideDropsDeadBlockExchange(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	b.Active = false
	st.Blocks = []CompressionBlock{b}

	msgs := []CoreMessage{
		msg("keep", RoleUser, CTText, "hello"),
		compressCall("c1", "t1", strings.Repeat("s", 1000)),
		compressResult("r1", "t1"),
	}
	got := hideWith(msgs, st)
	if strings.Join(ids(got), ",") != "keep" {
		t.Fatalf("got %v, want the dead call and its result both dropped", ids(got))
	}
}

// A call with no block yet is one that just failed; keeping the last couple
// lets the model see its own mistake.
func TestHideKeepsRecentOrphanedCalls(t *testing.T) {
	st := CreateInitialState("s", "/a")
	msgs := []CoreMessage{
		compressCall("c1", "t1", strings.Repeat("s", 1000)),
		compressCall("c2", "t2", strings.Repeat("s", 1000)),
		compressCall("c3", "t3", strings.Repeat("s", 1000)),
	}
	got := hideWith(msgs, st)
	kept := strings.Join(ids(got), ",")
	if kept != "c2,c3" {
		t.Fatalf("got %v, want only the last %d orphaned calls", ids(got), KeepLastOrphaned)
	}
}

// The summary is already rendered in context and stored in the archive; a third
// copy inside the call arguments earns nothing.
func TestHideStubsSummaryOfKeptCall(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	st.Blocks = []CompressionBlock{b}

	long := strings.Repeat("detail ", 500)
	got := hideWith([]CoreMessage{compressCall("c1", "t1", long), compressResult("r1", "t1")}, st)

	var obj map[string]any
	if err := json.Unmarshal([]byte(got[0].Text), &obj); err != nil {
		t.Fatalf("stubbed arguments are not valid JSON: %v\n%s", err, got[0].Text)
	}
	arr := obj["content"].([]any)
	summary := arr[0].(map[string]any)["summary"].(string)
	if len([]rune(summary)) != SummaryStubChars {
		t.Fatalf("stub length = %d runes, want %d", len([]rune(summary)), SummaryStubChars)
	}
	if !strings.HasSuffix(summary, "…") {
		t.Fatalf("stub = %q, want it to end with an ellipsis", summary[len(summary)-10:])
	}
}

func TestHideLeavesShortSummaryAlone(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	st.Blocks = []CompressionBlock{b}

	short := "a brief summary"
	msgs := []CoreMessage{compressCall("c1", "t1", short), compressResult("r1", "t1")}
	got := hideWith(msgs, st)
	if got[0].Text != msgs[0].Text {
		t.Fatalf("a short summary was rewritten:\n%s", got[0].Text)
	}
}

// Mangling arguments would be worse than leaving them long.
func TestHideLeavesUnparseableArgumentsAlone(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	st.Blocks = []CompressionBlock{b}

	broken := CoreMessage{
		ID: "c1", Role: RoleAssistant, ContentType: CTToolCall,
		ToolCallID: "t1", ToolName: CompressToolName, Text: "not json at all",
	}
	got := hideWith([]CoreMessage{broken, compressResult("r1", "t1")}, st)
	if got[0].Text != "not json at all" {
		t.Fatalf("unparseable arguments were rewritten: %q", got[0].Text)
	}
}

// content must go back in whichever form it arrived, or the shape changes under
// the model between turns.
func TestHidePreservesStringifiedContent(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	st.Blocks = []CompressionBlock{b}

	inner, _ := json.Marshal([]map[string]any{{
		"startId": "m1", "endId": "m2", "summary": strings.Repeat("d", 500),
	}})
	outer, _ := json.Marshal(map[string]any{"content": string(inner)})
	msgs := []CoreMessage{{
		ID: "c1", Role: RoleAssistant, ContentType: CTToolCall,
		ToolCallID: "t1", ToolName: CompressToolName, Text: string(outer),
	}, compressResult("r1", "t1")}

	got := hideWith(msgs, st)
	var obj map[string]any
	if err := json.Unmarshal([]byte(got[0].Text), &obj); err != nil {
		t.Fatalf("rewritten arguments are not valid JSON: %v", err)
	}
	if _, ok := obj["content"].(string); !ok {
		t.Fatalf("content came back as %T, want it kept as a string", obj["content"])
	}
}

// Providers that prefix the arguments with prose must keep that prefix.
func TestHidePreservesTextPrefix(t *testing.T) {
	st := CreateInitialState("s", "/a")
	b := block("b1", 1, "gone")
	b.CompressCallID = "t1"
	st.Blocks = []CompressionBlock{b}

	args, _ := json.Marshal(map[string]any{
		"content": []map[string]any{{"startId": "m1", "endId": "m2", "summary": strings.Repeat("d", 500)}},
	})
	msgs := []CoreMessage{{
		ID: "c1", Role: RoleAssistant, ContentType: CTToolCall,
		ToolCallID: "t1", ToolName: CompressToolName,
		Text: "I'll compress that range.\n" + string(args),
	}, compressResult("r1", "t1")}

	got := hideWith(msgs, st)
	if !strings.HasPrefix(got[0].Text, "I'll compress that range.\n") {
		t.Fatalf("the prose prefix was lost: %q", got[0].Text[:40])
	}
}

func TestHideNodeDisabledWithoutCompressCalls(t *testing.T) {
	n := hideCompressCallsNode()
	io := NodeIO{Messages: []CoreMessage{msg("a", RoleUser, CTText, "x")}}
	if n.Enabled(io, PipelineContext{}) {
		t.Fatal("the node must be skipped when there are no Compress calls")
	}
}
