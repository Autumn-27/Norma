package llm

import "testing"

func useMsg(ids ...string) Message {
	var c []ContentBlock
	for _, id := range ids {
		c = append(c, ContentBlock{Type: BlockToolUse, ID: id, Name: "Read", Input: []byte(`{}`)})
	}
	return Message{Role: RoleAssistant, Content: c}
}

func resMsg(ids ...string) Message {
	var c []ContentBlock
	for _, id := range ids {
		c = append(c, ContentBlock{Type: BlockToolResult, ToolUseID: id,
			Content: []ContentBlock{TextBlock("ok")}})
	}
	return Message{Role: RoleUser, Content: c}
}

func textMsg(s string) Message {
	return Message{Role: RoleUser, Content: []ContentBlock{TextBlock(s)}}
}

// countBlocks reports how many tool blocks of a kind survived.
func countBlocks(msgs []Message, t BlockType) int {
	n := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Type == t {
				n++
			}
		}
	}
	return n
}

// A well-formed exchange, including a parallel batch, must pass through byte for
// byte: this runs on every request for every session.
func TestPairToolBlocksKeepsWellFormedHistory(t *testing.T) {
	in := []Message{
		textMsg("do it"),
		useMsg("a", "b"),
		resMsg("a", "b"),
		textMsg("done"),
	}
	out := PairToolBlocks(in)
	if len(out) != len(in) {
		t.Fatalf("well-formed history lost messages: %d -> %d", len(in), len(out))
	}
	if countBlocks(out, BlockToolUse) != 2 || countBlocks(out, BlockToolResult) != 2 {
		t.Fatalf("well-formed exchange was trimmed: %+v", out)
	}
}

// The four shapes a strict provider rejects. Every one of them has BOTH halves
// present, which is why an existence check alone let them through.
func TestPairToolBlocksDropsMalformedExchanges(t *testing.T) {
	cases := []struct {
		name string
		in   []Message
	}{
		{"half missing", []Message{useMsg("a")}},
		{"result only", []Message{resMsg("a")}},
		{"result before its call", []Message{resMsg("a"), useMsg("a")}},
		{"message between call and result", []Message{useMsg("a"), textMsg("hi"), resMsg("a")}},
		{"duplicate result", []Message{useMsg("a"), resMsg("a"), resMsg("a")}},
		{"duplicate call", []Message{useMsg("a"), useMsg("a"), resMsg("a")}},
	}
	for _, c := range cases {
		out := PairToolBlocks(c.in)
		if n := countBlocks(out, BlockToolUse) + countBlocks(out, BlockToolResult); n != 0 {
			t.Errorf("%s: %d tool blocks survived; the exchange is invalid and must be dropped whole", c.name, n)
		}
	}
}

// Dropping one exchange must not disturb its neighbours.
func TestPairToolBlocksDropsOnlyTheBadPair(t *testing.T) {
	in := []Message{
		useMsg("good"), resMsg("good"),
		textMsg("mid"),
		useMsg("bad"), textMsg("hi"), resMsg("bad"),
	}
	out := PairToolBlocks(in)
	if countBlocks(out, BlockToolUse) != 1 || countBlocks(out, BlockToolResult) != 1 {
		t.Fatalf("want exactly the good pair to survive, got %+v", out)
	}
	for _, m := range out {
		for _, b := range m.Content {
			if b.Type == BlockToolUse && b.ID != "good" {
				t.Fatalf("kept the bad call %q", b.ID)
			}
		}
	}
}
