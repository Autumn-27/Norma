package memory

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Autumn-27/norma/tool"
)

func TestSaveScanReadRelevant(t *testing.T) {
	s := NewStore(t.TempDir())
	if err := s.Save(Memory{Name: "User Role", Description: "the user's job", Type: TypeUser, Content: "The user is a Go backend engineer."}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := s.Save(Memory{Name: "Test Policy", Description: "how to test", Type: TypeFeedback, Content: "Always run go test before claiming done."}); err != nil {
		t.Fatalf("save: %v", err)
	}

	mems, _ := s.Scan()
	if len(mems) != 2 {
		t.Fatalf("scan=%d", len(mems))
	}

	// Frontmatter round-trips.
	m, err := s.Read("User Role")
	if err != nil || m.Type != TypeUser || !strings.Contains(m.Content, "backend engineer") {
		t.Fatalf("read: %+v err=%v", m, err)
	}

	// Index lists both with a slug link.
	idx := s.Index()
	if !strings.Contains(idx, "[User Role](user-role.md)") || !strings.Contains(idx, "Test Policy") {
		t.Fatalf("index:\n%s", idx)
	}

	// Relevance picks the role memory for a role query.
	rel := s.Relevant("what is the user's role and job", 5)
	if len(rel) == 0 || rel[0].Name != "User Role" {
		t.Fatalf("relevant: %+v", rel)
	}

	// On-disk file has YAML frontmatter.
	data, _ := os.ReadFile(filepath.Join(s.Dir, "user-role.md"))
	if !strings.HasPrefix(string(data), "---\nname: User Role") {
		t.Fatalf("frontmatter:\n%s", data)
	}
}

func TestMemoryTools(t *testing.T) {
	s := NewStore(t.TempDir())
	rec, recall := RecordTool(s), RecallTool(s)

	res, _ := rec.Call(context.Background(), []byte(`{"name":"Deploy","description":"how to deploy","type":"project","content":"Deploy via make release."}`), &tool.ToolContext{})
	if res.IsError {
		t.Fatalf("record: %q", res.Flatten())
	}
	res, _ = recall.Call(context.Background(), []byte(`{"query":"how do I deploy"}`), &tool.ToolContext{})
	if !strings.Contains(res.Flatten(), "make release") {
		t.Fatalf("recall: %q", res.Flatten())
	}
}
