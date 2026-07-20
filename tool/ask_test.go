package tool

import (
	"context"
	"strings"
	"testing"
)

func TestAskUserQuestion(t *testing.T) {
	var asked []Question
	ask := func(_ context.Context, q Question) Answer {
		asked = append(asked, q)
		// Pick the first option's label.
		return Answer{Selected: []string{q.Options[0].Label}}
	}
	tl := NewAskUserQuestion(ask)

	input := `{"questions":[{"header":"DB","question":"Which database?","options":[{"label":"Postgres","description":"relational"},{"label":"SQLite","description":"embedded"}]}]}`
	if err := ValidateInput(tl.InputSchema(), []byte(input)); err != nil {
		t.Fatalf("schema: %v", err)
	}
	res, err := tl.Call(context.Background(), []byte(input), &ToolContext{})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if len(asked) != 1 || asked[0].Header != "DB" || len(asked[0].Options) != 2 {
		t.Fatalf("question not passed through: %+v", asked)
	}
	if !strings.Contains(res.Flatten(), "Which database?") || !strings.Contains(res.Flatten(), "Postgres") {
		t.Fatalf("answer not returned: %q", res.Flatten())
	}
}

func TestAskUserQuestionNoHandler(t *testing.T) {
	tl := NewAskUserQuestion(nil)
	res, _ := tl.Call(context.Background(), []byte(`{"questions":[{"question":"x","options":[{"label":"a"}]}]}`), &ToolContext{})
	if !res.IsError {
		t.Fatal("expected error when no handler configured")
	}
}
