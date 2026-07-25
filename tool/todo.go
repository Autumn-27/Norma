package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/Autumn-27/norma/permission"
)

// TodoStatus is the state of a todo item.
type TodoStatus string

const (
	TodoPending    TodoStatus = "pending"
	TodoInProgress TodoStatus = "in_progress"
	TodoCompleted  TodoStatus = "completed"
)

// Todo is one task in the agent's working plan.
type Todo struct {
	Content string     `json:"content"`
	Status  TodoStatus `json:"status"`
}

// TodoStore holds the current todo list across tool calls. It is safe for
// concurrent reads; the host can inspect progress via List.
type TodoStore struct {
	mu    sync.Mutex
	items []Todo
}

// NewTodoStore returns an empty store.
func NewTodoStore() *TodoStore { return &TodoStore{} }

// List returns a snapshot of the current todos.
func (s *TodoStore) List() []Todo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Todo(nil), s.items...)
}

func (s *TodoStore) set(items []Todo) {
	s.mu.Lock()
	s.items = items
	s.mu.Unlock()
}

// Tool returns the TodoWrite tool bound to this store. The agent uses it to
// plan and track multi-step work; each call replaces the whole list.
func (s *TodoStore) Tool() CoreTool {
	return Build(Spec{
		Name:        "TodoWrite",
		Description: "Records and updates a structured todo list for the current task. Use it to plan multi-step work and to mark progress — replace the whole list each call, marking exactly one item in_progress while working.",
		Prompt:      "status is one of: pending, in_progress, completed. Keep the list current: mark a task completed as soon as it's done; don't batch completions.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"todos": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"content": map[string]any{"type": "string"},
							"status":  map[string]any{"type": "string", "enum": []any{"pending", "in_progress", "completed"}},
						},
						"required": []any{"content", "status"},
					},
				},
			},
			"required": []any{"todos"},
		},
		ReadOnly:   func(json.RawMessage) bool { return false },
		Concurrent: func(json.RawMessage) bool { return false },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, input json.RawMessage, _ *ToolContext) (Result, error) {
			var in struct {
				Todos []Todo `json:"todos"`
			}
			if err := json.Unmarshal(input, &in); err != nil {
				return Result{}, err
			}
			s.set(in.Todos)
			return Text(renderTodos(in.Todos)), nil
		},
	})
}

func renderTodos(todos []Todo) string {
	if len(todos) == 0 {
		return "(todo list cleared)"
	}
	var b strings.Builder
	b.WriteString("Todos updated:\n")
	for _, t := range todos {
		box := "[ ]"
		switch t.Status {
		case TodoCompleted:
			box = "[x]"
		case TodoInProgress:
			box = "[~]"
		}
		fmt.Fprintf(&b, "%s %s\n", box, t.Content)
	}
	return b.String()
}
