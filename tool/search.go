package tool

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var skipDirs = map[string]bool{".git": true, "node_modules": true, "vendor": true}

// NewGlob builds the Glob tool: find files by pattern, newest first.
func NewGlob() CoreTool {
	return Build(Spec{
		Name:        "Glob",
		Description: "Finds files whose path matches a glob pattern (supports ** for any depth, e.g. \"**/*.go\"). Returns paths sorted by modification time, newest first.",
		Prompt:      "Use this instead of `find`. The pattern matches paths relative to the search directory.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{"type": "string", "description": "Glob pattern, e.g. src/**/*.go"},
				"path":    map[string]any{"type": "string", "description": "Directory to search (default working dir)."},
			},
			"required": []any{"pattern"},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runGlob,
	})
}

func runGlob(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	root := tc.WorkingDir
	if in.Path != "" {
		root = resolvePath(tc, in.Path)
	}
	if root == "" {
		root = "."
	}
	type match struct {
		path string
		mod  int64
	}
	var matches []match
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			rel = p
		}
		if matchGlob(in.Pattern, rel) {
			var mod int64
			if info, e := d.Info(); e == nil {
				mod = info.ModTime().UnixNano()
			}
			matches = append(matches, match{p, mod})
		}
		return nil
	})
	sort.Slice(matches, func(i, j int) bool { return matches[i].mod > matches[j].mod })
	if len(matches) == 0 {
		return Text("No files found"), nil
	}
	var b strings.Builder
	for _, m := range matches {
		b.WriteString(m.path)
		b.WriteByte('\n')
	}
	return Text(Capture(tc, b.String())), nil
}

func matchGlob(pattern, path string) bool {
	pattern = filepath.ToSlash(pattern)
	path = filepath.ToSlash(path)
	if !strings.Contains(pattern, "/") {
		if ok, _ := filepath.Match(pattern, filepath.Base(path)); ok {
			return true
		}
	}
	return globSegments(strings.Split(pattern, "/"), strings.Split(path, "/"))
}

func globSegments(pat, name []string) bool {
	if len(pat) == 0 {
		return len(name) == 0
	}
	if pat[0] == "**" {
		for i := 0; i <= len(name); i++ {
			if globSegments(pat[1:], name[i:]) {
				return true
			}
		}
		return false
	}
	if len(name) == 0 {
		return false
	}
	if ok, _ := filepath.Match(pat[0], name[0]); !ok {
		return false
	}
	return globSegments(pat[1:], name[1:])
}

// NewGrep builds the Grep tool: regex content search (RE2).
func NewGrep() CoreTool {
	return Build(Spec{
		Name:        "Grep",
		Description: "Searches file contents with a regular expression (RE2 syntax). Returns matching lines, files, or counts. Use this instead of shelling out to grep/rg.",
		Prompt:      "output_mode: \"content\" (default) shows matching lines as file:line:text; \"files_with_matches\" lists paths; \"count\" shows per-file counts. Filter files with the glob parameter.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern":     map[string]any{"type": "string", "description": "RE2 regular expression."},
				"path":        map[string]any{"type": "string", "description": "File or directory to search."},
				"glob":        map[string]any{"type": "string", "description": "Only search files matching this glob, e.g. *.go"},
				"output_mode": map[string]any{"type": "string", "description": "content | files_with_matches | count"},
				"-i":          map[string]any{"type": "boolean", "description": "Case-insensitive."},
				"-n":          map[string]any{"type": "boolean", "description": "Show line numbers (content mode)."},
				"-A":          map[string]any{"type": "integer", "description": "Lines of context to show after each match (content mode)."},
				"-B":          map[string]any{"type": "integer", "description": "Lines of context to show before each match (content mode)."},
				"-C":          map[string]any{"type": "integer", "description": "Lines of context before and after each match (content mode)."},
			},
			"required": []any{"pattern"},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runGrep,
	})
}

func runGrep(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		Pattern    string `json:"pattern"`
		Path       string `json:"path"`
		Glob       string `json:"glob"`
		OutputMode string `json:"output_mode"`
		IgnoreCase bool   `json:"-i"`
		LineNums   *bool  `json:"-n"`
		After      int    `json:"-A"`
		Before     int    `json:"-B"`
		Context    int    `json:"-C"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	before, after := in.Before, in.After
	if in.Context > 0 {
		before, after = in.Context, in.Context
	}
	pat := in.Pattern
	if in.IgnoreCase {
		pat = "(?i)" + pat
	}
	re, err := regexp.Compile(pat)
	if err != nil {
		return Errorf(fmt.Sprintf("Error: invalid regex: %v", err)), nil
	}
	root := tc.WorkingDir
	if in.Path != "" {
		root = resolvePath(tc, in.Path)
	}
	if root == "" {
		root = "."
	}
	mode := in.OutputMode
	if mode == "" {
		mode = "content"
	}
	showLines := true
	if in.LineNums != nil {
		showLines = *in.LineNums
	}

	var results []grepHit
	scan := func(p, base string) {
		fh := scanFile(p, base, re, mode, showLines, before, after)
		if fh != nil {
			results = append(results, *fh)
		}
	}
	info, statErr := os.Stat(root)
	if statErr != nil {
		return Errorf(fmt.Sprintf("Error: %v", statErr)), nil
	}
	if info.IsDir() {
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if skipDirs[d.Name()] {
					return filepath.SkipDir
				}
				return nil
			}
			if in.Glob != "" {
				if ok, _ := filepath.Match(in.Glob, filepath.Base(p)); !ok {
					return nil
				}
			}
			scan(p, root)
			return nil
		})
	} else {
		scan(root, filepath.Dir(root))
	}
	if len(results) == 0 {
		return Text("No matches found"), nil
	}
	sort.Slice(results, func(i, j int) bool { return results[i].path < results[j].path })

	var b strings.Builder
	switch mode {
	case "files_with_matches":
		for _, r := range results {
			b.WriteString(r.path)
			b.WriteByte('\n')
		}
	case "count":
		for _, r := range results {
			fmt.Fprintf(&b, "%s: %d\n", r.path, r.count)
		}
	default:
		for _, r := range results {
			for _, ln := range r.lines {
				b.WriteString(ln)
				b.WriteByte('\n')
			}
		}
	}
	return Text(Capture(tc, b.String())), nil
}

// grepHit holds one file's grep results.
type grepHit struct {
	path  string
	lines []string
	count int
}

func scanFile(path, base string, re *regexp.Regexp, mode string, showLines bool, before, after int) *grepHit {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	rel := path
	if r, e := filepath.Rel(base, path); e == nil {
		rel = r
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var lines []string
	var matches []int
	for sc.Scan() {
		line := sc.Text()
		if re.MatchString(line) {
			matches = append(matches, len(lines))
		}
		lines = append(lines, line)
	}
	if len(matches) == 0 {
		return nil
	}
	res := &grepHit{path: rel, count: len(matches)}
	if mode != "content" {
		return res
	}

	isMatch := make(map[int]bool, len(matches))
	for _, m := range matches {
		isMatch[m] = true
	}
	// Collect the set of line indices to print (matches plus context), in order.
	var order []int
	seen := map[int]bool{}
	for _, m := range matches {
		lo, hi := m-before, m+after
		if lo < 0 {
			lo = 0
		}
		if hi >= len(lines) {
			hi = len(lines) - 1
		}
		for j := lo; j <= hi; j++ {
			if !seen[j] {
				seen[j] = true
				order = append(order, j)
			}
		}
	}
	sort.Ints(order)

	hasContext := before > 0 || after > 0
	prev := -2
	for _, j := range order {
		if hasContext && prev >= 0 && j != prev+1 {
			res.lines = append(res.lines, "--") // separate non-contiguous groups
		}
		sep := "-" // context line
		if isMatch[j] {
			sep = ":" // matching line
		}
		if showLines {
			res.lines = append(res.lines, fmt.Sprintf("%s%s%d%s%s", rel, sep, j+1, sep, lines[j]))
		} else {
			res.lines = append(res.lines, fmt.Sprintf("%s%s%s", rel, sep, lines[j]))
		}
		prev = j
	}
	return res
}
