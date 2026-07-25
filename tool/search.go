package tool

import (
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
const grepDescription = "A powerful search tool built on ripgrep\n" +
	"\n" +
	"Usage:\n" +
	"- ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n" +
	"- Supports full regex syntax (e.g., \"log.*Error\", \"function\\s+\\w+\")\n" +
	"- Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n" +
	"- Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n" +
	"- Use Agent tool for open-ended searches requiring multiple rounds\n" +
	"- Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n" +
	"- Prefer `head_limit` to avoid large result sets.\n" +
	"- Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`\n" +
	"- Prefer Grep over terminal rg/grep for codebase search unless you specifically need shell features."

func NewGrep() CoreTool {
	return Build(Spec{
		Name:        "Grep",
		Description: grepDescription,
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern":     map[string]any{"type": "string", "description": "RE2 regular expression to search for."},
				"path":        map[string]any{"type": "string", "description": "File or directory to search. Defaults to the working directory."},
				"glob":        map[string]any{"type": "string", "description": "Only search files matching this glob, e.g. *.go or *.{ts,tsx}"},
				"type":        map[string]any{"type": "string", "description": "Only search a file type, e.g. go, py, js, ts, rust, java. Unknown types are treated as an extension (type:foo → *.foo)."},
				"output_mode": map[string]any{"type": "string", "description": "files_with_matches (default) | content | count"},
				"-i":          map[string]any{"type": "boolean", "description": "Case-insensitive."},
				"-n":          map[string]any{"type": "boolean", "description": "Show line numbers (content mode). Default true."},
				"-A":          map[string]any{"type": "integer", "description": "Lines of context to show after each match (content mode)."},
				"-B":          map[string]any{"type": "integer", "description": "Lines of context to show before each match (content mode)."},
				"-C":          map[string]any{"type": "integer", "description": "Lines of context before and after each match (content mode)."},
				"multiline":   map[string]any{"type": "boolean", "description": "Multiline mode: . matches newlines and patterns may span lines. Default false."},
				"head_limit":  map[string]any{"type": "integer", "description": "Limit output to the first N results (lines/files/counts). Default 250; pass 0 for unlimited."},
				"offset":      map[string]any{"type": "integer", "description": "Skip the first N results before applying head_limit (paging). Default 0."},
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
		Type       string `json:"type"`
		OutputMode string `json:"output_mode"`
		IgnoreCase bool   `json:"-i"`
		LineNums   *bool  `json:"-n"`
		After      int    `json:"-A"`
		Before     int    `json:"-B"`
		Context    int    `json:"-C"`
		Multiline  bool   `json:"multiline"`
		HeadLimit  *int   `json:"head_limit"`
		Offset     int    `json:"offset"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	before, after := in.Before, in.After
	if in.Context > 0 {
		before, after = in.Context, in.Context
	}
	// RE2 inline flags: (?i) case-insensitive, (?s) dotall for multiline.
	flags := ""
	if in.IgnoreCase {
		flags += "i"
	}
	if in.Multiline {
		flags += "s"
	}
	pat := in.Pattern
	if flags != "" {
		pat = "(?" + flags + ")" + pat
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
		mode = "files_with_matches"
	}
	showLines := true
	if in.LineNums != nil {
		showLines = *in.LineNums
	}

	// Prefer the ripgrep binary when available; fall back to the Go walk below.
	if rg := RipgrepPath(); rg != "" {
		if res, ok := runWithRipgrep(rg, ripgrepReq{
			pattern: in.Pattern, glob: in.Glob, typ: in.Type, mode: mode,
			ignoreCase: in.IgnoreCase, multiline: in.Multiline, showLines: showLines,
			before: before, after: after, root: root, headLimit: in.HeadLimit, offset: in.Offset,
		}, tc); ok {
			return res, nil
		}
	}

	typeGlobs := rgTypeGlobs(in.Type)
	var results []grepHit
	scan := func(p, base string) {
		fh := scanFile(p, base, re, mode, showLines, in.Multiline, before, after)
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
			base := filepath.Base(p)
			if in.Glob != "" {
				if ok, _ := filepath.Match(in.Glob, base); !ok {
					return nil
				}
			}
			if len(typeGlobs) > 0 && !matchAnyGlob(typeGlobs, base) {
				return nil
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

	// Flatten to output entries per mode, then page with offset + head_limit.
	var lines []string
	switch mode {
	case "content":
		for _, r := range results {
			lines = append(lines, r.lines...)
		}
	case "count":
		for _, r := range results {
			lines = append(lines, fmt.Sprintf("%s: %d", r.path, r.count))
		}
	default: // files_with_matches
		for _, r := range results {
			lines = append(lines, r.path)
		}
	}
	headLimit := 250
	if in.HeadLimit != nil {
		headLimit = *in.HeadLimit
	}
	kept, more := applyHeadOffset(lines, headLimit, in.Offset)
	out := strings.Join(kept, "\n")
	if len(kept) > 0 {
		out += "\n"
	}
	if more > 0 {
		out += fmt.Sprintf("... [%d more result(s) — page with offset=%d, or head_limit=0 for all] ...\n", more, in.Offset+len(kept))
	}
	return Text(Capture(tc, out)), nil
}

// rgFileTypes maps a ripgrep-style file type to its globs (common subset).
var rgFileTypes = map[string][]string{
	"go":       {"*.go"},
	"py":       {"*.py", "*.pyi"},
	"python":   {"*.py", "*.pyi"},
	"js":       {"*.js", "*.jsx", "*.mjs", "*.cjs", "*.vue"},
	"ts":       {"*.ts", "*.tsx", "*.mts", "*.cts"},
	"rust":     {"*.rs"},
	"java":     {"*.java"},
	"kotlin":   {"*.kt", "*.kts"},
	"c":        {"*.c", "*.h"},
	"cpp":      {"*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx", "*.h"},
	"cs":       {"*.cs"},
	"rb":       {"*.rb"},
	"ruby":     {"*.rb"},
	"php":      {"*.php"},
	"swift":    {"*.swift"},
	"sh":       {"*.sh", "*.bash", "*.zsh"},
	"html":     {"*.html", "*.htm"},
	"css":      {"*.css", "*.scss", "*.sass", "*.less"},
	"json":     {"*.json"},
	"yaml":     {"*.yaml", "*.yml"},
	"toml":     {"*.toml"},
	"xml":      {"*.xml"},
	"md":       {"*.md", "*.markdown"},
	"markdown": {"*.md", "*.markdown"},
	"sql":      {"*.sql"},
	"proto":    {"*.proto"},
}

// rgTypeGlobs resolves a type name to its globs; an unknown type falls back to
// treating it as a bare extension (type:foo → *.foo). Empty type → nil.
func rgTypeGlobs(t string) []string {
	if t == "" {
		return nil
	}
	if g, ok := rgFileTypes[strings.ToLower(t)]; ok {
		return g
	}
	return []string{"*." + t}
}

func matchAnyGlob(globs []string, name string) bool {
	for _, g := range globs {
		if ok, _ := filepath.Match(g, name); ok {
			return true
		}
	}
	return false
}

// applyHeadOffset skips the first offset entries then keeps at most limit (0 =
// unlimited). more is the number of entries beyond the kept window.
func applyHeadOffset(lines []string, limit, offset int) (kept []string, more int) {
	if offset > 0 {
		if offset >= len(lines) {
			return nil, 0
		}
		lines = lines[offset:]
	}
	if limit > 0 && len(lines) > limit {
		return lines[:limit], len(lines) - limit
	}
	return lines, 0
}

// grepHit holds one file's grep results.
type grepHit struct {
	path  string
	lines []string
	count int
}

func scanFile(path, base string, re *regexp.Regexp, mode string, showLines, multiline bool, before, after int) *grepHit {
	data, err := os.ReadFile(path)
	if err != nil || isBinary(data) {
		return nil
	}
	rel := path
	if r, e := filepath.Rel(base, path); e == nil {
		rel = r
	}
	lines := strings.Split(string(data), "\n")
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1] // drop the empty element from a trailing newline
	}

	var matches []int
	if multiline {
		// Match against the whole file; a line "matches" if it overlaps any match span.
		lineStart := make([]int, len(lines))
		off := 0
		for i, ln := range lines {
			lineStart[i] = off
			off += len(ln) + 1 // + newline
		}
		hit := make([]bool, len(lines))
		for _, loc := range re.FindAllIndex(data, -1) {
			s, e := loc[0], loc[1]
			for i := range lines {
				ls, le := lineStart[i], lineStart[i]+len(lines[i])
				if s <= le && e > ls {
					hit[i] = true
				}
			}
		}
		for i, h := range hit {
			if h {
				matches = append(matches, i)
			}
		}
	} else {
		for i, ln := range lines {
			if re.MatchString(ln) {
				matches = append(matches, i)
			}
		}
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
