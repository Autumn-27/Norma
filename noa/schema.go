package noa

// CompressToolDescription is what the model reads next to the tool's schema.
//
// The brake in the second half is load-bearing. Tool schemas ship with every
// request whether or not a nudge fired, so the model can always see that
// Compress exists. Without an explicit instruction to wait, it may call the
// tool on its own initiative — and a summary written without the rules is an
// irreversible loss, because the range it replaced is gone from the view.
const CompressToolDescription = `Replace older conversation ranges with summaries you write. The originals are ` +
	`archived to disk first. Do NOT call this on your own initiative — wait until the context manager asks you to, ` +
	`and follow the rules it provides. A summary written without those rules permanently loses information.`

// CompressToolSchema is the tool's JSON Schema.
//
// Note what is NOT here: "content" carries no "type": "array".
//
// Providers without strict tool support (vLLM, some Qwen deployments) stringify
// nested array arguments. A schema that insists on an array rejects that shape
// before ParseCompressArgs ever sees it — and a turn's only compression attempt
// is lost to a validation error rather than being repaired. Omitting the type
// lets anything through to the lenient parser, which knows how to recover an
// array from a string.
func CompressToolSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"topic": map[string]any{
				"type":        "string",
				"description": "Optional short title applied to ranges that don't carry their own.",
			},
			"content": map[string]any{
				"description": "One or more ranges to compress into separate summary blocks. Ranges must be disjoint.",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"topic": map[string]any{
							"type":        "string",
							"description": "Short 3-5 word label for THIS range.",
						},
						"startId": map[string]any{
							"type":        "string",
							"description": "mNNNNN message ref, or bN block id, at the start of the range.",
						},
						"endId": map[string]any{
							"type":        "string",
							"description": "mNNNNN message ref, or bN block id, at the end of the range. Inclusive.",
						},
						"summary": map[string]any{
							"type":        "string",
							"description": "Self-contained summary that replaces the range in context.",
						},
					},
					"required": []any{"startId", "endId", "summary"},
				},
			},
			"summaryMaxChars": map[string]any{
				"type":        "number",
				"description": "Raise the per-summary limit above the 20000-char default. Use it when the content genuinely needs more detail — do not truncate critical information just to fit.",
			},
		},
		"required": []any{"content"},
	}
}
