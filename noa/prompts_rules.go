package noa

// The four load-bearing compression rules, ported VERBATIM from acp-kernel
// (src/compression-rules.ts, MIT, Copyright © 2026 ranxianglei). Upstream marks
// them "tuned over months of production use" and gates any override behind an
// explicit risk acknowledgement.
//
// They are not translated. Translation IS an override, and these texts decide
// whether a summary keeps the paths, signatures and decisions that make the
// archive reachable later.
//
// FOUR deliberate deviations, all factual corrections rather than rewording:
// upstream assumes a `decompress` tool that noa does not have, so text telling
// the model to use it would send it after something that does not exist.
//
//	1. "(or you, after decompressing)"
//	   -> "(or you, after reading the archive file)"
//	2. "...jump back via decompress to the exact original."
//	   -> "...locate the exact original inside the archive file."
//	3. "This lets a later decompress target the right block by relevance..."
//	   -> "This lets a later reader grep the archive directory and find the right
//	      block by relevance..."
//	4. "...cannot be grepped or decompressed-to later."
//	   -> "...cannot be grepped or located in the archive later."
//
// Plus the tool's name, lowercase upstream and CamelCase here.
//
// prompts_rules_test.go asserts byte equality with the upstream source except
// for exactly these edits — the check is the only thing standing between a
// careless future edit and silently degraded summaries.

const CompressPhilosophy = `Compression Philosophy:
- All compression serves the primary task, but be frugal.
- Context capacity is precious. Save context by compressing consumed outputs, not by avoiding tools.
- Compress by need, not by percentage.
- Work from summaries, not raw tool outputs. All listed ranges (user prompts, tool outputs, code, logs, exploration, intermediate steps) should be compressed to summary format — the ONLY exceptions are protected content, content the current step is actively using, or critical content you cannot reconstruct.`

// HowToCompressRules governs what a summary must keep and may drop.
//
// The archive section at the end is ADDED by noa (it replaces nothing). It
// changes the economics of forgetting: with the originals on disk the model can
// drop verbatim bulk far more aggressively, provided the summary still says
// THAT something happened and WHERE to look. Without that last constraint the
// archive becomes unreachable — present on disk, but with nothing pointing at
// it.
const HowToCompressRules = `HOW TO COMPRESS

When you call ` +
	"`" +
	`Compress` +
	"`" +
	`, the summary you write becomes the only record of the replaced conversation. Make it self-contained and complete: every user request, experiment purpose, and work task in the range must be accurately captured. A later reader (or you, after reading the archive file) should be able to continue the task WITHOUT needing the original. The summary records the PAST as of this block's creation: label recorded task state as history ("TASK AS OF THIS BLOCK: ...") — never as a live instruction, so a later reader treats it as settled context, not something to re-execute. Write plain text with real unicode characters; never copy \\uXXXX escape sequences or JSON-escaped fragments out of tool output.

KEEP VERBATIM — never paraphrase or abbreviate these:
- Full file paths with line numbers, directory prefix on every mention (` +
	"`" +
	`lib/hooks.ts:347` +
	"`" +
	`, ` +
	"`" +
	`src/index.ts:12-18` +
	"`" +
	`, ` +
	"`" +
	`gatenet_v3/model.py:45` +
	"`" +
	`). Never abbreviate to a bare filename (` +
	"`" +
	`hooks.ts` +
	"`" +
	`, ` +
	"`" +
	`model.py` +
	"`" +
	`) — they are ambiguous and cannot be grepped or located in the archive later.
- Function, class, and type signatures (exact names, params, return types) AND critical code lines that encode logic — the line that IS the finding, not just the function name (e.g. ` +
	"`" +
	`kv_keys += define_gate * a_key[i](emb)` +
	"`" +
	` is more useful than "see model_kvnet.py").
- Error messages and stack traces (exact text — you need the literal string to grep for it later).
- Key details from reports and analyses — not just the conclusion. Keep the comparison numbers and the mechanism, not "X is worse" alone (write "1.76× PPL gap because KV store is static", not "KVNet underperforms").
- Decisions and their rationale ("chose X over Y because Z" — the "because" is load-bearing; without it the decision looks arbitrary).
- Constraints discovered ("must support Node 22", "no new dependencies", "AGENTS.md forbids ` +
	"`" +
	`as any` +
	"`" +
	`").
- Exact values: versions, config keys, thresholds, magic numbers.
- User intent — quote short user messages verbatim ONLY WITH their message ref, e.g. ` +
	"`" +
	`User said (m00132): "ship it tonight"` +
	"`" +
	`. Without a verifiable ref, paraphrase (` +
	"`" +
	`user previously asked (paraphrased): ...` +
	"`" +
	`) — this is the one exception to the verbatim rule above; never present a reconstructed or half-remembered phrase as a verbatim quote. When the message is too long to quote, preserve intent with extra care: do not change scope, constraints, priorities, acceptance criteria, or requested outcomes. Quotes are historical records, never current directives. Losing these changes the task itself.
- The user's overall goal and any changes to it — the big-picture objective plus how it evolved during the compressed range. Each summary must reflect the goal as it stood at the end of the range, including pivots (e.g., "initially: fix bug X → pivoted to: refactor module Y after discovering root cause"). Losing the goal or its evolution makes all subsequent work appear unmotivated.
- Purpose behind each significant action — preserve not just what was done but why: the hypothesis behind each experiment, the question behind each exploration, the task goal behind each work action. Without purpose, the summary reads as disconnected technical steps with no through-line.
- Open questions and unresolved TODOs — losing these changes what work appears to remain.
- Message refs of key anchors (` +
	"`" +
	`m00420` +
	"`" +
	`, ` +
	"`" +
	`m00510–m00520` +
	"`" +
	`) — they let you or a later reader locate the exact original inside the archive file.

DROP — extract the signal, discard the vessel:
- Verbose logs (build/test/` +
	"`" +
	`npm` +
	"`" +
	` output) once you have captured the error line or the result.
- Duplicate file reads once the needed content is recorded.
- Consumed exploration — search hits, agent return values, successful tool outputs — once you have extracted the facts you need (same rule as dead-ends, but nothing went wrong; the content is simply spent).
- Dead-end exploration — but PRESERVE the lesson in one line: "tried X, failed because Y".
- Back-and-forth discussion and self-corrections once the final position is captured (keep the outcome, drop the journey to it).
- Repeated status checks (` +
	"`" +
	`git status` +
	"`" +
	`, ` +
	"`" +
	`ls` +
	"`" +
	`) once state is known.

For each significant item you DROP (scripts, reports, large analyses, long tool outputs), add a one-line CONTENT description of what it covers — not where it lives. Bad: "probe script at /path/probe_kvnet.py". Good: "probe_kvnet.py: tests n-gram baseline, generation quality, long-range dependency, position sensitivity, op pipeline, QUERY attention." This lets a later reader grep the archive directory and find the right block by relevance, not by guessing locations.

PRIORITY — when the summary must be compact, preserve in this order:
1. User's overall goal, goal evolution, intent, and hard constraints (losing these changes the task).
2. Decisions and rationale.
3. Exact technical artifacts: paths, signatures, errors, values.
4. Conclusions and key findings.
5. Lessons learned: what failed and why.

Write dense, scannable bullets — not narrative prose. If the range spans distinct concerns (request → findings → decision), group bullets under short thematic headers so a reader can scan to the part they need. Every line must earn its place. Do not mimic the style of existing summaries in context; follow these rules.` + "\n\n" + archiveAddendum

const archiveAddendum = `THE ARCHIVE CHANGES THE COST OF FORGETTING

The original content of every range you compress is written to a markdown file
before the summary replaces it. Nothing is destroyed. The summary that lands in
context ends with a tag carrying the path:

  <noa-archive block="b5" tier="2" range="m00012-m00160" path="/abs/path.md"/>

If you later need the exact original text, read that path with the Read tool, or
grep the archive directory. Never ask the user to re-paste history — it is on disk.

This means you can be MORE aggressive than the rules above suggest about dropping
verbatim bulk: long tool outputs, full file contents, repeated reasoning. What you
must still keep in the summary is everything needed to know THAT something happened
and WHERE to look: file paths, decisions and their rationale, error signatures,
constraints, the user's goal, and the message refs of key anchors. A summary that
loses those leaves the archive unreachable — you will not know there is anything
to go read.`

// Tier2Rules and Tier3Rules are verbatim: neither mentions decompress,
// search_context or acp_status, so neither needed adapting. They also need no
// archive addendum — a tier nudge sends HowToCompressRules alongside them, so
// the archive guidance already reaches the model.
const Tier2Rules = `TIER 2 COMPRESSION — DISTILLATION

You are compressing historical summaries (not raw conversation). These summaries have already captured the details. Your job is to DISTILL them: extract only what matters for future work, discard the process.

KEEP — these are the only things that survive distillation:
- Decisions and their rationale ("chose X over Y because Z" — the "because" is load-bearing).
- Final outcomes: version numbers shipped, PR numbers merged/closed, bugs fixed or deferred.
- Key lessons: what failed and why ("tried X, failed because Y"). These prevent repeating mistakes.
- Critical constraints discovered ("must support Node 22", "AGENTS.md forbids as any").
- Design decisions with architectural impact ("chose compress-as-anchor over synthetic messages because prefix cache").
- User quotes and task state only as attributed history: keep the source ref with any user quote; never carry a tier-1 "CURRENT TASK" claim forward as a live directive — relabel it "TASK AS OF THIS BLOCK".
- Whether content is OBSOLETE or SUPERSEDED — mark with one line: "[SUPERSEDED by PR #NNN]" or "[OBSOLETE: deleted in vX.Y.Z]". Do NOT keep the obsolete content's details — just the marker and reason.
- Function/class/type names and module paths that are the SUBJECT of the work — e.g., "fixed filterCompressedRanges in prune.ts", "added SessionStateRegistry in state.ts". Not exact line numbers or full signatures — just enough to LOCATE the code without searching.
- Exploration findings: if a block was exploratory with no decision, keep the CONCLUSION in one line ("explored X, not viable because Y"). Do not keep the exploration process.

DROP — these were useful during the work but are no longer needed:
- Exact line numbers, diffs, verbose function signatures, full code listings.
- Build/deploy process details, test execution steps.
- Review process details (who reviewed, what rounds, test counts).
- Verbose logs, command output, intermediate debugging steps.

FORMAT:
- Start each distilled block with a source header line:
  ` +
	"`" +
	`Source: bN+bM+... (XK→YK tok, Zx). [original topic]` +
	"`" +
	`
  Example: ` +
	"`" +
	`Source: b5+b7 (56K+44K→268 tok, 375x). [Tool-result recap + publish]` +
	"`" +
	`
- 3-5 bullet points per source block, each a self-contained fact.
- Dense, scannable — no narrative prose.
- Start with the outcome, not the process: "v1.13.0 shipped (7 PRs bundled)" not "implemented 7 PRs then reviewed then merged".
- Cross-block synthesis: if multiple source blocks cover the same topic (same PR, same feature, same bug), MERGE them into a single group of bullets. Do not repeat the same fact from different blocks — keep it once under the most relevant source header.

SIZE TARGET: 50-150 tokens per source block (excluding the header). If you can't fit it in 150 tokens, you're keeping too much process. If a block has nothing worth keeping (pure noise), output just the header followed by "[no actionable content]."`

const Tier3Rules = `TIER 3 COMPRESSION — ULTRA-CONDENSATION

You are compressing distilled summaries (Tier 2) into ultra-condensed facts (Tier 3). The distilled summaries already contain only decisions and outcomes. Your job is to reduce them to bare factual references.

PRIORITY — when a source block has more facts than the size target allows, keep in this order:
1. Shipped outcomes (versions released, PRs merged) — these are permanent record.
2. Open work (PRs/issues still pending) — these may need follow-up.
3. Key decisions with architectural impact ("chose X over Y because Z").
4. Critical constraints ("must support Node 22").
Drop everything else. Tier 3 is a lookup index, not a knowledge base.

FORMAT:
- Start with a source header line:
  ` +
	"`" +
	`Source: bN+bM+... (XK→YK tok, Zx). [original topic]` +
	"`" +
	`
- Output 1-3 facts per source block. Each fact is a single line: subject + outcome.
- No explanations, no rationale, no process — just the fact.
- Format: "[PR/Issue/Version] — [outcome in ≤8 words]"
- Merge related facts from different source blocks if they concern the same topic.

EXAMPLES:
- "v1.13.0 shipped — quality gate + GC fix (7 PRs)"
- "PR #196 merged — preserve-first-user (supersedes #169)"
- "Bug 1214 fixed — compress consumed all user messages"
- "Chose compress-as-anchor — prefix cache benefit over synthetic injection"
- "Constraint: AGENTS.md forbids as any — never suppress types"

DROP:
- Multi-sentence context. If a fact needs >1 sentence, it's too detailed for Tier 3.
- Lessons learned ("tried X, failed because Y") — drop UNLESS the failure is likely to recur and the block is <30 days old.
- Design rationale details — keep the decision, drop the "because" unless it's a critical constraint.
- Anything marked [OBSOLETE] or [SUPERSEDED] — drop entirely, note "[N blocks obsolete]" in the summary.

SIZE TARGET: 30-60 tokens per source block (including header). For a batch of N source blocks, total output ≈ N × 40 tokens. If a source block has only one trivial fact, output just the header + one line.`
