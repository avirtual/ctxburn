# ctxburn

**Grade your Claude Code sessions on token cost.** Point it at your transcripts; it tells you which sessions burned money, why, and what to do about it.

No dependencies. No setup. It reads the JSONL transcripts Claude Code already writes to `~/.claude/projects`.

```
$ ctxburn
========================================================================
SESSION GRADER — 18 sessions · 1122M replay tokens · $4,300 total
grades: A:1  B:1  C:3  D:1  F:12
grade = last-10-turn avg context (absolute tokens/turn = cost).
========================================================================

  [F]  ca65462f  (aws-eks-infra)  $316
       last-10 avg 488K · 542 turns · 0 compactions · 133.3M replay
     FINDINGS:
       - Cost: $315.83 on claude-opus — context replay+write $266 (84%),
         output $49 (16%). (Output is 0.5% of tokens but 16% of cost.)
       - Last-10-turn avg context: 488K re-sent every turn (the cost driver).
       - Ran 542 turns without a single compaction.
       - Largest dead weight: Read aws-eks-infra — 4.8M tokens (fetched 6x).
     SUGGESTIONS:
       > Compact or /clear early and repeatedly.
       > Drop large one-shot outputs once consumed.
       > Pin or summarize frequently-needed files instead of re-reading.
```

## Why

The bill for an agentic coding session is **~96% context replay and <1% generated output.** Every turn re-sends the entire accumulated context. So:

> **cost ≈ (context carried per turn) × (number of turns)** — the area under the context-growth curve.

A long session that never compacts is a triangle: each late turn drags the whole history. Turn 500 can cost 30× turn 1. The lever that matters is **context discipline** — compact or `/clear` early and often. Plugins that shave tool output or tune caching are optimizing the rounding error; ctxburn measures the thing that actually moves the bill.

## Install

Not on PyPI yet — install from source:

```bash
git clone https://github.com/avirtual/ctxburn
cd ctxburn
pipx install .          # gives you the `ctxburn` command on PATH
# or for development:
pip install -e .
# or run it without installing:
python3 -m ctxburn.cli --help
```

## Usage

```bash
ctxburn                        # grade the last 7 days under ~/.claude/projects
ctxburn /path/to/project       # a specific project folder
ctxburn --all                  # ignore the time window
ctxburn --since 30             # last 30 days
ctxburn --session 845a0984     # full report for one session id
ctxburn --top 10               # detail the 10 most-improvable sessions
ctxburn --window 1000000       # context window for the ceiling-risk flag
ctxburn --by-project           # roll cost/grades up by project, with restart boot-floor
ctxburn --json                 # machine-readable output
```

By default it focuses on the sessions worth improving (grade C/D/F) and collapses the efficient ones to a one-liner.

`--by-project` rolls the per-session view up by project (cwd): one row per project with its boot count (= sessions, each a cold context load), worst/avg grade, total replay, and **boot-floor** — the one-time cache *write* a restart re-pays on each cold boot, summed across boots. It's the cross-session lens: a restart-heavy agent's bill is `boots × (boot-floor + the read-replay tail each session then grows)`, which no single-session view shows. Per-session detail also now prints a **"where to cut"** line — the turn index where context first crossed each grade band, i.e. where a `/clear` would have capped the tail.

## How the grade works

The grade is the **last-10-turn average context** — the tokens you were re-sending every turn by the end of the session. This is absolute (tokens = dollars), **not** normalized by window: replaying 200K tokens costs the same whether your window is 200K or 1M, so window size doesn't change your grade. It's used only for a secondary "you're near the ceiling" risk flag.

| grade | last-10 avg context | meaning |
|-------|---------------------|---------|
| A | < 40K  | lean — system + a few files + recent history |
| B | < 80K  | healthy |
| C | < 130K | getting heavy |
| D | < 180K | should have compacted a while ago |
| F | ≥ 180K | re-sending a near-full window every turn |

Modifiers: a 150+ turn session with no compaction drops a grade; a session that compacted and ended lean earns one back.

## Cost

ctxburn prices all four token tiers (uncached input, output, cache-write, cache-read) per model, matched on the `model` field in the transcript. **These are public list prices** (`opus` / `sonnet` / `haiku`) — enterprise contracts differ, so edit `PRICING` in `cli.py` to match your actual rates.

A note the dollars make obvious: output is ~0.5% of *tokens* but ~16% of *cost* on Opus, because output is priced ~50× cache-read. Context replay still dominates, but output isn't free.

## Caveats

- Tool-result sizes (for the "dead weight" finding) are estimated from content length (~chars/4); the dead-weight/stale-carry figure is directional, shown as "where to look," not a hard claim.
- "Used until" for stale-carry uses file/command recurrence as a proxy.
- The grade and cost (from token counts and pricing) are exact; the diagnostic offenders are heuristic.

## License

MIT © Bogdan Ionescu
