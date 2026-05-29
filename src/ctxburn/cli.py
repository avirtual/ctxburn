#!/usr/bin/env python3
"""
ctxburn — grade Claude Code sessions on token cost.

Reads JSONL transcripts and grades each session on the mundane, provable axis:
how much context it carried per turn (vs the model window) and how it managed
compaction. Surfaces the sessions that could have been improved; stays quiet
about the efficient ones.

The bill for an agentic coding session is ~96% context replay and <1% generated
output. Replay = (turns) x (context per turn). A long session that never
compacts is a triangle: each late turn re-sends the entire accumulated history.
This tool measures that and tells the developer where it hurt.

Usage:
  ctxburn [PATH] [--since DAYS] [--all] [--session SID] [--min-turns N]
           [--top N] [--window TOKENS] [--json]

  PATH         a project folder, a ~/.claude/projects root, or a single .jsonl
               (default: ~/.claude/projects)
  --since N    only sessions active in the last N days (default: 7)
  --all        ignore the time window
  --session    grade one session id (full report)
  --min-turns  skip sessions shorter than this (default: 20)
  --top N      show full findings for the N most-improvable sessions (default: 6)
  --window     assumed context window in tokens (default: 200000)
  --json       emit machine-readable JSON instead of the report
"""
import argparse, json, glob, os, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

DEFAULT_ROOT = os.path.expanduser("~/.claude/projects")

# Per-model list prices, USD per *million* tokens: (input, output, cache_write_5m, cache_read).
# These are public list prices — enterprise contracts differ. Edit this dict to override.
# Matched by FIRST substring hit against the turn's `model` field, so list most-specific
# keys first (e.g. "opus-4-8" before "opus"); falls back to sonnet for unknown models.
# Source: https://platform.claude.com/docs/en/about-claude/pricing (verified
# against `claude /usage` on real opus-4-6 and opus-4-8 sessions, matched to
# the cent). Keys are matched by FIRST substring hit, so the legacy-priced
# variants are listed BEFORE the generic family key.
PRICING = {
    # Opus 4.1 is the OLD $15/$75 tier (deprecated). Listed first so it wins the
    # substring match before the current "opus" entry. (Base Opus 4 is retired
    # and won't appear in Claude Code transcripts, so it isn't special-cased.)
    "opus-4-1": (15.00, 75.00, 18.75, 1.50),
    # Current Opus (4.5 / 4.6 / 4.7 / 4.8) — $5/$25, 1/3 of the old list price.
    "opus":     (5.00,  25.00,  6.25, 0.50),
    "sonnet":   (3.00,  15.00,  3.75, 0.30),
    # Current Haiku 4.5. (Retired Haiku 3.5 was 0.80/4.00/1.00/0.08.)
    "haiku":    (1.00,   5.00,  1.25, 0.10),
}
PRICING_FALLBACK = "sonnet"
WEB_SEARCH_PER_1K = 10.00   # server-side web search: $10 per 1,000 searches
# Caveat: Opus 4.7+ use a new tokenizer (~35% more tokens for the same text).
# Cost here is exact (it reads real usage counts from the transcript); only the
# char/4 estimate in _tok (dead-weight/offender heuristic) under-counts on 4.7+.
# Also: US-only inference (inference_geo="us") adds a 1.1x multiplier not modeled.

def price_for(model):
    lo = (model or "").lower()
    for key, p in PRICING.items():
        if key in lo:
            return p
    return PRICING[PRICING_FALLBACK]


# ---------- parsing ----------

def _tok(s):
    return max(0, len(s) // 4)  # rough char->token estimate for result bodies

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if not isinstance(b, dict):
                continue
            ty = b.get("type")
            if ty == "text":
                out.append(b.get("text", ""))
            elif ty == "tool_result":
                c = b.get("content")
                out.append(c if isinstance(c, str) else _text_of(c))
            elif ty == "tool_use":
                out.append(json.dumps(b.get("input", {})))
        return "\n".join(out)
    return ""

def _key_of(name, inp):
    if not isinstance(inp, dict):
        return (name, None)
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return ("file", inp.get("file_path") or inp.get("notebook_path"))
    if name == "Bash":
        return ("bash", inp.get("command", "")[:70])
    if name in ("Grep", "Glob"):
        return ("search", str(inp.get("pattern") or inp.get("glob")))
    return (name, json.dumps(inp)[:60])

def _fold_subagent_dir(path, seen_msg, cost):
    """Fold cost from subagent transcripts stored in a `<session>/subagents/`
    subdirectory next to the main transcript (a layout used by some Claude Code
    / Agent-SDK versions). Updates `cost` in place; returns (turns, usd).
    Costs only — subagent context does not enter the main grade curve.
    (The inline-`isSidechain` layout is handled directly in parse_session.)"""
    stem = os.path.basename(path)
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    subdir = os.path.join(os.path.dirname(path), stem, "subagents")
    st, scost = 0, 0.0
    for sf in sorted(glob.glob(os.path.join(subdir, "*.jsonl"))):
        try:
            fh = open(sf, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                m = o.get("message", {})
                if not (isinstance(m, dict) and m.get("role") == "assistant"):
                    continue
                u = m.get("usage") or {}
                cr = u.get("cache_read_input_tokens", 0) or 0
                cc = u.get("cache_creation_input_tokens", 0) or 0
                it = u.get("input_tokens", 0) or 0
                ot = u.get("output_tokens", 0) or 0
                if cr + cc + ot == 0:
                    continue
                mid = m.get("id") or o.get("requestId")
                if mid is not None:
                    if mid in seen_msg:
                        continue
                    seen_msg.add(mid)
                p_in, p_out, p_cw, p_cr = price_for(m.get("model", ""))
                ws = (u.get("server_tool_use") or {}).get("web_search_requests", 0) or 0
                sc = ws * WEB_SEARCH_PER_1K / 1000.0
                cost["read"]   += cr * p_cr / 1e6
                cost["write"]  += cc * p_cw / 1e6
                cost["inp"]    += it * p_in / 1e6
                cost["out"]    += ot * p_out / 1e6
                cost["search"] += sc
                st += 1
                scost += (cr * p_cr + cc * p_cw + it * p_in + ot * p_out) / 1e6 + sc
    return st, scost


def parse_session(path):
    """Return a metrics dict for one .jsonl session, or None if not gradeable."""
    crs = []                       # cache_read per assistant turn (the context curve)
    out_tot = 0
    comps = 0
    multi = 0
    toolcalls = 0
    pending = {}                   # tool_use_id -> (turn, name, key)
    results = []                   # (turn, name, key, result_tokens)
    key_turns = defaultdict(list)
    last_ts = None
    cost = dict(read=0.0, write=0.0, inp=0.0, out=0.0, search=0.0)  # USD by component
    turn_cost = []                 # USD incurred per assistant turn (for ramp milestones)
    models = Counter()
    cwds = Counter()
    sub_turns = 0                  # subagent/sidechain turns (cost folded in, grade excluded)
    sub_cost = 0.0
    boot_write_tokens = 0          # cache-write tokens on the cold-boot (first main) turn
    boot_input_tokens = 0          # uncached input tokens on that same turn
    boot_floor_cost = 0.0          # USD a restart re-pays (write-rate, NOT read-rate)
    seen_msg = set()               # dedup: each API response is logged multiple times
                                   # (streaming partials + final) under one message.id;
                                   # counting every record double-counts tokens & cost.

    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return None
    with fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            ts = o.get("timestamp")
            if ts:
                last_ts = ts
            cwd = o.get("cwd")
            if cwd:
                cwds[cwd] += 1
            if o.get("isCompactSummary") or o.get("compactMetadata"):
                comps += 1
            t = o.get("type")
            m = o.get("message", {})
            if isinstance(m, dict) and m.get("role") == "assistant":
                u = m.get("usage") or {}
                cr = u.get("cache_read_input_tokens", 0) or 0
                cc = u.get("cache_creation_input_tokens", 0) or 0
                it = u.get("input_tokens", 0) or 0
                ot = u.get("output_tokens", 0) or 0
                if cr + cc + ot == 0:
                    continue
                # one API response, logged multiple times -> count its usage once
                mid = m.get("id") or o.get("requestId")
                if mid is not None:
                    if mid in seen_msg:
                        continue
                    seen_msg.add(mid)
                mdl = m.get("model", "")
                p_in, p_out, p_cw, p_cr = price_for(mdl)
                ws = (u.get("server_tool_use") or {}).get("web_search_requests", 0) or 0
                sc = ws * WEB_SEARCH_PER_1K / 1000.0
                tc = (cr * p_cr + cc * p_cw + it * p_in + ot * p_out) / 1e6 + sc
                # cost is whole-work-session (main loop + Task-tool subagents)
                cost["read"]   += cr * p_cr / 1e6
                cost["write"]  += cc * p_cw / 1e6
                cost["inp"]    += it * p_in / 1e6
                cost["out"]    += ot * p_out / 1e6
                cost["search"] += sc

                # A subagent/sidechain turn carries its own context, not the main
                # loop's — fold its COST in, but keep it out of the context curve so
                # the grade reflects the main session's discipline.
                if bool(o.get("isSidechain")) or t != "assistant":
                    sub_turns += 1
                    sub_cost += tc
                    continue

                crs.append(cr)
                if len(crs) == 1:
                    # Cold-boot turn: the context is cache-WRITTEN here (cc), not read
                    # (cr0 is ~0 — nothing prior to read). This one-time write is the
                    # floor a restart re-pays, priced ~12.5x the per-turn read replay.
                    boot_write_tokens = cc
                    boot_input_tokens = it
                    boot_floor_cost = (cc * p_cw + it * p_in) / 1e6
                out_tot += ot
                models[mdl] += 1
                turn_cost.append(tc)
                ntool = 0
                for b in (m.get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        ntool += 1
                        toolcalls += 1
                        k = _key_of(b.get("name"), b.get("input"))
                        pending[b.get("id")] = (len(crs), b.get("name"), k)
                        key_turns[k].append(len(crs))
                if ntool >= 2:
                    multi += 1
            elif t == "user" and isinstance(m, dict):
                for b in (m.get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tid = b.get("tool_use_id")
                        if tid in pending:
                            at, name, k = pending.pop(tid)
                            results.append((at, name, k, _tok(_text_of(b.get("content")))))

    # fold in any subagent transcripts stored in a sibling <session>/subagents/ dir
    fst, fsc = _fold_subagent_dir(path, seen_msg, cost)
    sub_turns += fst
    sub_cost += fsc

    n = len(crs)
    if n == 0:
        return None

    area = sum(crs)
    peak = max(crs)
    sess_avg = area / n
    last10 = sum(crs[-10:]) / min(10, n)

    # where-to-cut: first turn (1-based) at which the context curve crossed each
    # grade band — turns a tail-only grade into a concrete "compact here" turn.
    crossings = []
    for band in GRADE_BANDS:
        for i, c in enumerate(crs):
            if c >= band:
                crossings.append((band, i + 1))
                break

    # cumulative cost through the first N turns — shows how fast the bill ramps.
    # For sessions shorter than N, this equals the full cost (whole session ran).
    cum = []
    run = 0.0
    for tc in turn_cost:
        run += tc
        cum.append(run)
    def cost_through(k):
        return cum[min(k, len(cum)) - 1] if cum else 0.0
    c25, c50, c100 = cost_through(25), cost_through(50), cost_through(100)

    # dead-weight carry: result size * turns replayed after its key was last touched
    stale = 0
    carry = defaultdict(lambda: [0, 0, 0])   # (name,key) -> [stale, total_carry, fetches]
    for at, name, k, rt in results:
        if rt == 0:
            continue
        last_use = max(key_turns.get(k, [at]))
        stale += rt * max(0, n - last_use)
        carry[(name, k)][0] += rt * max(0, n - last_use)
        carry[(name, k)][1] += rt * (n - at)
        carry[(name, k)][2] += 1
    offenders = sorted(carry.items(), key=lambda x: -x[1][0])[:3]

    fetches = defaultdict(int)
    for at, name, k, rt in results:
        fetches[(name, k)] += 1
    top_reread = max(((k, c) for k, c in fetches.items() if c > 1),
                     key=lambda x: x[1], default=None)

    return dict(
        path=path, sid=os.path.basename(path)[:8],
        project=(cwds.most_common(1)[0][0] if cwds
                 else os.path.basename(os.path.dirname(path))),
        last_ts=last_ts, turns=n, compactions=comps,
        area=area, peak=peak, sess_avg=sess_avg, last10=last10,
        output=out_tot, toolcalls=toolcalls, multi=multi,
        stale=stale, offenders=offenders, top_reread=top_reread,
        cost=cost, cost_total=sum(cost.values()),
        sub_turns=sub_turns, sub_cost=sub_cost,
        cost25=c25, cost50=c50, cost100=c100,
        boot_write_tokens=boot_write_tokens, boot_input_tokens=boot_input_tokens,
        boot_floor_cost=boot_floor_cost, crossings=crossings,
        model=(models.most_common(1)[0][0] if models else "?"),
    )


# ---------- grading ----------

GRADES = ["A", "B", "C", "D", "F"]

# absolute context-per-turn bands (tokens re-sent every turn = cost per turn).
# Window-independent on purpose: cost is flat per token, so it does not matter
# whether 488K is 49% of a 1M window or 244% of a 200K one — the bill is the same.
GRADE_BANDS = [40_000, 80_000, 130_000, 180_000]  # A|B|C|D|F cutoffs
HEALTHY_CTX = 60_000  # a lean working context for a coding session
TABLE_CAP = 20        # max rows in the overview table (keeps output friendly)

def grade_session(r, window=None):
    """Grade on absolute last-10-turn avg context (= cost per turn).
    The window does NOT affect cost and does not drive the grade."""
    c = r["last10"]
    base = sum(1 for b in GRADE_BANDS if c >= b)
    # long marathon with no compaction: one step worse
    if r["turns"] >= 150 and r["compactions"] == 0:
        base = min(4, base + 1)
    # compacted and ended well below peak: one step better (rewarded the discipline)
    if r["compactions"] >= 1 and r["last10"] < 0.6 * r["peak"]:
        base = max(0, base - 1)
    return GRADES[base], c

def improvable_score(r, window=None):
    """Rank by recoverable spend: total replay scaled by how bloated the tail was."""
    return r["area"] * max(0.0, (r["last10"] - HEALTHY_CTX) / max(1, r["last10"]))


# ---------- reporting ----------

def fmt_k(x):
    return f"{x/1e3:.0f}K"

def short_model(m):
    """claude-opus-4-6 -> opus-4-6; keep unknowns as-is, cap length."""
    if not m or m == "?":
        return "?"
    s = m
    for pre in ("claude-", "anthropic/", "anthropic."):
        if s.startswith(pre):
            s = s[len(pre):]
    return s[:14]

def pretty_path(p, width=52):
    home = os.path.expanduser("~")
    if p and p.startswith(home):
        p = "~" + p[len(home):]
    if p and len(p) > width:
        p = "\u2026" + p[-(width - 1):]
    return p


# ---------- table rendering ----------

def _use_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

GRADE_COLOR = {"A": "32", "B": "36", "C": "33", "D": "33;1", "F": "31"}

def color(s, code):
    if not code or not _use_color():
        return s
    return f"\033[{code}m{s}\033[0m"

def bold(s):
    return color(s, "1")

def render_table(headers, rows, aligns, colorizers=None):
    """Aligned text table. aligns: per-col 'l'/'r'. colorizers: per-col
    callable(raw_value, padded_str)->str or None (color applied after padding
    so width alignment stays correct)."""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))

    def pad(i, c):
        c = str(c)
        return c.rjust(widths[i]) if aligns[i] == "r" else c.ljust(widths[i])

    def fmt(row, header=False):
        cells = []
        for i, c in enumerate(row):
            p = pad(i, c)
            if header:
                p = bold(p)
            elif colorizers and i < len(colorizers) and colorizers[i]:
                p = colorizers[i](str(c), p)
            cells.append(p)
        return "  ".join(cells).rstrip()

    rule = "  ".join("\u2500" * w for w in widths)
    out = [fmt(headers, header=True), rule]
    out += [fmt(r) for r in rows]
    return out

COST_COMPONENTS = [
    ("read",   "cache read (context replay)"),
    ("write",  "cache write"),
    ("out",    "output (generated)"),
    ("inp",    "input (uncached)"),
    ("search", "web search"),
]

def cost_breakdown_lines(r, indent="       "):
    """Per-component cost table: $ and % of session total, largest first."""
    c, tot = r["cost"], (r["cost_total"] or 1e-9)
    rows = [(label, c.get(key, 0.0)) for key, label in COST_COMPONENTS]
    rows = [(lbl, v) for lbl, v in rows if v > 0]
    rows.sort(key=lambda x: -x[1])
    headers = ["COMPONENT", "COST", "SHARE"]
    table = [[lbl, f"${v:,.2f}", f"{100*v/tot:.0f}%"] for lbl, v in rows]
    table.append(["total", f"${r['cost_total']:,.2f}", "100%"])
    out = render_table(headers, table, ["l", "r", "r"])
    return [indent + ln for ln in out]

def findings_and_suggestions(r, window):
    F, S = [], []
    out_pct = 100 * r["output"] / r["area"] if r["area"] else 0
    F.append(f"{r['turns']} turns, {r['compactions']} compaction(s). "
             f"{r['area']/1e6:.1f}M tokens of context replay for {fmt_k(r['output'])} "
             f"of generated output ({out_pct:.1f}%).")
    c, ct = r["cost"], r["cost_total"] or 1e-9
    ctx_cost = c["read"] + c["write"]
    if r.get("sub_turns"):
        F.append(f"Work-session cost includes {r['sub_turns']} subagent turn(s) "
                 f"(${r['sub_cost']:.2f}) folded in alongside the main loop.")
    F.append(f"Cost: ${r['cost_total']:.2f} on {r['model']} — "
             f"context replay+write ${ctx_cost:.2f} ({100*ctx_cost/ct:.0f}%), "
             f"output ${c['out']:.2f} ({100*c['out']/ct:.0f}%). "
             f"(Output is {out_pct:.1f}% of tokens but {100*c['out']/ct:.0f}% of cost — "
             f"it's priced ~50x context.)")
    F.append(f"Last-10-turn avg context: {fmt_k(r['last10'])} re-sent every turn "
             f"(this is the cost driver). Peak {fmt_k(r['peak'])}, "
             f"session-avg {fmt_k(r['sess_avg'])}.")
    if r["last10"] >= 0.85 * window:
        F.append(f"Tail context is at/over the {window//1000}K window — "
                 f"you were riding forced auto-compaction (a risk flag, separate from cost).")
    if r["turns"] >= 150 and r["compactions"] == 0:
        F.append(f"Ran {r['turns']} turns without a single compaction — "
                 f"every late turn re-sent the entire accumulated history.")
        S.append("Compact or /clear *early and repeatedly*. Cost per turn is the "
                 "context you carry, not the turn number — once the tail is heavy "
                 "every remaining turn pays for it. A reset caps context height for "
                 "every turn after it, so the earlier you cut the more turns benefit.")
    if r["offenders"]:
        (name, k), (st, ca, cnt) = r["offenders"][0]
        kd = str(k[1])[:50] if k and k[1] else "?"
        F.append(f"Largest dead weight: {name} `{kd}` — {ca/1e6:.1f}M replay tokens "
                 f"(fetched {cnt}x, carried long after last use).")
        S.append("Drop large one-shot outputs (CLI dumps, full-file reads) once "
                 "consumed instead of carrying them to the end.")
    if r["top_reread"] and r["top_reread"][1] >= 3:
        (name, k), c = r["top_reread"]
        kd = str(k[1])[:42] if k and k[1] else "?"
        F.append(f"Re-reads: `{kd}` fetched {c}x — full content re-injected each time.")
        S.append("Pin or summarize frequently-needed files once instead of re-reading.")
    if r["toolcalls"] and r["multi"] / r["toolcalls"] < 0.1:
        F.append(f"Parallel tool calls: {r['multi']}/{r['toolcalls']} "
                 f"({100*r['multi']/r['toolcalls']:.0f}%) — most calls took a separate round trip.")
        S.append("Batch independent reads/greps into one turn; each saved round trip "
                 "removes a full context re-read.")
    if r.get("crossings"):
        parts = ", ".join(f"{fmt_k(b)} at turn {t}" for b, t in r["crossings"])
        # anchor the cut advice at the 130K ("getting heavy") crossing if it happened,
        # else the heaviest band actually crossed.
        cut = next((t for b, t in r["crossings"] if b == 130_000),
                   r["crossings"][-1][1])
        F.append(f"Where to cut: context crossed {parts}.")
        S.append(f"A /clear or /compact near turn {cut} would have capped context for "
                 f"every later turn — that crossing is where the costly tail begins.")
    return F, S

def report(sessions, window, top, as_json):
    graded = []
    for r in sessions:
        g, frac = grade_session(r, window)
        r["grade"] = g
        r["improvable"] = improvable_score(r, window)
        graded.append(r)
    graded.sort(key=lambda r: -r["improvable"])

    if as_json:
        out = [{k: r[k] for k in ("sid", "project", "model", "turns", "compactions",
                                  "area", "peak", "sess_avg", "last10", "output",
                                  "cost25", "cost50", "cost100", "boot_floor_cost",
                                  "boot_write_tokens", "crossings",
                                  "cost_total", "grade", "improvable")} for r in graded]
        print(json.dumps(out, indent=2))
        return

    n = len(graded)
    total_replay = sum(r["area"] for r in graded)
    total_cost = sum(r["cost_total"] for r in graded)
    dist = defaultdict(int)
    for r in graded:
        dist[r["grade"]] += 1

    # ---- banner ----
    title = (f"ctxburn — {n} session{'s' if n != 1 else ''} · "
             f"{total_replay/1e6:.0f}M replay tokens · ${total_cost:,.0f} total")
    print()
    print(bold(title))
    grade_chips = "   ".join(color(f"{g} {dist[g]}", GRADE_COLOR[g])
                             for g in GRADES if dist[g])
    print("grades:  " + grade_chips)
    print(color("grade = last-10-turn avg context carried per turn (= $/turn). "
                 f"window {window//1000}K is a ceiling-risk flag only.", "2"))

    # ---- overview table, worst-improvable first, bounded for friendliness ----
    # Lead with the sessions worth improving (C/D/F); fall back to the full
    # roster only when everything is already efficient.
    improvable = [r for r in graded if r["grade"] in ("C", "D", "F")]
    efficient = [r for r in graded if r["grade"] in ("A", "B")]
    table_src = improvable if improvable else graded
    table_rows = table_src[:TABLE_CAP]

    grade_col = lambda v, p: color(p, GRADE_COLOR.get(v.strip(), ""))
    money = lambda v: f"${v:,.0f}"
    headers = ["", "SESSION", "PROJECT", "MODEL", "TURNS", "COMP", "LAST-10",
               "@25", "@50", "@100", "TOTAL"]
    aligns  = ["l", "l", "l", "l", "r", "r", "r", "r", "r", "r", "r"]
    rows = [[r["grade"], r["sid"], pretty_path(r["project"], 34),
             short_model(r["model"]), r["turns"], r["compactions"],
             fmt_k(r["last10"]), money(r["cost25"]), money(r["cost50"]),
             money(r["cost100"]), money(r["cost_total"])] for r in table_rows]
    print()
    for line in render_table(headers, rows, aligns, colorizers=[grade_col]):
        print("  " + line)

    overflow = len(table_src) - len(table_rows)
    tail = []
    if overflow > 0:
        tail.append(f"+{overflow} more improvable session(s) not shown")
    if improvable and efficient:
        tail.append(f"{len(efficient)} efficient (A/B) hidden")
    if tail:
        print(color("  … " + " · ".join(tail) + ".", "2"))

    # ---- detailed findings for the most improvable C/D/F sessions ----
    detail = improvable[:top]
    if detail:
        print()
        print(bold(f"  WHERE IT HURT — top {len(detail)} improvable session"
                   f"{'s' if len(detail) != 1 else ''}"))
    for r in detail:
        F, S = findings_and_suggestions(r, window)
        print(f"\n  {'─'*72}")
        print(f"  {color('['+r['grade']+']', GRADE_COLOR.get(r['grade'], ''))}  "
              f"{bold(r['sid'])}  {pretty_path(r['project'])}  "
              f"${r['cost_total']:,.0f}")
        print(f"       last-10 avg {fmt_k(r['last10'])} · {r['turns']} turns · "
              f"{r['compactions']} compactions · {r['area']/1e6:.1f}M replay")
        print(f"     {bold('COST BREAKDOWN')}")
        for ln in cost_breakdown_lines(r):
            print(ln)
        print(f"     {bold('FINDINGS')}")
        for x in F:
            print(f"       - {x}")
        print(f"     {bold('SUGGESTIONS')}")
        for x in S:
            print(f"       > {x}")

    print()


def report_by_project(sessions, window, as_json):
    """Roll grades up by project (cwd). Surfaces the cross-session cost driver a
    per-session view hides: restart-heavy agents re-pay their boot context every
    cold start. `boots` = session count (each JSONL is one cold context load);
    `boot-floor` = the one-time cache-WRITE re-paid on each of those boots."""
    for r in sessions:
        r["grade"], _ = grade_session(r, window)

    groups = defaultdict(list)
    for r in sessions:
        groups[r["project"]].append(r)

    rows = []
    for proj, rs in groups.items():
        grades = [r["grade"] for r in rs]
        worst = max(grades, key=GRADES.index)
        avg = GRADES[round(sum(GRADES.index(g) for g in grades) / len(grades))]
        rows.append(dict(
            project=proj, sessions=len(rs),
            cost=sum(r["cost_total"] for r in rs),
            replay=sum(r["area"] for r in rs),
            boot_floor=sum(r.get("boot_floor_cost", 0.0) for r in rs),
            worst=worst, avg=avg,
        ))
    rows.sort(key=lambda x: -x["cost"])

    if as_json:
        print(json.dumps(rows, indent=2))
        return

    n_proj = len(rows)
    tot_cost = sum(x["cost"] for x in rows)
    tot_boot = sum(x["boot_floor"] for x in rows)
    tot_sess = sum(x["sessions"] for x in rows)
    print()
    print(bold(f"ctxburn by project — {n_proj} project{'s' if n_proj != 1 else ''} · "
               f"{tot_sess} sessions · ${tot_cost:,.0f} total · "
               f"${tot_boot:,.0f} boot-floor"))
    print(color("boots = sessions (each cold-loads context). boot-floor = the one-time "
                "cache-WRITE a restart re-pays, summed across boots.", "2"))

    grade_col = lambda v, p: color(p, GRADE_COLOR.get(v.strip(), ""))
    money = lambda v: f"${v:,.0f}"
    headers = ["PROJECT", "BOOTS", "WORST", "AVG", "REPLAY", "BOOT-FLOOR", "TOTAL"]
    aligns  = ["l", "r", "l", "l", "r", "r", "r"]
    table = [[pretty_path(x["project"], 40), x["sessions"], x["worst"], x["avg"],
              f"{x['replay']/1e6:.0f}M", money(x["boot_floor"]), money(x["cost"])]
             for x in rows[:TABLE_CAP]]
    colorizers = [None, None, grade_col, None, None, None, None]
    print()
    for line in render_table(headers, table, aligns, colorizers=colorizers):
        print("  " + line)
    overflow = len(rows) - len(table)
    if overflow > 0:
        print(color(f"  … +{overflow} more project(s) not shown.", "2"))
    print()


# ---------- cli ----------

def collect_files(path, session):
    if session:
        hits = glob.glob(os.path.join(path, "**", f"{session}*.jsonl"), recursive=True)
        if not hits and os.path.isdir(path):
            hits = glob.glob(os.path.join(path, f"{session}*.jsonl"))
        return hits
    if os.path.isfile(path) and path.endswith(".jsonl"):
        return [path]
    return glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)

def within_days(last_ts, days):
    if not last_ts:
        return False
    try:
        dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    except Exception:
        return True  # undated -> don't filter out
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)

def main(argv=None):
    ap = argparse.ArgumentParser(description="Grade Claude Code sessions on context efficiency.")
    ap.add_argument("path", nargs="?", default=DEFAULT_ROOT)
    ap.add_argument("--since", type=int, default=7, help="only sessions active in the last N days")
    ap.add_argument("--all", action="store_true", help="ignore the time window")
    ap.add_argument("--session", help="grade one session id")
    ap.add_argument("--min-turns", type=int, default=20)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--window", type=int, default=200_000)
    ap.add_argument("--by-project", action="store_true",
                    help="roll cost/grades up by project (cwd); show restart boot-floor")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    files = collect_files(a.path, a.session)
    if not files:
        print(f"no .jsonl sessions found under {a.path}", file=sys.stderr)
        return 1

    # dedupe by session id (filename stem): the same session can appear under
    # multiple paths (backups, copies). Keep the first occurrence.
    seen, uniq = set(), []
    for f in files:
        b = os.path.basename(f)
        if b in seen:
            continue
        seen.add(b)
        uniq.append(f)
    files = uniq

    sessions = []
    for f in files:
        r = parse_session(f)
        if not r:
            continue
        if a.session:
            sessions.append(r)
            continue
        if r["turns"] < a.min_turns:
            continue
        if not a.all and not within_days(r["last_ts"], a.since):
            continue
        sessions.append(r)

    if not sessions:
        scope = "any time" if a.all else f"the last {a.since} days"
        print(f"no gradeable sessions (>= {a.min_turns} turns) in {scope}. "
              f"Try --all or --since N.", file=sys.stderr)
        return 0

    # project rollup: cross-session view of the restart/boot-floor cost driver
    if a.by_project:
        report_by_project(sessions, a.window, a.json)
        return 0

    # single-session mode: always show full detail regardless of grade
    if a.session:
        for r in sessions:
            r["grade"], _ = grade_session(r, a.window)
            r["improvable"] = improvable_score(r, a.window)
            F, S = findings_and_suggestions(r, a.window)
            print(f"\n  [{r['grade']}]  {r['sid']}  ({pretty_path(r['project'])})")
            print(f"     {bold('COST BREAKDOWN')}")
            for ln in cost_breakdown_lines(r):
                print(ln)
            print(f"     {bold('FINDINGS')}")
            for x in F: print(f"       - {x}")
            print(f"     {bold('SUGGESTIONS')}")
            for x in S: print(f"       > {x}")
        print()
        return 0

    report(sessions, a.window, a.top, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
