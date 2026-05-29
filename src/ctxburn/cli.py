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
# These are public list prices — enterprise contracts differ. Override with --price-* or edit here.
# Matched by substring against the turn's `model` field; falls back to sonnet for unknown models.
PRICING = {
    "opus":   (15.00, 75.00, 18.75, 1.50),
    "sonnet": (3.00,  15.00,  3.75, 0.30),
    "haiku":  (0.80,   4.00,  1.00, 0.08),
}
PRICING_FALLBACK = "sonnet"

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
    cost = dict(read=0.0, write=0.0, inp=0.0, out=0.0)  # USD by tier
    models = Counter()
    cwds = Counter()

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
            if t == "assistant" and isinstance(m, dict) and m.get("role") == "assistant":
                u = m.get("usage") or {}
                cr = u.get("cache_read_input_tokens", 0) or 0
                cc = u.get("cache_creation_input_tokens", 0) or 0
                it = u.get("input_tokens", 0) or 0
                ot = u.get("output_tokens", 0) or 0
                if cr + cc + ot == 0:
                    continue
                crs.append(cr)
                out_tot += ot
                mdl = m.get("model", "")
                models[mdl] += 1
                p_in, p_out, p_cw, p_cr = price_for(mdl)
                cost["read"]  += cr * p_cr / 1e6
                cost["write"] += cc * p_cw / 1e6
                cost["inp"]   += it * p_in / 1e6
                cost["out"]   += ot * p_out / 1e6
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

    n = len(crs)
    if n == 0:
        return None

    area = sum(crs)
    peak = max(crs)
    sess_avg = area / n
    last10 = sum(crs[-10:]) / min(10, n)

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
        model=(models.most_common(1)[0][0] if models else "?"),
    )


# ---------- grading ----------

GRADES = ["A", "B", "C", "D", "F"]

# absolute context-per-turn bands (tokens re-sent every turn = cost per turn).
# Window-independent on purpose: cost is flat per token, so it does not matter
# whether 488K is 49% of a 1M window or 244% of a 200K one — the bill is the same.
GRADE_BANDS = [40_000, 80_000, 130_000, 180_000]  # A|B|C|D|F cutoffs
HEALTHY_CTX = 60_000  # a lean working context for a coding session

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

def pretty_path(p, width=52):
    home = os.path.expanduser("~")
    if p and p.startswith(home):
        p = "~" + p[len(home):]
    if p and len(p) > width:
        p = "\u2026" + p[-(width - 1):]
    return p

def findings_and_suggestions(r, window):
    F, S = [], []
    out_pct = 100 * r["output"] / r["area"] if r["area"] else 0
    F.append(f"{r['turns']} turns, {r['compactions']} compaction(s). "
             f"{r['area']/1e6:.1f}M tokens of context replay for {fmt_k(r['output'])} "
             f"of generated output ({out_pct:.1f}%).")
    c, ct = r["cost"], r["cost_total"] or 1e-9
    ctx_cost = c["read"] + c["write"]
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
        S.append("Compact or /clear *early and repeatedly*. The first ~100 turns are "
                 "nearly free; cost is in the tail you let grow. A reset caps the "
                 "context height for every turn after it.")
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
                                  "cost_total", "grade", "improvable")} for r in graded]
        print(json.dumps(out, indent=2))
        return

    n = len(graded)
    total_replay = sum(r["area"] for r in graded)
    total_cost = sum(r["cost_total"] for r in graded)
    dist = defaultdict(int)
    for r in graded:
        dist[r["grade"]] += 1
    print(f"\n{'='*72}")
    print(f"SESSION GRADER — {n} sessions · {total_replay/1e6:.0f}M replay tokens · "
          f"${total_cost:,.0f} total")
    print(f"grades: " + "  ".join(f"{g}:{dist[g]}" for g in GRADES if dist[g]))
    print(f"grade = last-10-turn avg context (absolute tokens/turn = cost). "
          f"window {window//1000}K used only for the ceiling-risk flag.")
    print(f"{'='*72}")

    detail = [r for r in graded if r["grade"] in ("C", "D", "F")][:top]
    for r in detail:
        F, S = findings_and_suggestions(r, window)
        print(f"\n  {'─'*72}")
        print(f"  [{r['grade']}]  {r['sid']}  ({pretty_path(r['project'])})  ${r['cost_total']:,.0f}")
        print(f"       last-10 avg {fmt_k(r['last10'])} · {r['turns']} turns · "
              f"{r['compactions']} compactions · {r['area']/1e6:.1f}M replay")
        print(f"     FINDINGS:")
        for x in F:
            print(f"       - {x}")
        print(f"     SUGGESTIONS:")
        for x in S:
            print(f"       > {x}")

    shown = {r["sid"] for r in detail}
    rest = [r for r in graded if r["sid"] not in shown]
    if rest:
        good = [r for r in rest if r["grade"] in ("A", "B")]
        mid = [r for r in rest if r["grade"] not in ("A", "B")]
        print(f"\n  {'-'*68}")
        if mid:
            print(f"  {len(mid)} more improvable session(s) not detailed (raise --top to see):")
            for r in mid:
                print(f"     [{r['grade']}] {r['sid']} {fmt_k(r['last10'])} last-10 · {r['turns']}t")
        if good:
            print(f"  {len(good)} efficient session(s) (A/B) — skipped: "
                  + ", ".join(r["sid"] for r in good[:12])
                  + (" ..." if len(good) > 12 else ""))
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

    # single-session mode: always show full detail regardless of grade
    if a.session:
        for r in sessions:
            r["grade"], _ = grade_session(r, a.window)
            r["improvable"] = improvable_score(r, a.window)
            F, S = findings_and_suggestions(r, a.window)
            print(f"\n  [{r['grade']}]  {r['sid']}  ({pretty_path(r['project'])})")
            print(f"     FINDINGS:")
            for x in F: print(f"       - {x}")
            print(f"     SUGGESTIONS:")
            for x in S: print(f"       > {x}")
        print()
        return 0

    report(sessions, a.window, a.top, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
