"""Smoke tests: build a tiny synthetic transcript and check the core paths."""
import json, os, tempfile
from ctxburn import cli


def _write_session(tmpdir):
    lines = [
        {"type": "user", "message": {"role": "user", "content": "fix the thing"},
         "timestamp": "2099-01-01T00:00:00Z"},
        {"type": "assistant", "message": {
            "role": "assistant", "model": "claude-opus-4-7",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                         "input": {"file_path": "/a/b.tf"}}],
            "usage": {"cache_read_input_tokens": 50000, "output_tokens": 200,
                      "cache_creation_input_tokens": 1000, "input_tokens": 0}},
         "timestamp": "2099-01-01T00:01:00Z"},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 4000}]}},
        {"type": "assistant", "message": {
            "role": "assistant", "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"cache_read_input_tokens": 60000, "output_tokens": 300,
                      "cache_creation_input_tokens": 0, "input_tokens": 0}},
         "timestamp": "2099-01-01T00:02:00Z"},
    ]
    p = os.path.join(tmpdir, "deadbeef-0000-0000-0000-000000000000.jsonl")
    with open(p, "w") as fh:
        for o in lines:
            fh.write(json.dumps(o) + "\n")
    return p


def test_parse_and_grade():
    with tempfile.TemporaryDirectory() as d:
        r = cli.parse_session(_write_session(d))
        assert r is not None
        assert r["turns"] == 2
        assert r["model"] == "claude-opus-4-7"
        assert r["cost_total"] > 0
        # cache_read cost = (50000+60000) * opus cache-read rate / 1e6
        p_cr = cli.PRICING["opus"][3]
        assert abs(r["cost"]["read"] - (110000 * p_cr / 1e6)) < 1e-9
        g, c = cli.grade_session(r)
        assert g in cli.GRADES


def test_pricing_lookup():
    assert cli.price_for("claude-opus-4-7") == cli.PRICING["opus"]
    assert cli.price_for("claude-sonnet-4-5") == cli.PRICING["sonnet"]
    assert cli.price_for("claude-3-5-haiku") == cli.PRICING["haiku"]
    # unknown -> fallback
    assert cli.price_for("gpt-something") == cli.PRICING[cli.PRICING_FALLBACK]


def test_dedup_by_message_id():
    """The same API response is logged multiple times under one message.id;
    its usage must be counted exactly once (else tokens/cost double)."""
    with tempfile.TemporaryDirectory() as d:
        turn = {"type": "assistant", "message": {
            "role": "assistant", "model": "claude-opus-4-8", "id": "msg_dupe",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"cache_read_input_tokens": 40000, "output_tokens": 100,
                      "cache_creation_input_tokens": 0, "input_tokens": 0}},
            "timestamp": "2099-01-01T00:01:00Z"}
        p = os.path.join(d, "dup00000-0000-0000-0000-000000000000.jsonl")
        with open(p, "w") as fh:
            # same message.id written three times (streaming partials + final)
            for _ in range(3):
                fh.write(json.dumps(turn) + "\n")
        r = cli.parse_session(p)
        assert r["turns"] == 1                       # not 3
        assert r["area"] == 40000                     # cache_read counted once
        assert r["output"] == 100


def test_subagent_cost_folded_grade_excluded():
    """A Task-tool subagent turn (sidechain, own model) adds to work-session
    cost but must NOT enter the main context curve / turn count / grade."""
    with tempfile.TemporaryDirectory() as d:
        main = {"type": "assistant", "message": {
            "role": "assistant", "model": "claude-opus-4-8", "id": "m_main",
            "content": [{"type": "text", "text": "main"}],
            "usage": {"cache_read_input_tokens": 50000, "output_tokens": 200,
                      "cache_creation_input_tokens": 0, "input_tokens": 0}},
            "timestamp": "2099-01-01T00:01:00Z"}
        sub = {"type": "assistant", "isSidechain": True, "message": {
            "role": "assistant", "model": "claude-haiku-4-5", "id": "m_sub",
            "content": [{"type": "text", "text": "sub"}],
            "usage": {"cache_read_input_tokens": 80000, "output_tokens": 500,
                      "cache_creation_input_tokens": 0, "input_tokens": 0}},
            "timestamp": "2099-01-01T00:01:30Z"}
        p = os.path.join(d, "sub00000-0000-0000-0000-000000000000.jsonl")
        with open(p, "w") as fh:
            fh.write(json.dumps(main) + "\n")
            fh.write(json.dumps(sub) + "\n")
        r = cli.parse_session(p)
        assert r["turns"] == 1            # main only — subagent excluded from curve
        assert r["area"] == 50000         # context curve is the main loop's
        assert r["sub_turns"] == 1
        # subagent cost priced at HAIKU rate, folded into the total
        hp = cli.PRICING["haiku"]
        exp_sub = (80000 * hp[3] + 500 * hp[1]) / 1e6
        assert abs(r["sub_cost"] - exp_sub) < 1e-9
        assert r["cost_total"] > r["cost_total"] - r["sub_cost"] > 0


def test_window_independent_grade():
    """Same session must get the same grade regardless of --window (cost != capacity)."""
    with tempfile.TemporaryDirectory() as d:
        r = cli.parse_session(_write_session(d))
        g200, _ = cli.grade_session(r, window=200_000)
        g1m, _ = cli.grade_session(r, window=1_000_000)
        assert g200 == g1m


# ---- helpers for the new-feature tests ----

def _asst(model, cr, cc=0, it=0, ot=100, mid=None, sidechain=False, cwd=None,
          ts="2099-01-01T00:01:00Z"):
    o = {"type": "assistant", "message": {
        "role": "assistant", "model": model,
        "content": [{"type": "text", "text": "x"}],
        "usage": {"cache_read_input_tokens": cr, "cache_creation_input_tokens": cc,
                  "input_tokens": it, "output_tokens": ot}},
        "timestamp": ts}
    if mid is not None:
        o["message"]["id"] = mid
    if sidechain:
        o["isSidechain"] = True
    if cwd is not None:
        o["cwd"] = cwd
    return o


def _write(d, name, records):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        for o in records:
            fh.write(json.dumps(o) + "\n")
    return p


def test_boot_floor_priced_at_write_rate_not_read():
    """The cold-boot floor is the turn-0 cache-WRITE (cc), NOT the cache-read.
    Pricing crs[0] at the read rate would be ~12.5x too low and break the
    'matches /usage to the cent' property. cr0 here is deliberately large so a
    read-rate regression would be caught."""
    with tempfile.TemporaryDirectory() as d:
        recs = [
            _asst("claude-opus-4-7", cr=99999, cc=10000, it=500, mid="b0"),
            _asst("claude-opus-4-7", cr=60000, cc=0, it=0, mid="b1"),
        ]
        r = cli.parse_session(_write(d, "boot0000-0000-0000-0000-000000000000.jsonl", recs))
        p_in, p_out, p_cw, p_cr = cli.PRICING["opus"]
        expected = (10000 * p_cw + 500 * p_in) / 1e6
        assert abs(r["boot_floor_cost"] - expected) < 1e-12
        assert r["boot_write_tokens"] == 10000
        # the read-rate mistake would yield this very different number
        wrong = (99999 * p_cr) / 1e6
        assert abs(r["boot_floor_cost"] - wrong) > 1e-6


def test_crossings_detect_first_band_turn():
    """`crossings` records the first turn (1-based) the context curve crossed
    each grade band, in band order."""
    with tempfile.TemporaryDirectory() as d:
        recs = [
            _asst("claude-opus-4-7", cr=30000, mid="c1"),   # turn1: under 40K
            _asst("claude-opus-4-7", cr=50000, mid="c2"),   # turn2: crosses 40K
            _asst("claude-opus-4-7", cr=90000, mid="c3"),   # turn3: crosses 80K
            _asst("claude-opus-4-7", cr=140000, mid="c4"),  # turn4: crosses 130K
            _asst("claude-opus-4-7", cr=190000, mid="c5"),  # turn5: crosses 180K
        ]
        r = cli.parse_session(_write(d, "cross000-0000-0000-0000-000000000000.jsonl", recs))
        assert r["crossings"] == [(40000, 2), (80000, 3), (130000, 4), (180000, 5)]


def test_crossings_only_bands_actually_crossed():
    """A lean session that never leaves the A band records no crossings."""
    with tempfile.TemporaryDirectory() as d:
        recs = [_asst("claude-opus-4-7", cr=20000, mid=f"l{i}") for i in range(3)]
        r = cli.parse_session(_write(d, "lean0000-0000-0000-0000-000000000000.jsonl", recs))
        assert r["crossings"] == []


def test_by_project_grouping_and_sort():
    """report_by_project groups sessions by cwd, sums cost + boot-floor per
    project, and sorts projects by total cost descending."""
    import io, contextlib
    with tempfile.TemporaryDirectory() as d:
        # project A: two sessions; project B: one cheaper session
        a1 = [_asst("claude-opus-4-7", cr=120000, cc=8000, mid="a1", cwd="/projA")
              for _ in range(1)] + [_asst("claude-opus-4-7", cr=120000, mid="a1b", cwd="/projA")]
        a2 = [_asst("claude-opus-4-7", cr=100000, cc=5000, mid="a2", cwd="/projA")]
        b1 = [_asst("claude-opus-4-7", cr=10000, cc=1000, mid="b1", cwd="/projB")]
        sessions = [
            cli.parse_session(_write(d, "aaa10000-0000-0000-0000-000000000000.jsonl", a1)),
            cli.parse_session(_write(d, "aaa20000-0000-0000-0000-000000000000.jsonl", a2)),
            cli.parse_session(_write(d, "bbb10000-0000-0000-0000-000000000000.jsonl", b1)),
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.report_by_project(sessions, 200_000, as_json=True)
        rows = json.loads(buf.getvalue())
        assert len(rows) == 2                       # two projects
        assert rows[0]["project"] == "/projA"       # costliest first
        assert rows[0]["sessions"] == 2             # boots = session count
        assert rows[1]["project"] == "/projB"
        # project boot-floor = sum of per-session boot-floor (write-rate)
        p_in, p_out, p_cw, p_cr = cli.PRICING["opus"]
        expA = (8000 * p_cw) / 1e6 + (5000 * p_cw) / 1e6
        assert abs(rows[0]["boot_floor"] - expA) < 1e-12
        assert rows[0]["cost"] >= rows[1]["cost"]
