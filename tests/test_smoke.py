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


def test_window_independent_grade():
    """Same session must get the same grade regardless of --window (cost != capacity)."""
    with tempfile.TemporaryDirectory() as d:
        r = cli.parse_session(_write_session(d))
        g200, _ = cli.grade_session(r, window=200_000)
        g1m, _ = cli.grade_session(r, window=1_000_000)
        assert g200 == g1m
