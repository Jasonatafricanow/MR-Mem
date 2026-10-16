"""Tree engine compile windows (design 4.7): Add stores raw text at once; the
compiler runs on windows that close at natural breakpoints, and flush compiles
the rest."""

import json
from pathlib import Path

import pytest

from mr_mem_aml.compiler import SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime

pytest.importorskip("lce")


class TreeBackend:
    """One block per NEW message, no matters; optionally carries the tail."""

    def __init__(self, carry_last: int = 0):
        self.windows: list[list[str]] = []
        self.carry_last = carry_last

    def complete(self, prompt: str) -> str:
        material = json.loads(prompt.split("INPUT:\n", 1)[1])
        new = material["NEW"]
        self.windows.append([m["text"] for m in new])
        carry = None
        if self.carry_last and len(new) > self.carry_last:
            carry = new[-self.carry_last]["id"]
        return json.dumps({
            "matters": [],
            "blocks": [{"content": f"{m['role']}: {m['text']}", "sources": [m["id"]], "in": []}
                       for m in new],
            "carry_from": carry,
        })


def runtime(tmp_path: Path, backend, **window) -> MemoryRuntime:
    config = RuntimeConfig(
        data_dir=tmp_path / "data", compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree", compile_window={"soft": 3, "hard": 5, **window},
    )
    return MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))


def add(rt, rid, *texts, start_minute=0):
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t,
                 "timestamp": 1_690_000_000_000 + (start_minute + i) * 60_000}
                for i, t in enumerate(texts)]
    return rt.add({"request_id": rid, "user_id": "u1", "session_id": "s1",
                   "messages": messages})


def test_windows_close_at_the_next_turn_of_the_person_and_flush_compiles_the_rest(tmp_path):
    backend = TreeBackend()
    rt = runtime(tmp_path, backend)
    try:
        assert add(rt, "r1", "u1", "a1")["blocks"] == 0  # below the soft cap: waits
        assert add(rt, "r2", "u2", "a2", start_minute=2)["blocks"] == 0  # no breakpoint yet
        assert backend.windows == []
        assert add(rt, "r3", "u3", "a3", start_minute=4)["blocks"] == 4
        assert backend.windows == [["u1", "a1", "u2", "a2"]]  # closed before "u3"
        assert rt.flush() == {"status": "ok", "windows": 1}
        assert backend.windows[-1] == ["u3", "a3"]
        assert rt.flush() == {"status": "ok", "windows": 0}
    finally:
        rt.close()


def test_a_carried_tail_opens_the_next_window(tmp_path):
    backend = TreeBackend(carry_last=1)
    rt = runtime(tmp_path, backend)
    try:
        add(rt, "r1", "u1", "a1", "u2", "a2", "u3", "a3")  # closes before "u3"
        assert backend.windows == [["u1", "a1", "u2", "a2"]]
        rt.flush()  # "a2" was carried; the final window keeps its own tail
        assert backend.windows[1:] == [["a2", "u3", "a3"]]
        blocks = [m.content for m in rt.core.load_all()]
        assert sorted(blocks) == sorted(
            ["user: u1", "assistant: a1", "user: u2", "assistant: a2", "user: u3",
             "assistant: a3"])
    finally:
        rt.close()


def test_a_long_pause_is_a_breakpoint(tmp_path):
    backend = TreeBackend()
    rt = runtime(tmp_path, backend, gap_minutes=30)
    try:
        add(rt, "r1", "u1", "a1", "u1b")
        add(rt, "r2", "a1b", "u2", start_minute=90)  # "a1b" comes after a 90-minute pause
        assert backend.windows == [["u1", "a1", "u1b"]]
    finally:
        rt.close()


def test_points_link_each_block_to_the_steps_it_carries_on():
    from datetime import UTC, datetime

    from mr_mem_aml.compiler import SemanticCompiler
    from mr_mem_aml.source_store import RawMessage

    msg = RawMessage("r:0", "u", "s", "r", 0, "user", "ate bacalhau in Lisbon; egg was tasty",
                     datetime(2026, 5, 1, tzinfo=UTC), "f")
    raw = json.dumps({"blocks": [
        {"content": "meal", "sources": ["m1"], "after": ["N1", "N4"], "happened": True},
        {"content": "egg", "sources": ["m1"], "after": [], "happened": True},
        {"content": "more", "sources": ["m1"], "after": ["b1"], "happened": False}]})
    meal, egg, more = SemanticCompiler(None, schema="points").parse(raw, (msg,))
    assert [(x.thread, x.after, x.committed) for x in meal.lines] == [
        (None, "N1", True), (None, "N4", True)]  # one block, two lines meet
    assert [(x.thread, x.after) for x in egg.lines] == [(None, None)]  # a point of its own
    assert [(x.after, x.committed) for x in more.lines] == [("b1", False)]


def test_a_journaled_window_is_applied_without_a_model_call(tmp_path):
    backend = TreeBackend()
    rt = runtime(tmp_path / "a", backend)
    try:
        add(rt, "r1", "u1", "a1", "u2", "a2", "u3", "a3")
        compiled = rt.source.open_window("u1")  # applied, so not open
        assert compiled is None and len(backend.windows) == 1
        import sqlite3

        row = sqlite3.connect(tmp_path / "a" / "data" / "aml_source.sqlite").execute(
            "SELECT window_id,user_id,first_seq,last_seq,compiled FROM windows").fetchone()
    finally:
        rt.close()
    replay = TreeBackend()
    rt = runtime(tmp_path / "b", replay)
    try:
        rt.source.put_window(row[0], row[1], row[2], row[3], json.loads(row[4]))
        assert add(rt, "r1", "u1", "a1", "u2", "a2", "u3", "a3")["blocks"] == 4
        assert replay.windows == []  # the journal answered; no model call
    finally:
        rt.close()
