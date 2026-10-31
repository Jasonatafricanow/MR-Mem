"""Tree engine compile windows (design 4.7): Add stores raw text at once; the
compiler runs on windows that close at natural breakpoints, and flush compiles
the rest."""

import json
from datetime import UTC, datetime
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
        return json.dumps(
            {
                "matters": [],
                "blocks": [
                    {"content": f"{m['role']}: {m['text']}", "sources": [m["id"]], "in": []}
                    for m in new
                ],
                "carry_from": carry,
            }
        )


def runtime(tmp_path: Path, backend, scene: str = "chat", **window) -> MemoryRuntime:
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        compile_window={"soft": 3, "hard": 5, **window},
        scene=scene,
    )
    return MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))


def add(rt, rid, *texts, start_minute=0):
    messages = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": t,
            "timestamp": 1_690_000_000_000 + (start_minute + i) * 60_000,
        }
        for i, t in enumerate(texts)
    ]
    return rt.add({"request_id": rid, "user_id": "u1", "session_id": "s1", "messages": messages})


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


def test_work_windows_close_after_the_feedback_before_the_next_report(tmp_path):
    backend = TreeBackend()
    rt = runtime(tmp_path, backend, scene="work")
    try:
        add(rt, "r1", "u1", "a1", "u2", "a2", "u3", "a3")
        assert backend.windows == [["u1", "a1", "u2"]]  # "a1" and its feedback "u2" stay together
        rt.flush()
        assert backend.windows[1:] == [["a2", "u3", "a3"]]
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
            ["user: u1", "assistant: a1", "user: u2", "assistant: a2", "user: u3", "assistant: a3"]
        )
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

    msg = RawMessage(
        "r:0",
        "u",
        "s",
        "r",
        0,
        "user",
        "ate bacalhau in Lisbon; egg was tasty",
        datetime(2026, 5, 1, tzinfo=UTC),
        "f",
    )
    raw = json.dumps(
        {
            "blocks": [
                {"content": "meal", "sources": ["m1"], "after": ["N1", "N4"], "happened": True},
                {"content": "egg", "sources": ["m1"], "after": [], "happened": True},
                {"content": "more", "sources": ["m1"], "after": ["b1"], "happened": False},
            ]
        }
    )
    meal, egg, more = SemanticCompiler(None, schema="points").parse(raw, (msg,))
    assert [(x.thread, x.after, x.committed) for x in meal.lines] == [
        (None, "N1", True),
        (None, "N4", True),
    ]  # one block, two lines meet
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

        row = (
            sqlite3.connect(tmp_path / "a" / "data" / "aml_source.sqlite")
            .execute("SELECT window_id,user_id,first_seq,last_seq,compiled FROM windows")
            .fetchone()
        )
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


class OneBlockBackend:
    """One block citing every NEW message."""

    def complete(self, prompt: str) -> str:
        new = json.loads(prompt.split("INPUT:\n", 1)[1])["NEW"]
        return json.dumps(
            {
                "matters": [],
                "blocks": [{"content": "all", "sources": [m["id"] for m in new], "in": []}],
                "carry_from": None,
            }
        )


def test_a_work_block_running_past_the_feedback_is_counted_not_refused(tmp_path):
    rt = runtime(tmp_path, OneBlockBackend(), scene="work")
    try:
        add(rt, "r1", "u1", "a1", "u2", "a2")
        rt.flush()  # one window: u1 a1 u2 | a2 ... the first block has "a1" after "u1"
        assert len(rt.past_feedback) == 1
        assert len(rt.core.load_all()) == 2  # counted, the Block still stands
    finally:
        rt.close()


def test_a_window_can_be_counted_in_characters(tmp_path):
    backend = TreeBackend()
    rt = runtime(tmp_path, backend, soft=10, hard=1000, measure="chars")
    try:
        add(rt, "r1", "u1", "a much longer answer")  # 22 characters: past soft already
        add(rt, "r2", "u2", "a2", start_minute=2)
        assert backend.windows == [
            ["u1", "a much longer answer"]
        ]  # closed at the person's next turn
    finally:
        rt.close()


def test_a_window_compiled_twice_applies_the_first_journal_entry_not_its_own(tmp_path):
    rt = runtime(tmp_path, TreeBackend())
    try:
        add(rt, "r1", "u1", "a1", "u2", "a2", "u3", "a3")
        rt.source.put_request(
            {
                "request_id": "r2",
                "user_id": "u1",
                "session_id": "s1",
                "messages": [{"role": "user", "content": t} for t in ("u4", "a4", "u5", "a5")],
            },
            received_at=datetime(2026, 10, 30, tzinfo=UTC),
        )
        first, second = rt._window_job("u1", final=True), rt._window_job("u1", final=True)
        assert first.window[0] == second.window[0]  # two threads took the same window
        mine = rt.compile_job(second)
        mine["blocks"] = [
            {**b, "content": b["content"] + " (other output)"} for b in mine["blocks"]
        ]
        rt.finish_add(first, rt.compile_job(first))
        assert rt.finish_add(second, mine).get("replayed")  # no identity conflict, nothing twice
    finally:
        rt.close()


def test_flushing_one_users_history_leaves_another_users_alone(tmp_path):
    from mr_mem_aml.server import RuntimeHost

    backend = TreeBackend()
    host = RuntimeHost(lambda: runtime(tmp_path, backend, soft=100, hard=200))
    try:
        for user in ("a", "b"):
            host.add(
                {
                    "request_id": f"r{user}",
                    "user_id": user,
                    "session_id": user,
                    "messages": [
                        {"role": "user", "content": f"{user}1"},
                        {"role": "assistant", "content": f"{user}2"},
                    ],
                }
            )
        host.flush("a")
        assert backend.windows == [["a1", "a2"]]  # b is still adding: its tail is not closed for it
        assert host.call(host.runtime.source.users_with_pending) == ("b",)
    finally:
        host.call(host.runtime.close)


def test_a_compile_that_keeps_failing_keeps_the_raw_and_add_still_succeeds(tmp_path):
    from mr_mem_aml.compiler import CompileError

    class Down:
        windows = []

        def complete(self, prompt):
            raise CompileError("model down")

    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        compile_window={"soft": 3, "hard": 5},
        keep_raw_on_compile_failure=True,
        placement="points",
        raw_base=True,
        line_read={"projection": "relay", "raw_fill": 0, "hit_anchors": True},
    )
    rt = MemoryRuntime(config, compiler=SemanticCompiler(Down(), schema="tree"))
    try:
        out = add(rt, "r1", "u1", "a1", "u2", "a2", "u3", "a3")
        assert out["success"] is True and rt.compile_failed_kept_raw >= 1
        assert rt.source.open_window("u1") is None  # the window is passed, not retried on every Add
        found = rt.search({"user_id": "u1", "query": "u1", "top_k": 100})["data"]
        assert any("u1" in item["content"] for item in found)
        assert rt.search({"user_id": "another", "query": "u1"})["data"] == []
    finally:
        rt.close()


def test_search_first_compiles_what_is_left_of_that_users_history(tmp_path):
    from mr_mem_aml.server import RuntimeHost

    backend = TreeBackend()
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        compile_window={"soft": 100, "hard": 200},
        compile_before_search=True,
    )
    host = RuntimeHost(
        lambda: MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))
    )
    try:
        host.add(
            {
                "request_id": "r1",
                "user_id": "a",
                "session_id": "s",
                "messages": [
                    {"role": "user", "content": "a1"},
                    {"role": "assistant", "content": "a2"},
                ],
            }
        )
        assert backend.windows == []  # the tail is still open
        host.search({"query": "a1?", "user_id": "a", "top_k": 5}, None)
        assert backend.windows == [["a1", "a2"]]
    finally:
        host.call(host.runtime.close)


class SlowBackend(TreeBackend):
    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = seconds

    def complete(self, prompt: str) -> str:
        import time

        time.sleep(self.seconds)
        return super().complete(prompt)


def test_background_compile_add_returns_at_once_and_search_waits_at_most_search_wait(tmp_path):
    import time

    from mr_mem_aml.server import RuntimeHost

    backend = SlowBackend(1.0)
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        compile_window={"soft": 3, "hard": 5},
        compile_before_search=True,
        background_compile={"search_wait": 0.2, "idle_tail": 60},
    )
    host = RuntimeHost(
        lambda: MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))
    )
    texts = ["a1", "a2", "a3", "a4", "a5", "a6"]
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)
    ]
    try:
        start = time.monotonic()
        got = host.add(
            {"request_id": "r1", "user_id": "a", "session_id": "s", "messages": messages}
        )
        assert got["request_id"] == "r1" and time.monotonic() - start < 0.8  # not waiting on it
        start = time.monotonic()
        host.search({"query": "a1?", "user_id": "a", "top_k": 5}, None)  # searchable meanwhile
        assert time.monotonic() - start < 0.8
        deadline = time.monotonic() + 10
        while sum(map(len, backend.windows)) < len(texts) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sum(map(len, backend.windows)) == len(texts)  # every window, tail included
    finally:
        host.close()


def test_a_window_of_many_short_messages_closes_at_max_messages(tmp_path):
    # by characters these never reach soft; step 1's output grows with the message count
    backend = TreeBackend()
    rt = runtime(tmp_path, backend, soft=10_000, hard=20_000, measure="chars", max_messages=6)
    try:
        add(rt, "r1", *[f"m{i}" for i in range(20)])
        rt.flush()
        assert all(len(w) <= 6 for w in backend.windows)
        assert sum(map(len, backend.windows)) == 20
    finally:
        rt.close()


def test_after_a_restart_background_compile_picks_up_what_was_left(tmp_path):
    import time

    from mr_mem_aml.server import RuntimeHost

    backend = TreeBackend()
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        compile_window={"soft": 3, "hard": 5},
        compile_before_search=True,
        background_compile={"search_wait": 0.2, "idle_tail": 0.1},
    )
    texts = ["a1", "a2", "a3", "a4", "a5", "a6"]
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)
    ]
    rt = MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))
    rt.source.put_request(  # stored before the restart, never compiled
        {"request_id": "r1", "user_id": "a", "session_id": "s", "messages": messages},
        received_at=datetime.now(UTC),
    )
    rt.close()
    assert backend.windows == []
    host = RuntimeHost(
        lambda: MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))
    )
    try:
        deadline = time.monotonic() + 10
        while sum(map(len, backend.windows)) < len(texts) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sum(map(len, backend.windows)) == len(texts)  # no Add, no Search needed
    finally:
        host.close()


def test_a_history_past_max_chars_per_user_keeps_raw_only(tmp_path):
    backend = TreeBackend()
    rt = runtime(tmp_path, backend, soft=4, hard=6, max_chars_per_user=8)
    try:
        add(rt, "r1", *[f"m{i}" for i in range(12)])  # 2-3 characters each
        rt.flush()
        compiled = sum(map(len, backend.windows))
        assert 0 < compiled < 12 and rt.compile_capped > 0  # the rest never went to the model
        assert rt.source.users_with_pending() == ()  # capped windows are done, not left pending
    finally:
        rt.close()


def test_backpressure_holds_an_add_while_the_users_backlog_is_compiling(tmp_path):
    import time

    from mr_mem_aml.runtime import RetryableError
    from mr_mem_aml.server import RuntimeHost

    backend = SlowBackend(1.0)
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        compile_window={"soft": 3, "hard": 5},
        compile_before_search=True,
        background_compile={
            "search_wait": 0.2,
            "idle_tail": 0.1,
            "max_backlog": 4,
            "add_wait": 0.3,
        },
    )
    host = RuntimeHost(
        lambda: MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="tree"))
    )
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"a{i}"} for i in range(6)
    ]
    payload = {"request_id": "r1", "user_id": "a", "session_id": "s", "messages": messages}
    try:
        with pytest.raises(RetryableError):  # stored, but the backlog is still compiling
            host.add(payload)
        deadline = time.monotonic() + 10
        while True:  # AML retries the same Add; it answers once the backlog is down
            try:
                got = host.add(payload)
                break
            except RetryableError:
                assert time.monotonic() < deadline
                time.sleep(0.2)
        assert got["request_id"] == "r1"
    finally:
        host.close()


def test_a_window_with_no_sb_skips_the_stretch_step(tmp_path):
    # capped windows (and compiles that kept raw only) have no SB; the stretch step must not
    # run on them (with 8+ closed stretches it averaged no vectors and failed every retry)
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree",
        placement="moves",
        step1="topic",
        layer2="stretches",
        scene="chat",
        compile_window={"soft": 4, "hard": 6, "max_chars_per_user": 1},
    )

    class NoSBs:  # step 1 finds nothing to compile: an empty list, which it accepts
        def complete(self, prompt: str) -> str:
            return "[]"

    rt = MemoryRuntime(config, compiler=SemanticCompiler(NoSBs(), schema="moves", step1="topic"))

    def no_stretch(*args, **kwargs):
        raise AssertionError("the stretch step ran on a window with no SB")

    rt._stretch = no_stretch
    try:
        add(rt, "r1", *[f"m{i}" for i in range(6)])
        add(rt, "r2", *[f"n{i}" for i in range(6)])
        rt.flush()
        assert rt.compile_capped > 0 and rt.source.users_with_pending() == ()
    finally:
        rt.close()
