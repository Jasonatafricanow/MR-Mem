"""End-to-end AML runtime with a deterministic fake compiler (tests/aml_live_backend.py) and
real Line trees: the Add/Search contract, journals, retries, restarts, HTTP and the source
store's time rules."""

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from mr_mem_aml.compiler import CompileError, SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import ConflictingReplay, InvalidRequest, MemoryRuntime, RetryableError
from tests.aml_live_backend import LiveBackend

pytest.importorskip("lce")


def runtime(tmp_path: Path, backend=None, **overrides) -> tuple[MemoryRuntime, LiveBackend]:
    backend = backend or LiveBackend()
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        **{"raw_base": True, **overrides},
    )
    return MemoryRuntime(config, compiler=SemanticCompiler(backend)), backend


def add(rt, rid, *texts, session="s1", user="u1", start=1_690_000_000_000):
    messages = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": t,
            "timestamp": start + i * 60_000,
        }
        for i, t in enumerate(texts)
    ]
    return rt.add({"request_id": rid, "user_id": user, "session_id": session, "messages": messages})


def ids(result):
    return [item["id"].split(":")[0] for item in result["data"]]


def test_add_then_search_and_replay(tmp_path):
    rt, backend = runtime(tmp_path, line_read={})
    try:
        first = add(rt, "r1", "I adopted a cat named Miso", "Congrats!")
        assert {k: first[k] for k in ("success", "request_id", "user_id", "session_id")} == {
            "success": True,
            "request_id": "r1",
            "user_id": "u1",
            "session_id": "s1",
        }
        assert rt.flush() == {"status": "ok", "windows": 1}
        calls = backend.calls
        again = add(rt, "r1", "I adopted a cat named Miso", "Congrats!")
        assert again["replayed"] is True and again["success"] is True
        assert backend.calls == calls
        with pytest.raises(ConflictingReplay):
            add(rt, "r1", "different content")
        found = rt.search({"query": "what is the cat called", "user_id": "u1", "top_k": 10})
        assert any("Miso" in item["content"] for item in found["data"])
        assert any(item.get("created_at", "").startswith("2023-07-22") for item in found["data"])
        assert rt.search({"query": "cat", "user_id": "other", "top_k": 5}) == {"data": []}
    finally:
        rt.close()


def test_compiled_journal_survives_crash_without_second_model_call(tmp_path, monkeypatch):
    rt, backend = runtime(tmp_path)
    try:
        original = rt._admit
        state = {"crash": True}

        def crash_once(*args):
            if state["crash"]:
                state["crash"] = False
                raise RetryableError("simulated crash after compile")
            return original(*args)

        monkeypatch.setattr(rt, "_admit", crash_once)
        add(rt, "r1", "Remember my passport expires in May")
        with pytest.raises(RetryableError):
            rt.flush()
        windows = len(backend.windows)
        assert rt.flush()["windows"] == 1  # the journaled window is applied
        assert len(backend.windows) == windows  # without a second step-1 call
        assert len(rt.core.load_all()) == 1
    finally:
        rt.close()


def test_an_invalid_model_answer_is_asked_again(tmp_path):
    backend = LiveBackend()
    backend.fail_next = 1
    rt, _ = runtime(tmp_path, backend)
    try:
        add(rt, "r1", "hello there, I run on Tuesdays")
        rt.flush()
        assert len(backend.windows) == 1 and backend.calls == 3  # bad answer, step 1, stretches
        assert len(rt.core.load_all()) == 1
    finally:
        rt.close()


def test_restart_keeps_everything(tmp_path):
    rt, _ = runtime(tmp_path, line_read={})
    add(rt, "r1", "I want a new laptop", "Which one?", "A 16GB model")
    rt.flush()
    rt.close()
    rt2, backend2 = runtime(tmp_path, line_read={})
    try:
        data = rt2.search({"query": "laptop", "user_id": "u1", "top_k": 10})["data"]
        assert any("laptop" in item["content"] for item in data) and backend2.calls == 0
        assert len(rt2.core.load_all()) == 3
    finally:
        rt2.close()


def test_http_surface(tmp_path):
    from mr_mem_aml.server import RuntimeHost, build_server

    # The owner thread holds every store; HTTP requests run on their own threads.
    host = RuntimeHost(lambda: runtime(tmp_path)[0])
    config = host.runtime.config
    server = build_server(
        RuntimeConfig(**{**config.__dict__, "port": 0}),
        host,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body, headers=None):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())

    try:
        status, body = post(
            "/add",
            {
                "request_id": "h1",
                "user_id": "u1",
                "session_id": "s",
                "messages": [{"role": "user", "content": "I like tea"}],
            },
        )
        assert status == 200 and body["success"] is True
        status, body = post(
            "/search",
            {"query": "tea", "user_id": "u1", "top_k": 5},
            {"X-MRMem-Search-Mode": "baseline"},
        )
        assert status == 200 and body["data"][0]["content"].endswith("I like tea")
        with pytest.raises(urllib.error.HTTPError) as err:
            post(
                "/add",
                {
                    "request_id": "h1",
                    "user_id": "u1",
                    "session_id": "s",
                    "messages": [{"role": "user", "content": "changed"}],
                },
            )
        assert err.value.code == 422  # permanent; AML would retry a 409
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            assert json.loads(response.read())["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(30)
        host.close()


def test_source_text_for_hits_applies_to_both_modes(tmp_path):
    from mr_mem.integrations.path_a_read import ReadPolicy

    rt, _ = runtime(tmp_path, raw_base=False, read_policy=ReadPolicy(source_text_for_hits=True))
    try:
        add(rt, "r1", "I like green tea")
        rt.flush()
        for mode in ("baseline", "path_a"):
            data = rt.search({"query": "tea", "user_id": "u1", "top_k": 5}, mode=mode)["data"]
            assert [i["id"].split(":")[0] for i in data] == ["memory", "source"]
            assert data[1]["content"].endswith("user: I like green tea")
    finally:
        rt.close()


def test_auth_accepts_aml_schemes_and_search_accepts_options(tmp_path):
    from http.server import BaseHTTPRequestHandler

    from mr_mem_aml.server import make_handler

    rt, _ = runtime(tmp_path)
    try:
        handler = make_handler(rt, "sekret")

        def authorized(headers):
            probe = BaseHTTPRequestHandler.__new__(handler)
            probe.headers = headers
            return handler._authorized(probe)

        assert authorized({"Authorization": "Bearer sekret"})
        assert authorized({"Authorization": "Token sekret"})
        assert authorized({"X-Api-Key": "sekret"})
        assert not authorized({"Authorization": "Bearer wrong"})
        assert not authorized({})
        add(rt, "r1", "I prefer window seats on flights")
        found = rt.search(
            {
                "query": "which window seats",
                "user_id": "u1",
                "top_k": 1,
                "options": ["aisle", "window"],
            }
        )
        assert len(found["data"]) == 1
        with pytest.raises(InvalidRequest):
            rt.search({"query": "seat", "user_id": "u1", "top_k": 5, "options": "window"})
    finally:
        rt.close()


class SlowBackend(LiveBackend):
    """Records overlapping model calls to prove cross-user compile concurrency."""

    def __init__(self, delay):
        super().__init__()
        self.delay, self.active, self.peak = delay, 0, 0
        self.lock = threading.Lock()

    def complete(self, prompt):
        import time

        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            out = super().complete(prompt)
            self.active -= 1
        return out


def test_compiles_run_in_parallel_across_users_and_in_order_within_a_user(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from mr_mem_aml.server import RuntimeHost

    backend = SlowBackend(0.3)
    host = RuntimeHost(
        lambda: runtime(
            tmp_path,
            backend=backend,
            compile_concurrency=4,
            compile_window={"soft": 1, "hard": 1},  # each message its own window
        )[0]
    )

    def send(user, index):
        return host.add(
            {
                "request_id": f"{user}-{index}",
                "user_id": user,
                "session_id": "s",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{user} note {index}",
                        "timestamp": 1_690_000_000_000 + index * 60_000,
                    }
                ],
            }
        )

    try:
        with ThreadPoolExecutor(8) as pool:
            # Like an AML client: chunks of one user are sent one after another,
            # different users are sent concurrently.
            jobs = [
                pool.submit(lambda u: [send(u, i) for i in range(3)], user)
                for user in ("u1", "u2", "u3", "u4")
            ]
            assert all(r["success"] for job in jobs for r in job.result())
        host.flush()
        assert backend.peak >= 2  # different users overlapped
        order = [w[0] for w in backend.windows]
        for user in ("u1", "u2", "u3", "u4"):
            mine = [text for text in order if text.startswith(user)]
            assert mine == [f"{user} note {i}" for i in range(3)]  # per-user order kept
        assert len(host.call(host.runtime.core.load_all)) == 12
    finally:
        host.close()


def test_raw_base_searches_raw_messages_and_rebuilds_after_a_restart(tmp_path):
    rt, _ = runtime(tmp_path)
    try:
        add(rt, "r1", "No money, not buying that laptop", "Makes sense.")
        add(rt, "r2", "Went hiking at the lake", start=1_690_100_000_000)
        query = {"query": "buying laptop money", "user_id": "u1", "top_k": 10}
        baseline = rt.search(query, mode="baseline")["data"]
        assert set(ids({"data": baseline})) == {"raw"}
        assert baseline[0]["content"].startswith("User: No money")  # P1 unit text
        rt.close()
        rt, _ = runtime(tmp_path)  # raw index rebuilds from stored messages
        assert ids({"data": rt.search(query, mode="baseline")["data"]})[0] == "raw"
    finally:
        rt.close()


def test_rejected_compile_keeps_raw_evidence_when_configured(tmp_path):
    class Rejecting:
        def complete(self, prompt):
            raise CompileError("compiler endpoint rejected the request: HTTP 400: moderation")

    rt, _ = runtime(tmp_path, backend=Rejecting(), skip_rejected_compiles=True)
    try:
        assert add(rt, "r1", "Something the provider refuses")["success"]
        rt.flush()
        assert rt.compile_rejected == 1
        data = rt.search(
            {"query": "provider refuses", "user_id": "u1", "top_k": 5}, mode="baseline"
        )["data"]
        assert data and data[0]["content"].startswith("User: Something")
    finally:
        rt.close()


def test_message_time_is_only_claimed_when_the_source_gave_it(tmp_path):
    from datetime import UTC, datetime

    from mr_mem_aml.source_store import AmlSourceStore

    store = AmlSourceStore(tmp_path / "s.sqlite")
    store.put_request(
        {
            "request_id": "q1",
            "user_id": "u",
            "session_id": "s",
            "messages": [
                {"role": "user", "content": "timed", "timestamp": 1_700_000_000_000},
                {"role": "user", "content": "untimed"},
            ],
        },
        received_at=datetime(2026, 10, 14, tzinfo=UTC),
    )
    timed, untimed = store.request_messages("q1")
    assert store.time_given("q1:0") and not store.time_given("q1:1")
    assert (timed.time_given, untimed.time_given) == (True, False)
    assert store.render(untimed.ref) == "[Original message] user: untimed"

    class Seen:
        def complete(self, prompt):
            self.prompt = prompt
            return "JSON:\n[]\nSTOP: M2"

    seen = Seen()
    SemanticCompiler(seen).compile((timed, untimed), (), {"language": "en"})
    assert seen.prompt.count("2023-11-14") == 1  # no invented time for the untimed message
    assert seen.prompt.count("2026-10-14") == 0  # nor the storage time
    store.close()


def test_an_untimed_message_is_dated_by_its_session_so_far_never_by_storage(tmp_path):
    from datetime import UTC, datetime

    from mr_mem_aml.line_read import _when
    from mr_mem_aml.source_store import AmlSourceStore

    store = AmlSourceStore(tmp_path / "s.sqlite")
    stored = datetime(2026, 10, 14, tzinfo=UTC)
    store.put_request(
        {
            "request_id": "q1",
            "user_id": "u",
            "session_id": "a",
            "messages": [
                {"role": "user", "content": "before any time"},
                {"role": "user", "content": "timed", "timestamp": 1_710_432_000_000},  # 2024-03-14
                {"role": "user", "content": "after it"},
            ],
        },
        received_at=stored,
    )
    store.put_request(
        {
            "request_id": "q2",
            "user_id": "u",
            "session_id": "b",
            "messages": [{"role": "user", "content": "other session, untimed"}],
        },
        received_at=stored,
    )
    at = store.session_times("u")
    assert set(at) == {"q1:2"}  # not from a later message, another session, or the storage time
    assert at["q1:2"].date().isoformat() == "2024-03-14"
    assert _when(None, 0, (1, 3), at["q1:2"], "en") == "S1 #3, session began 2024-03-14"
    assert _when(None, 0, (2, 4), None, "en") == "S2 #4"
    store.close()


def test_a_threads_summary_is_kept_per_window_and_read_as_the_latest(tmp_path):
    from mr_mem_aml.source_store import AmlSourceStore

    store = AmlSourceStore(tmp_path / "s.sqlite")
    store.put_threads("u", "w1", 10, {"m1": "t:m1"}, {"t:m1": "coverage 65%"})
    store.put_threads(
        "u", "w1", 10, {"m1": "t:other"}, {"t:m1": "replayed differently"}
    )  # first wins
    store.put_threads("u", "w2", 20, {"m2": "t:m1"}, {"t:m1": "coverage 65%, later 78%"})
    got = store.threads("u")
    assert got == {
        "t:m1": {"summary": "coverage 65%, later 78%", "last_seq": 20, "members": ["m1", "m2"]}
    }
    store.close()


def test_a_thread_begun_off_another_threads_step_keeps_where_and_why(tmp_path):
    from mr_mem_aml.source_store import AmlSourceStore

    store = AmlSourceStore(tmp_path / "s.sqlite")
    store.put_threads(
        "u",
        "w1",
        10,
        {"m1": "t:m1"},
        {"t:m1": "Because builds failed, the person on 03-14 chose another model"},
    )
    store.put_threads(
        "u",
        "w2",
        20,
        {"m2": "t:m2"},
        {"t:m2": "the person turned to the readout"},
        {"t:m2": {"thread": "t:m1", "step": "m1", "why": "the model was settled"}},
    )
    assert store.threads("u")["t:m2"]["origin"] == {
        "thread": "t:m1",
        "step": "m1",
        "why": "the model was settled",
    }
    assert "origin" not in store.threads("u")["t:m1"]
    store.close()


def test_configured_auth_cannot_silently_become_public(tmp_path, monkeypatch):
    from mr_mem_aml.server import build_server
    config = RuntimeConfig(data_dir=tmp_path, compiler={"type": "command", "argv": ["x"]},
                           auth_token_env="MRMEM_TEST_MISSING_TOKEN")
    monkeypatch.delenv("MRMEM_TEST_MISSING_TOKEN", raising=False)
    with pytest.raises(ValueError, match="authentication"):
        build_server(config, None)
    monkeypatch.setenv("MRMEM_TEST_MISSING_TOKEN", " ")
    with pytest.raises(ValueError, match="authentication"):
        build_server(config, None)
