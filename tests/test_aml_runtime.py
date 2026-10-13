"""End-to-end AML runtime with a deterministic fake compiler and real LCE."""

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from mr_mem_aml.compiler import CompileError, SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import ConflictingReplay, InvalidRequest, MemoryRuntime, RetryableError

pytest.importorskip("lce")


class ScriptedBackend:
    """Compiles NEW messages by a tiny script embedded in each user message.

    "text || track:Q|summary|mature" adds a Thread signal; "|| commit:Q|summary"
    adds a first-appearance commit. Assistant messages become attributed blocks.
    """

    def __init__(self):
        self.calls = 0
        self.fail_next = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.fail_next:
            self.fail_next -= 1
            return "not json"
        material = json.loads(prompt.split("INPUT:\n", 1)[1].split("\n\nYour previous", 1)[0])
        blocks = []
        for message in material["NEW"]:
            text, _, script = message["text"].partition(" || ")
            who = "The user" if message["role"] == "user" else "The assistant"
            block = {"content": f"{who} said: {text}", "sources": [message["id"]],
                     "thread": None, "commit": None}
            if script.startswith("track:"):
                question, summary, mature = (script[6:].split("|") + ["", ""])[:3]
                block["thread"] = {"action": "track", "question": question,
                                   "summary": summary or None, "mature": mature == "mature"}
            elif script.startswith("commit:"):
                question, summary = script[7:].split("|")
                block["commit"] = {"question": question, "summary": summary}
            blocks.append(block)
        return json.dumps({"blocks": blocks})


def runtime(tmp_path: Path, backend=None, **overrides) -> tuple[MemoryRuntime, ScriptedBackend]:
    backend = backend or ScriptedBackend()
    config = RuntimeConfig(data_dir=tmp_path / "data", compiler={"type": "command", "argv": ["x"]},
                           **overrides)
    return MemoryRuntime(config, compiler=SemanticCompiler(backend)), backend


def add(rt, rid, *texts, session="s1", user="u1", start=1_690_000_000_000):
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t,
         "timestamp": start + i * 60_000}
        for i, t in enumerate(texts)
    ]
    return rt.add({"request_id": rid, "user_id": user, "session_id": session,
                   "messages": messages})


def ids(result):
    return [item["id"].split(":")[0] for item in result["data"]]


def test_add_then_search_and_replay(tmp_path):
    rt, backend = runtime(tmp_path)
    try:
        first = add(rt, "r1", "I adopted a cat named Miso", "Congrats!")
        assert first["blocks"] == 2
        assert {k: first[k] for k in ("success", "request_id", "user_id", "session_id")} == {
            "success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"}
        again = add(rt, "r1", "I adopted a cat named Miso", "Congrats!")
        assert again["replayed"] is True and again["success"] is True
        assert backend.calls == 1
        with pytest.raises(ConflictingReplay):
            add(rt, "r1", "different content")
        found = rt.search({"query": "what is the cat called", "user_id": "u1", "top_k": 10})
        assert found["data"][0]["content"].endswith("The user said: I adopted a cat named Miso")
        assert found["data"][0]["created_at"].startswith("2023-07-22")
        assert rt.search({"query": "cat", "user_id": "other", "top_k": 5}) == {"data": []}
    finally:
        rt.close()


def test_thread_matures_across_requests_and_reads_first_with_evidence(tmp_path):
    rt, _ = runtime(tmp_path)
    try:
        add(rt, "r1", "I want a new laptop || track:Replace laptop?|Considering it")
        add(rt, "r2", "Prices are too high, I will wait || track:Replace laptop?|"
                      "Postponed because of price|mature", start=1_690_100_000_000)
        query = {"query": "laptop price decision", "user_id": "u1", "top_k": 20}
        path_a = rt.search(query, mode="path_a")["data"]
        assert path_a[0]["id"].startswith("lce:")
        assert "Postponed because of price" in path_a[0]["content"]
        assert "as of 2023-07-23" in path_a[0]["content"]
        kinds = ids({"data": path_a})
        assert kinds[1:3] == ["memory", "memory"]  # canonical support first
        assert "source" in kinds  # then the original messages
        baseline = rt.search(query, mode="baseline")["data"]
        assert all(item["id"].startswith("memory:") for item in baseline)
        support = {i["id"]: i["content"] for i in path_a if i["id"].startswith("memory:")}
        assert all(support[i["id"]] == i["content"] for i in baseline if i["id"] in support)
    finally:
        rt.close()


def test_open_thread_is_read_before_maturity(tmp_path):
    rt, _ = runtime(tmp_path)
    try:
        add(rt, "r1", "Thinking about moving to Lisbon || track:Move to Lisbon?|Exploring")
        data = rt.search({"query": "Lisbon move", "user_id": "u1", "top_k": 10})["data"]
        assert data[0]["id"].startswith("thread:")
        assert "Question: Move to Lisbon?" in data[0]["content"]
    finally:
        rt.close()


def test_first_appearance_commit_reaches_lce(tmp_path):
    rt, _ = runtime(tmp_path)
    try:
        add(rt, "r1", "A single volume spike never confirms a breakout; I wait for a retest "
                      "|| commit:When is a breakout confirmed?|Only after a held retest")
        data = rt.search({"query": "breakout confirmed", "user_id": "u1", "top_k": 10})["data"]
        assert data[0]["id"].startswith("lce:")
        assert "Only after a held retest" in data[0]["content"]
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
        with pytest.raises(RetryableError):
            add(rt, "r1", "Remember my passport expires in May")
        assert add(rt, "r1", "Remember my passport expires in May")["success"] is True
        assert backend.calls == 1
    finally:
        rt.close()


def test_invalid_model_output_retries_then_fails_retryably(tmp_path):
    backend = ScriptedBackend()
    backend.fail_next = 1
    rt, _ = runtime(tmp_path, backend)
    try:
        assert add(rt, "r1", "hello there, I run on Tuesdays")["success"] is True
        assert backend.calls == 2
        backend.fail_next = 10
        with pytest.raises(RetryableError):
            add(rt, "r2", "another message")
        backend.fail_next = 0
        assert add(rt, "r2", "another message")["success"] is True
    finally:
        rt.close()


def test_restart_keeps_everything(tmp_path):
    rt, _ = runtime(tmp_path)
    add(rt, "r1", "I want a new laptop || track:Replace laptop?|Considering it")
    add(rt, "r2", "I bought it || track:Replace laptop?|Bought a new laptop|mature",
        start=1_690_100_000_000)
    rt.close()
    rt2, backend2 = runtime(tmp_path)
    try:
        data = rt2.search({"query": "laptop", "user_id": "u1", "top_k": 10})["data"]
        assert data[0]["id"].startswith("lce:") and backend2.calls == 0
    finally:
        rt2.close()


def test_context_only_policy_drops_assistant_blocks():
    from datetime import UTC, datetime

    from mr_mem_aml.source_store import RawMessage

    now = datetime(2026, 1, 1, tzinfo=UTC)
    new = (RawMessage("r:0", "u", "s", "r", 0, "assistant", "Try yoga", now, "f"),)
    compiler = SemanticCompiler(ScriptedBackend(), assistant_policy="context_only", attempts=1)
    with pytest.raises(CompileError):
        compiler.compile(new, ())


def test_http_surface(tmp_path):
    from mr_mem_aml.server import RuntimeHost, build_server

    # The owner thread holds every store; HTTP requests run on their own threads.
    host = RuntimeHost(lambda: runtime(tmp_path)[0])
    config = host.runtime.config
    server = build_server(
        RuntimeConfig(**{**config.__dict__, "port": 0}), host,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body, headers=None):
        request = urllib.request.Request(
            base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())

    try:
        status, body = post("/add", {"request_id": "h1", "user_id": "u1", "session_id": "s",
                                     "messages": [{"role": "user", "content": "I like tea"}]})
        assert status == 200 and body["success"] is True
        status, body = post("/search", {"query": "tea", "user_id": "u1", "top_k": 5},
                            {"X-MRMem-Search-Mode": "baseline"})
        assert status == 200 and body["data"][0]["content"].endswith("I like tea")
        with pytest.raises(urllib.error.HTTPError) as err:
            post("/add", {"request_id": "h1", "user_id": "u1", "session_id": "s",
                          "messages": [{"role": "user", "content": "changed"}]})
        assert err.value.code == 422  # permanent; AML would retry a 409
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            assert json.loads(response.read())["lce_enabled"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(30)
        host.close()


def test_only_projections_with_relevant_support_are_read(tmp_path):
    from mr_mem.integrations.path_a_read import ReadPolicy

    rt, _ = runtime(tmp_path, read_policy=ReadPolicy(support_rank_window=1))
    try:
        add(rt, "r1", "I stopped drinking coffee because it ruins my sleep "
                      "|| commit:Should I drink coffee?|No, it ruins sleep")
        add(rt, "r2", "My savings goal is twenty thousand dollars for a house deposit "
                      "|| commit:What is the savings goal?|20k for a house deposit",
            start=1_690_100_000_000)
        data = rt.search({"query": "coffee sleep", "user_id": "u1", "top_k": 10})["data"]
        lce = [item["content"] for item in data if item["id"].startswith("lce:")]
        assert len(lce) == 1 and "ruins sleep" in lce[0]
    finally:
        rt.close()


def test_settling_signal_on_compiled_line_revises_it(tmp_path):
    rt, _ = runtime(tmp_path)
    try:
        add(rt, "r1", "I want a new laptop || track:Replace laptop?|Considering it",
            "ok", "I need 16GB || track:Replace laptop?|Wants a 16GB model|mature")
        assert "Wants a 16GB model" in rt.search(
            {"query": "laptop", "user_id": "u1", "top_k": 5})["data"][0]["content"]
        add(rt, "r2", "Not buying until after Japan || track:Replace laptop?|"
                      "Postponed until after the Japan trip|mature", start=1_690_100_000_000)
        top = rt.search({"query": "laptop", "user_id": "u1", "top_k": 5})["data"][0]
        assert "revision 2" in top["content"]
        assert "Postponed until after the Japan trip" in top["content"]
        assert rt.core.products.list_threads(rt.core.load_all()[0].scope) == ()
    finally:
        rt.close()


def test_source_text_for_hits_applies_to_both_modes(tmp_path):
    from mr_mem.integrations.path_a_read import ReadPolicy

    rt, _ = runtime(tmp_path, read_policy=ReadPolicy(source_text_for_hits=True))
    try:
        add(rt, "r1", "I like green tea")
        for mode in ("baseline", "path_a"):
            data = rt.search({"query": "tea", "user_id": "u1", "top_k": 5}, mode=mode)["data"]
            assert [i["id"].split(":")[0] for i in data] == ["memory", "source"]
            assert data[1]["content"].endswith("user: I like green tea")
    finally:
        rt.close()


def test_thread_superseded_by_newer_lineage_head_is_not_served(tmp_path):
    rt, _ = runtime(tmp_path)
    try:
        add(rt, "r1", "Budget is capped at 10k || commit:What is the budget?|Capped at 10k")
        add(rt, "r2", "Maybe the cap could move || track:What is the budget?|Cap may change",
            start=1_690_100_000_000)
        query = {"query": "budget cap", "user_id": "u1", "top_k": 10}
        kinds = ids(rt.search(query))
        assert "thread" in kinds  # the open Thread is newer than revision 1
        add(rt, "r3", "Final: the cap is now 12k || track:What is the budget?|Cap is 12k|mature",
            start=1_690_200_000_000)
        data = rt.search(query)["data"]
        assert "thread" not in ids({"data": data})  # revision 2 supersedes it
        assert "Cap is 12k" in data[0]["content"] and "revision 2" in data[0]["content"]
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
        found = rt.search({"query": "which window seats", "user_id": "u1", "top_k": 1,
                           "options": ["aisle", "window"]})
        assert len(found["data"]) == 1
        with pytest.raises(InvalidRequest):
            rt.search({"query": "seat", "user_id": "u1", "top_k": 5, "options": "window"})
    finally:
        rt.close()


class SlowBackend(ScriptedBackend):
    """Records overlapping model calls to prove cross-user compile concurrency."""

    def __init__(self, delay):
        super().__init__()
        self.delay, self.active, self.peak, self.order = delay, 0, 0, []
        self.lock = threading.Lock()

    def complete(self, prompt):
        import time

        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        out = super().complete(prompt)
        with self.lock:
            self.active -= 1
            new = prompt.split("INPUT:" + chr(10), 1)[1]
            material = json.loads(new.split(chr(10) * 2 + "Your previous", 1)[0])
            self.order.append(material["NEW"][0]["text"])
        return out


def test_compiles_run_in_parallel_across_users_and_in_order_within_a_user(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from mr_mem_aml.server import RuntimeHost

    backend = SlowBackend(0.3)
    host = RuntimeHost(lambda: runtime(tmp_path, backend=backend, compile_concurrency=4)[0])

    def send(user, index):
        return host.add({
            "request_id": f"{user}-{index}", "user_id": user, "session_id": "s",
            "messages": [{"role": "user", "content": f"{user} note {index}",
                          "timestamp": 1_690_000_000_000 + index * 60_000}],
        })

    try:
        with ThreadPoolExecutor(8) as pool:
            # Like an AML client: chunks of one user are sent one after another,
            # different users are sent concurrently.
            jobs = [
                pool.submit(lambda u: [send(u, i) for i in range(3)], user)
                for user in ("u1", "u2", "u3", "u4")
            ]
            assert all(r["success"] for job in jobs for r in job.result())
        assert backend.peak >= 2  # different users overlapped
        for user in ("u1", "u2", "u3", "u4"):
            mine = [text for text in backend.order if text.startswith(user)]
            assert mine == [f"{user} note {i}" for i in range(3)]  # per-user order kept
        found = host.search({"query": "note", "user_id": "u3", "top_k": 10}, None)
        assert len(found["data"]) == 3
    finally:
        host.close()


def test_compile_failure_through_host_is_retryable(tmp_path):
    from mr_mem_aml.runtime import CompileFailed
    from mr_mem_aml.server import RuntimeHost

    backend = ScriptedBackend()
    host = RuntimeHost(lambda: runtime(tmp_path, backend=backend)[0])
    payload = {"request_id": "r1", "user_id": "u1", "session_id": "s",
               "messages": [{"role": "user", "content": "I keep bees"}]}
    try:
        backend.fail_next = 3
        with pytest.raises(CompileFailed):
            host.add(payload)
        assert host.add(payload)["blocks"] == 1  # same request resumes
    finally:
        host.close()


def test_compiler_sees_recent_cued_and_due_lines(tmp_path):
    """Dormant lines are not shown by default but come back when new messages
    recall them or when their stated time arrives (message time, not wall clock)."""
    rt, backend = runtime(tmp_path, line_policy={"recent": 1, "cued": 1, "due": 1})
    try:
        add(rt, "r1", "Trip to Portugal next year || track:Portugal trip|Plans to visit Lisbon")
        add(rt, "r2", "Flight to Tokyo || track:Tokyo flight|Flight at 2023-07-27 15:00",
            start=1_690_100_000_000)
        add(rt, "r3", "New laptop maybe || track:Laptop purchase|Comparing models",
            start=1_690_200_000_000)
        add(rt, "r4", "New phone maybe || track:Phone purchase|Comparing phones",
            start=1_690_250_000_000)

        def shown(text, start):
            job = rt.prepare_add({"request_id": f"probe-{start}", "user_id": "u1",
                                  "session_id": "s1",
                                  "messages": [{"role": "user", "content": text,
                                                "timestamp": start}]})
            return {line["question"]: line for line in job.known_lines}

        lines = shown("Small talk about the weather", 1_690_260_000_000)  # 2023-07-25 04:40
        assert lines["Phone purchase"]["state"] == "open"  # most recent, its own line
        assert "Portugal trip" not in lines  # dormant
        assert "Tokyo flight" not in lines  # its time has not come yet

        lines = shown("Booked a hotel in Portugal for the trip", 1_690_270_000_000)
        assert "Portugal trip" in lines and lines["Portugal trip"]["state"] == "open"
        assert lines["Portugal trip"]["summary"] == "Plans to visit Lisbon"

        lines = shown("Small talk again", 1_690_470_000_000)  # 2023-07-27 15:20
        assert "Tokyo flight" in lines  # due: the plan's time has come
    finally:
        rt.close()
