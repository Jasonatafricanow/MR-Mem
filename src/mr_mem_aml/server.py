"""AML-style HTTP surface: POST /add, POST /search, POST /admin/flush, GET /health.

Concurrency model:

- One owner thread holds the runtime: every store (canonical SQLite, raw source,
  LCE, Qdrant) is only ever touched from that thread, in arrival order.
- HTTP requests are served by a thread pool. Add is split in three phases:
  prepare (owner) -> model compile (request thread, in parallel) -> finish (owner).
  The model call dominates Add latency, so different users compile concurrently,
  bounded by ``compile_concurrency``.
- Requests of one ``user_id`` are serialized end to end, so each chunk compiles
  with the context and known lines its predecessors produced.

``X-MRMem-Search-Mode: baseline|path_a`` overrides the configured read mode per
request, so one compiled dataset can be evaluated in both modes.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import (
    CompileFailed,
    ConflictingReplay,
    InvalidRequest,
    MemoryRuntime,
    RetryableError,
)

log = logging.getLogger("mr_mem_aml.http")
MAX_BODY = 64 * 1024 * 1024


class RuntimeHost:
    """Owns a MemoryRuntime on one dedicated thread and serializes store access."""

    def __init__(self, factory) -> None:
        self._owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mr-mem-owner")
        self.runtime: MemoryRuntime = self._owner.submit(factory).result()
        concurrency = getattr(self.runtime.config, "compile_concurrency", 1)
        self._compile_slots = threading.Semaphore(max(1, concurrency))
        self._user_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        # background compile (config background_compile): Add returns once the raw text is
        # stored and searchable; a user's due windows compile here, one user at a time per user
        self._background = getattr(self.runtime.config, "background_compile", None)
        self._bg: dict[str, dict] = {}
        self._closing = False
        self._bg_pool = (
            ThreadPoolExecutor(max(2, 2 * concurrency), thread_name_prefix="mr-mem-compile")
            if self._background
            else None
        )
        if self._background and self.runtime.config.compile_window is not None:
            # after a restart nothing would kick a user whose Adds are all in: what was left to
            # compile waited for that user's first Search, which then found only raw hits
            for user in self.call(self.runtime.source.users_with_pending):
                self._kick(user)

    def call(self, fn, *args, **kwargs):
        return self._owner.submit(fn, *args, **kwargs).result()

    def _user_lock(self, payload: object) -> threading.Lock:
        user = str(payload.get("user_id")) if isinstance(payload, dict) else ""
        with self._guard:
            return self._user_locks.setdefault(user, threading.Lock())

    def add(self, payload: object) -> dict:
        runtime = self.runtime
        if self._background and runtime.config.compile_window is not None:
            # stored and searchable before this returns (AML's Add contract); the compile of the
            # windows now due goes on in the background (a 20-second gateway cut synchronous Adds)
            job = self.call(runtime.prepare_add, payload)
            request = job if isinstance(job, dict) else job.request
            if isinstance(payload, dict) and isinstance(payload.get("user_id"), str):
                self._kick(payload["user_id"])
                self._backpressure(payload["user_id"])
            if isinstance(job, dict):
                return job
            from mr_mem_aml.runtime import _echo

            return {**_echo(request), "blocks": 0}
        with self._user_lock(payload):
            job = self.call(runtime.prepare_add, payload)
            if isinstance(job, dict):
                return job
            try:
                with self._compile_slots:
                    compiled = runtime.compile_job(job)
            except CompileFailed as exc:
                self.call(runtime.fail_add, job, str(exc))
                raise
            return self._finish(job, compiled)

    def _finish(self, job, compiled) -> dict:
        """runtime.finish_steps: its store work on the owner thread, its model calls (step 2,
        threads, summaries) off it, each under a compile slot, so users place in parallel and
        calls that need nothing of each other go together."""
        steps = self.runtime.finish_steps(job, compiled)
        sent, thrown = None, None
        while True:
            try:
                call = (
                    self.call(steps.throw, thrown)
                    if thrown is not None
                    else self.call(steps.send, sent)
                )
            except StopIteration as done:
                return done.value
            try:
                if isinstance(call, list):
                    with ThreadPoolExecutor(len(call)) as pool:
                        sent = [f.result() for f in [pool.submit(self._slotted, c) for c in call]]
                else:
                    sent = self._slotted(call)
                thrown = None
            except Exception as exc:  # noqa: BLE001 - handed back to the step that asked
                sent, thrown = None, exc

    def _slotted(self, call):
        with self._compile_slots:
            return call[0](*call[1:])

    def _backpressure(self, user: str) -> None:
        """background_compile.max_backlog: while more than this many characters of the user's
        history wait for a compile window, the Add (already stored) waits up to add_wait seconds
        and then answers 503; AML retries it (Add: up to 32 attempts), the retry replays as done.
        Ingestion so keeps the pace of compiling, as when Add compiled itself: otherwise a long
        history was all written in minutes, and its Searches came while most of it was still
        uncompiled (a window takes ~20 s; a 1M-token history has ~60)."""
        limit = (self._background or {}).get("max_backlog")
        if not limit:
            return
        # in messages as well: short messages close windows at max_messages long before the
        # characters reach the limit
        cap = 2 * int((self.runtime.config.compile_window or {}).get("max_messages") or 10**9)
        deadline = time.monotonic() + float(self._background.get("add_wait", 12))

        def over() -> bool:
            chars, count = self.call(self.runtime.source.pending_size, user)
            return chars > limit or count > cap

        while over():
            if time.monotonic() >= deadline:
                raise RetryableError("compiling this user's earlier history; retry the Add")
            time.sleep(0.5)

    def _warm(self, payload: object) -> None:
        """The query's vector, fetched on this request thread into the embedding cache before
        the Search runs on the owner thread (which then reads it locally): one embedding call
        on the owner thread held every other user's Search and Add behind it."""
        embedding = self.runtime.embedding
        if (
            embedding is None
            or not hasattr(embedding, "embed_many")
            or not isinstance(payload, dict)
        ):
            return
        query, options = payload.get("query"), payload.get("options")
        if not isinstance(query, str) or not query.strip():
            return
        text = query.strip()
        if self.runtime.config.append_options and isinstance(options, list) and options:
            text = text + "\n" + "\n".join(str(o) for o in options)
        try:
            embedding.embed_many([text[:4096]])
        except Exception as exc:  # noqa: BLE001 - the Search embeds it itself
            log.warning("query vector not prefetched: %s", type(exc).__name__)

    def _kick(self, user: str, final: bool = False) -> threading.Event:
        """Have the user's due windows compiled in the background (``final``: the tail too, as
        flush does); returns the event set when the user's compiling is idle again. An Add with
        no Add after it for ``idle_tail`` seconds has the tail compiled, so a Search seldom
        finds anything left to compile."""
        with self._guard:
            if self._closing:  # nothing new is scheduled once the host is closing
                done = threading.Event()
                done.set()
                return done
            state = self._bg.setdefault(
                user,
                {
                    "running": False,
                    "again": False,
                    "final": False,
                    "idle": threading.Event(),
                    "timer": None,
                },
            )
            if state["timer"] is not None and not final:
                state["timer"].cancel()
                state["timer"] = None
            state["final"] = state["final"] or final
            if state["running"]:
                state["again"] = True
                return state["idle"]
            state["running"] = True
            state["idle"].clear()
        self._bg_pool.submit(self._drain, user)
        return state["idle"]

    def _drain(self, user: str) -> None:
        runtime, state = self.runtime, self._bg[user]
        while True:
            with self._guard:
                final, state["final"], state["again"] = state["final"], False, False
            try:
                with self._user_lock({"user_id": user}):
                    while (job := self.call(runtime._window_job, user, final=final)) is not None:
                        with self._compile_slots:
                            compiled = runtime.compile_job(job)
                        self._finish(job, compiled)
            except Exception as exc:  # noqa: BLE001 - raw stays searchable; the next kick retries
                runtime.window_failures += 1
                log.warning("background compile of %s stopped: %s", user, str(exc)[:300])
            with self._guard:
                if state["again"] or state["final"]:
                    continue
                state["running"] = False
                state["idle"].set()
                if not final and not self._closing:  # no Add for a while: compile the tail
                    timer = threading.Timer(
                        float(self._background.get("idle_tail", 30)), self._kick, (user, True)
                    )
                    timer.daemon = True
                    state["timer"] = timer
                    timer.start()
                return

    def search(self, payload: object, mode: str | None) -> dict:
        runtime = self.runtime
        if (
            runtime.config.compile_before_search
            and runtime.config.compile_window is not None
            and isinstance(payload, dict)
            and isinstance(payload.get("user_id"), str)
        ):
            if self._background:  # compile what is left, but answer within search_wait
                idle = self._kick(payload["user_id"], final=True)
                idle.wait(float(self._background.get("search_wait", 15)))
            else:
                self.flush(payload["user_id"])  # nothing to do once the history is compiled
        self._warm(payload)
        found = self.call(runtime.search, payload, mode=mode)
        if runtime.decision is not None and (mode or runtime.config.search_mode) == "path_a":
            # the decision model's calls off the owner thread; the hits read on it
            question, hits, lang = self.call(runtime.decision_hits, payload)
            note = runtime.decision.note(question, hits, lang)
            if note:
                from mr_mem_aml.runtime import _within

                first = found["data"][0]["score"] if found["data"] else 0.0
                found["data"] = _within(  # within top_k, the furthest stretch summary out first
                    [
                        {"id": "decision-note", "content": note, "score": float(first) + 1.0},
                        *found["data"],
                    ],
                    payload.get("top_k", 100),
                )
        return found

    def flush(self, user_id: str | None = None) -> dict:
        """Compile what is left of one user's history (or every user's), tail included: as Add does,
        under that user's lock with the model call off the owner thread, so a user still adding
        is never compiled from two threads at once."""
        runtime = self.runtime
        if runtime.config.compile_window is None:
            return self.call(runtime.flush)
        users = [user_id] if user_id else list(self.call(runtime.source.users_with_pending))
        windows = 0
        for user in users:
            with self._user_lock({"user_id": user}):
                while (job := self.call(runtime._window_job, user, final=True)) is not None:
                    with self._compile_slots:
                        compiled = runtime.compile_job(job)
                    self._finish(job, compiled)
                    windows += 1
        return {"status": "ok", "windows": windows}

    def health(self) -> dict:
        runtime = self.runtime
        config = runtime.config
        return {
            "status": "ok",
            "search_mode": config.search_mode,
            "lce_enabled": config.lce_enabled,
            "dense": runtime.index is not None,
            "hyde": config.hyde,
            "compile_concurrency": config.compile_concurrency,
            "thread_failures": runtime.thread_failures,
            # compiles that failed after their retries (e.g. output cut at max_tokens) and kept
            # only their raw text, and those the endpoint refused: 0 window_failures alone hid them
            "compile_failed_kept_raw": runtime.compile_failed_kept_raw,
            "compile_rejected": runtime.compile_rejected,
            "compile_capped": runtime.compile_capped,
            "decision": runtime.decision.health() if runtime.decision else {"configured": False},
            "background_compile": (
                {
                    "users_compiling": sum(s["running"] for s in list(self._bg.values())),
                    "window_failures": runtime.window_failures,
                }
                if self._background
                else {"configured": False}
            ),
        }

    def close(self) -> None:
        if self._bg_pool is not None:
            with self._guard:
                self._closing = True
                for state in self._bg.values():
                    if state["timer"] is not None:
                        state["timer"].cancel()
            # a window cut short recompiles from its journal at the next start
            self._bg_pool.shutdown(wait=False, cancel_futures=True)
        try:
            self.call(self.runtime.close)
        finally:
            self._owner.shutdown(wait=True)


def make_handler(host: RuntimeHost, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # route through logging
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: dict, headers: dict | None = None) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if token is None:
                return True
            # AML auth schemes: "Bearer <t>", "Token <t>", or "X-Api-Key: <t>".
            header = self.headers.get("Authorization", "")
            scheme, _, value = header.partition(" ")
            if scheme.lower() in ("bearer", "token") and hmac.compare_digest(value.strip(), token):
                return True
            return hmac.compare_digest(self.headers.get("X-Api-Key", ""), token)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") != "/health":
                return self._send(404, {"error": "not found"})
            self._send(200, host.health())

        def do_POST(self):  # noqa: N802
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._send(400, {"error": "missing or oversized body"})
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._send(400, {"error": "invalid JSON"})
            route = self.path.rstrip("/")
            started = time.monotonic()
            user = payload.get("user_id") if isinstance(payload, dict) else None

            def done(body: dict) -> None:  # one line per request: who, how long, what came back
                kinds = ""
                if route == "/search":
                    count: dict[str, int] = {}
                    for item in body.get("data", []):
                        i = str(item.get("id", ""))
                        k = (
                            "thread"
                            if i.startswith("thread:")
                            else "raw"
                            if i.startswith("raw:")
                            else i
                            if i in ("standing", "decision-note")
                            else "line"
                        )
                        count[k] = count.get(k, 0) + 1
                    kinds = " " + " ".join(f"{k}={v}" for k, v in sorted(count.items()))
                log.info(
                    "%s user=%s status=200 ms=%d%s",
                    route,
                    user,
                    (time.monotonic() - started) * 1000,
                    kinds,
                )
                self._send(200, body)

            try:
                if route == "/add":
                    return done(host.add(payload))
                if route == "/search":
                    return done(host.search(payload, self.headers.get("X-MRMem-Search-Mode")))
                if route == "/admin/flush":
                    return self._send(200, host.flush())
                return self._send(404, {"error": "not found"})
            except ConflictingReplay as exc:
                # 409 is retried by AML; a payload conflict is permanent.
                return self._send(422, {"error": str(exc)})
            except InvalidRequest as exc:
                return self._send(400, {"error": str(exc)})
            except RetryableError as exc:
                log.warning(
                    "%s user=%s status=503 ms=%d: %s",
                    route,
                    user,
                    (time.monotonic() - started) * 1000,
                    exc,
                )
                return self._send(503, {"error": str(exc)}, {"Retry-After": "10"})
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                log.exception("unhandled failure on %s", route)
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def build_server(config: RuntimeConfig, host: RuntimeHost) -> _Server:
    token = os.environ.get(config.auth_token_env) if config.auth_token_env else None
    if config.auth_token_env and not (token and token.strip()):
        raise ValueError("configured authentication environment variable is missing or empty")
    return _Server((config.host, config.port), make_handler(host, token))


def serve(config: RuntimeConfig) -> None:
    host = RuntimeHost(lambda: MemoryRuntime(config))
    server = build_server(config, host)
    log.info(
        "MR-Mem Path A listening on http://%s:%s (mode=%s, lce=%s, compile_concurrency=%s)",
        config.host,
        config.port,
        config.search_mode,
        config.lce_enabled,
        config.compile_concurrency,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        host.close()
