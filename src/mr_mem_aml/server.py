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

    def call(self, fn, *args, **kwargs):
        return self._owner.submit(fn, *args, **kwargs).result()

    def _user_lock(self, payload: object) -> threading.Lock:
        user = str(payload.get("user_id")) if isinstance(payload, dict) else ""
        with self._guard:
            return self._user_locks.setdefault(user, threading.Lock())

    def add(self, payload: object) -> dict:
        runtime = self.runtime
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
            return self.call(runtime.finish_add, job, compiled)

    def search(self, payload: object, mode: str | None) -> dict:
        return self.call(self.runtime.search, payload, mode=mode)

    def flush(self) -> dict:
        return self.call(self.runtime.flush)

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
        }

    def close(self) -> None:
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
            try:
                if route == "/add":
                    return self._send(200, host.add(payload))
                if route == "/search":
                    return self._send(
                        200, host.search(payload, self.headers.get("X-MRMem-Search-Mode"))
                    )
                if route == "/admin/flush":
                    return self._send(200, host.flush())
                return self._send(404, {"error": "not found"})
            except ConflictingReplay as exc:
                # 409 is retried by AML; a payload conflict is permanent.
                return self._send(422, {"error": str(exc)})
            except InvalidRequest as exc:
                return self._send(400, {"error": str(exc)})
            except RetryableError as exc:
                log.warning("retryable failure on %s: %s", route, exc)
                return self._send(503, {"error": str(exc)}, {"Retry-After": "30"})
            except Exception as exc:  # noqa: BLE001 - HTTP boundary
                log.exception("unhandled failure on %s", route)
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def build_server(config: RuntimeConfig, host: RuntimeHost) -> _Server:
    token = os.environ.get(config.auth_token_env) if config.auth_token_env else None
    return _Server((config.host, config.port), make_handler(host, token))


def serve(config: RuntimeConfig) -> None:
    host = RuntimeHost(lambda: MemoryRuntime(config))
    server = build_server(config, host)
    log.info(
        "MR-Mem Path A listening on http://%s:%s (mode=%s, lce=%s, compile_concurrency=%s)",
        config.host, config.port, config.search_mode, config.lce_enabled,
        config.compile_concurrency,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        host.close()
