"""AML-style HTTP surface: POST /add, POST /search, POST /admin/flush, GET /health.

Single-threaded on purpose: one writer owns the canonical database, and
requests are served in arrival order. ``X-MRMem-Search-Mode: baseline|path_a``
overrides the configured read mode per request, so one compiled dataset can be
evaluated in both modes.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from mr_mem_aml.runtime import (
    ConflictingReplay,
    InvalidRequest,
    MemoryRuntime,
    RetryableError,
)

log = logging.getLogger("mr_mem_aml.http")
MAX_BODY = 64 * 1024 * 1024


def make_handler(runtime: MemoryRuntime, token: str | None):
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
            config = runtime.config
            self._send(200, {
                "status": "ok",
                "search_mode": config.search_mode,
                "lce_enabled": config.lce_enabled,
                "dense": runtime.index is not None,
                "hyde": config.hyde,
                "thread_failures": runtime.thread_failures,
            })

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
                    return self._send(200, runtime.add(payload))
                if route == "/search":
                    mode = self.headers.get("X-MRMem-Search-Mode")
                    return self._send(200, runtime.search(payload, mode=mode))
                if route == "/admin/flush":
                    return self._send(200, runtime.flush())
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


class _QueuedHTTPServer(HTTPServer):
    # Concurrent AML callers wait in the listen backlog and are served in order.
    request_queue_size = 128


def serve(runtime: MemoryRuntime) -> None:
    config = runtime.config
    token = os.environ.get(config.auth_token_env) if config.auth_token_env else None
    server = _QueuedHTTPServer((config.host, config.port), make_handler(runtime, token))
    log.info("MR-Mem Path A listening on http://%s:%s (mode=%s, lce=%s)",
             config.host, config.port, config.search_mode, config.lce_enabled)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()
