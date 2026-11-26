"""Offline Add/Search smoke test: synthetic input, temporary storage, no provider calls."""

import json
import os
import re
import secrets
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from mr_mem_aml.compiler import SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime
from mr_mem_aml.server import RuntimeHost, build_server

_NEW = re.compile(r"\nNEW[:：]\n")
_SBS = re.compile(r"\nSBS[:：]\n")
_MESSAGE = re.compile(r"^\[M(\d+)[^\]]*?\s(\S+)\] ?(.*)$")


class DemoCompiler:
    """Deterministic protocol fixture, not a semantic model or quality benchmark: it speaks the
    one compile path (docs/LCE_FROZEN.md 2) in the chat scene, step 1 one SB per message, the
    stretch layer one open stretch over the window's SBs."""

    def complete(self, prompt: str) -> str:
        if _SBS.search(prompt):
            body = _SBS.split(prompt, maxsplit=1)[1]
            opened = len(re.findall(r"^o\d+\b", body, re.M))
            sbs = len(re.findall(r"^s\d+\b", body, re.M))
            return json.dumps({"stretches": [{"sbs": f"{'o1' if opened else 's1'}-s{sbs}",
                                              "from": None, "closed": False}], "settles": {}})
        messages = [m for line in _NEW.split(prompt, maxsplit=1)[1].split("\n")
                    if (m := _MESSAGE.match(line))]
        sbs = [{"turns": f"M{m.group(1)}", "text": m.group(3), "when": "before",
                "status": "confirmed", "standing": False} for m in messages]
        return "JSON:\n" + json.dumps(sbs) + f"\nSTOP: M{messages[-1].group(1)}"


@contextmanager
def running(config, token):
    host = RuntimeHost(lambda: MemoryRuntime(
        config, compiler=SemanticCompiler(DemoCompiler())))
    server = build_server(config, host)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(route, payload, scheme="Bearer"):
        headers = {"Content-Type": "application/json"}
        if scheme == "X-Api-Key":
            headers[scheme] = token
        elif scheme:
            headers["Authorization"] = f"{scheme} {token}"
        request = urllib.request.Request(
            base + route, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            assert response.status == 200  # public health, no model calls
        yield post
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        host.close()


def main() -> None:
    token = secrets.token_urlsafe(24)
    env_name = "MRMEM_OFFLINE_SMOKE_TOKEN"
    previous = os.environ.get(env_name)
    os.environ[env_name] = token
    try:
        with tempfile.TemporaryDirectory(prefix="mrmem-smoke-") as tmp:
            config = RuntimeConfig(
                data_dir=Path(tmp), compiler={"type": "command", "argv": ["unused"]},
                host="127.0.0.1", port=0,
                raw_base=True, line_read={"raw_fill": 0, "hit_anchors": True},
                compile_before_search=True,  # Search compiles the open tail: the compile path runs
                auth_token_env=env_name,
            )
            payload = {
                "request_id": "demo-request", "user_id": "demo-user", "session_id": "demo",
                "messages": [{"role": "user", "content": "The demo garden has blue flowers.",
                              "timestamp": 1704067200000}],
            }
            with running(config, token) as post:
                try:
                    post("/add", payload, scheme=None)
                except urllib.error.HTTPError as exc:
                    assert exc.code == 401
                else:
                    raise AssertionError("Unauthenticated Add was accepted")
                for scheme in ("Bearer", "Token"):
                    added = post("/add", payload, scheme=scheme)
                    assert added["success"] is True
                    assert all(added[k] == payload[k]
                               for k in ("request_id", "user_id", "session_id"))
                result = post("/search", {"user_id": "demo-user", "query": "blue flowers",
                                          "top_k": 1}, scheme="X-Api-Key")
                assert len(result["data"]) == 1
                assert all(i["id"] and i["content"] for i in result["data"])
                assert post("/search", {"user_id": "other-user", "query": "flowers"})["data"] == []
            with running(config, token) as post:
                result = post("/search", {"user_id": "demo-user", "query": "blue flowers"})
                assert any("blue flowers" in item["content"] for item in result["data"])
            print("OFFLINE SMOKE OK: AML envelope, auth, replay, isolation, restart; "
                  "no model calls.")
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


if __name__ == "__main__":
    main()
