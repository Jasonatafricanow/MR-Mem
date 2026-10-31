"""Offline Add/Search smoke test: synthetic input, temporary storage, no provider calls."""

import json
import tempfile
import threading
import urllib.request
from pathlib import Path

from mr_mem_aml.compiler import SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime
from mr_mem_aml.server import RuntimeHost, build_server


class DemoCompiler:
    """Deterministic protocol fixture, not a semantic model or quality benchmark."""

    def complete(self, prompt: str) -> str:
        material = json.loads(prompt.split("INPUT:\n", 1)[1])
        return json.dumps({
            "matters": [],
            "blocks": [{"content": m["text"], "sources": [m["id"]], "in": []}
                       for m in material["NEW"]],
            "carry_from": None,
        })


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mrmem-smoke-") as tmp:
        config = RuntimeConfig(
            data_dir=Path(tmp), compiler={"type": "command", "argv": ["unused"]},
            host="127.0.0.1", port=0, path_a_engine="tree", placement="points",
            raw_base=True, line_read={"raw_fill": 0, "hit_anchors": True},
        )
        host = RuntimeHost(lambda: MemoryRuntime(
            config, compiler=SemanticCompiler(DemoCompiler(), schema="tree")))
        server = build_server(config, host)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def post(route, payload):
            request = urllib.request.Request(
                base + route, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        try:
            added = post("/add", {
                "request_id": "demo-request", "user_id": "demo-user", "session_id": "demo",
                "messages": [{"role": "user", "content": "The demo garden has blue flowers.",
                              "timestamp": 1704067200000}],
            })
            assert added["success"]
            result = post("/search", {"user_id": "demo-user", "query": "blue flowers"})
            assert any("blue flowers" in item["content"] for item in result["data"])
            assert post("/search", {"user_id": "other-user", "query": "blue flowers"})["data"] == []
            print("OFFLINE SMOKE OK: Add, Search and user isolation; no model calls.")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            host.close()


if __name__ == "__main__":
    main()
