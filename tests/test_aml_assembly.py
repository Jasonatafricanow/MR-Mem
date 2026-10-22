"""Unfilled compiler port -> durable pending input -> restart and resume."""

from datetime import UTC, datetime

import pytest

from mr_mem_aml.compiler import CompiledBlock
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import CompileFailed, MemoryRuntime
from mr_mem_aml.server import RuntimeHost


class AcceptedFixtureCompiler:
    """Accepted test fixture, not a semantic algorithm or quality measurement."""

    version = "assembly-fixture-v1"

    def __init__(self):
        self.calls = 0

    def compile(self, messages, context, known_lines, *, window=False, **kwargs):
        self.calls += 1
        blocks = tuple(CompiledBlock(m.content, (m.record_id,), None, None) for m in messages)
        return (blocks, len(messages)) if window else blocks


@pytest.mark.parametrize("windowed", [False, True])
def test_pending_input_survives_restart_then_compiles_once(tmp_path, windowed):
    config = RuntimeConfig(
        data_dir=tmp_path, compiler={"type": "disabled"}, lce_enabled=False,
        raw_base=True, path_a_engine="tree" if windowed else "flat",
        compile_window={"soft": 2, "hard": 3} if windowed else None,
    )
    payload = {"request_id": "r1", "user_id": "u1", "session_id": "s1",
               "messages": [{"role": "user", "content": "I keep bees",
                             "timestamp": 1_690_000_000_000}]}
    host = RuntimeHost(lambda: MemoryRuntime(config))
    try:
        assert host.health()["status"] == "degraded"
        assert host.health()["compiler_configured"] is False
        for _ in range(2):  # At-least-once delivery before the algorithm is connected.
            with pytest.raises(CompileFailed, match="semantic_compiler_not_configured"):
                host.add(payload)
        with pytest.raises(CompileFailed):
            host.flush()
        assert host.call(host.runtime.core.load_all) == ()
        assert len(host.call(host.runtime.source.request_messages, "r1")) == 1
        found = host.search({"query": "bees", "user_id": "u1"}, "baseline")
        assert "bees" in found["data"][0]["content"]
        assert host.search({"query": "bees", "user_id": "u2"}, "baseline")["data"] == []
    finally:
        host.close()

    compiler = AcceptedFixtureCompiler()
    host = RuntimeHost(lambda: MemoryRuntime(config, compiler=compiler))
    try:
        assert host.health()["compiler_configured"] is True
        assert host.health()["semantic_validation"] == "not_checked"
        host.add(payload)
        if windowed:
            host.flush()
        assert host.add(payload)["replayed"] is True
        memories = host.call(host.runtime.core.load_all)
        assert len(memories) == compiler.calls == 1
        assert memories[0].occurred_at == datetime.fromtimestamp(1_690_000_000, UTC)
        assert len(host.call(host.runtime.source.request_messages, "r1")) == 1
    finally:
        host.close()


def test_disabled_compiler_rejects_hyde_dependency(tmp_path):
    with pytest.raises(ValueError, match="hyde"):
        RuntimeConfig(data_dir=tmp_path, compiler={"type": "disabled"}, hyde="fallback")
