"""Existing graph writer -> shared read-only neighbourhood -> AML Search."""

from datetime import UTC, datetime, timedelta

import pytest

from mr_mem.integrations.line_read import LineContextReader
from mr_mem.memory.retrieval import MemorySurfaceBudget
from mr_mem_aml.compiler import CompiledBlock, LineOut
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime
from tests.test_lce_path_a_tree import A, Clock, Env, at
from tests.test_native_semantics import SCOPE

KNOWN = datetime(2026, 10, 2, tzinfo=UTC)


@pytest.fixture
def env(tmp_path):
    value = Env(tmp_path)
    yield value
    value.close()


def graph_read(env, seed, **kwargs):
    options = dict(
        scope=SCOPE, ranks={env.memory_of[seed]: 1}, store=env.core.canonical,
        knowledge_at=KNOWN, now=at(100), budget=MemorySurfaceBudget(),
    )
    options.update(kwargs)
    return LineContextReader(env.tree.lines.path.parent.parent).read(**options)


def test_graph_neighbourhood_keeps_branches_provenance_and_read_only_storage(env):
    env.step(0, ("Considering a laptop", [A(happened=False)]),
             ("Repair succeeded", [A(after="b1")]),
             ("Prices increased", [A(after="b1")]))
    before = {p: p.read_bytes() for p in env.tree.lines.path.parent.glob("*.sqlite")}
    result = graph_read(env, "Considering a laptop")
    assert len(result) == 1
    group = result[0]
    assert all(text in group.content for text in env.memory_of)
    assert group.content.count(" -> ") == 2
    assert set(group.memory_ids) == set(env.memory_of.values())
    assert len(group.source_keys) == 3
    assert "in progress" in group.content  # idle time is not abandonment
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("boundary", ["scope", "knowledge", "hidden", "budget"])
def test_graph_expansion_respects_each_boundary(env, boundary):
    env.step(0, ("Considering a laptop", [A(happened=False)]),
             ("Repair succeeded", [A(after="b1")]))
    options = {}
    if boundary == "scope":
        from dataclasses import replace
        options["scope"] = replace(SCOPE, user_id="someone-else")
    elif boundary == "knowledge":
        options["knowledge_at"] = at(1)  # source existed; canonical not yet known
    elif boundary == "hidden":
        options["allowed"] = lambda m: m.content != "Repair succeeded"
    else:
        options["budget"] = MemorySurfaceBudget(max_characters=10)
    result = graph_read(env, "Considering a laptop", **options)
    if boundary == "hidden":
        assert "Repair succeeded" not in result[0].content
        assert " -> " not in result[0].content
        assert "partial" in result[0].content
    else:
        assert result == ()


@pytest.mark.parametrize("boundary", ["hidden", "future"])
def test_settlement_is_not_known_until_the_settling_block_is_available(env, monkeypatch, boundary):
    from types import SimpleNamespace

    env.step(0, ("Considering a laptop", [A(happened=False)]))
    if boundary == "future":
        monkeypatch.setattr(Clock, "now", lambda self: KNOWN + timedelta(days=1))
    env.known = SimpleNamespace(refs={"N1": env.node("Considering a laptop")})
    env.step(1, ("Repair succeeded", [A(after="N1")]))
    env.tree.tense.settle(env.memory_of["Considering a laptop"], at=at(1),
                          by=env.memory_of["Repair succeeded"])
    hidden = graph_read(env, "Considering a laptop", allowed=lambda m:
                        boundary != "hidden" or m.content != "Repair succeeded")
    assert "in progress" in hidden[0].content
    later = graph_read(env, "Considering a laptop", knowledge_at=KNOWN + timedelta(days=2))
    assert "in progress" not in later[0].content


def test_missing_graph_is_empty_but_partial_storage_is_an_error(env, tmp_path):
    import sqlite3

    from mr_mem.lce.path_a import scope_root

    seed = env.block("Considering a laptop", 0)
    root = tmp_path / "never-created"
    reader = LineContextReader(root)
    options = dict(scope=SCOPE, ranks={seed.memory_id: 1}, store=env.core.canonical,
                   knowledge_at=KNOWN, now=at(1), budget=MemorySurfaceBudget())
    assert reader.read(**options) == ()
    assert not root.exists()
    base = scope_root(root, SCOPE)
    base.mkdir(parents=True)
    (base / "lines.sqlite").write_bytes(env.tree.lines.path.read_bytes())
    with pytest.raises(sqlite3.OperationalError):
        reader.read(**options)
    assert not (base / "tense.sqlite").exists()


def test_bounded_neighbourhood_does_not_invent_an_edge_across_omitted_steps(env):
    env.step(0, ("First step", [A()]), ("Second step", [A(after="b1")]),
             ("Third step", [A(after="b2")]))
    reader = LineContextReader(env.tree.lines.path.parent.parent, max_nodes=2)
    result = reader.read(scope=SCOPE, ranks={env.memory_of["First step"]: 1},
                         store=env.core.canonical, knowledge_at=KNOWN, now=at(1),
                         budget=MemorySurfaceBudget())
    assert "partial" in result[0].content
    assert "Third step" not in result[0].content
    assert result[0].content.count(" -> ") == 1


class AcceptedGraphCompiler:
    """Explicit fixture edges. No model or semantic-quality claim."""

    version = "graph-fixture-v1"

    def compile(self, messages, context, known_lines, **kwargs):
        return tuple(CompiledBlock(
            m.content, (m.record_id,), None, None,
            (LineOut(None, None if i == 0 else f"b{i}", False, None),),
        ) for i, m in enumerate(messages))


@pytest.mark.parametrize("raw_base", [False, True])
def test_aml_tree_search_after_restart_preserves_baseline(tmp_path, raw_base):
    config = RuntimeConfig(data_dir=tmp_path, compiler={"type": "disabled"},
                           path_a_engine="tree", placement="points", raw_base=raw_base)
    payload = {"request_id": "r", "user_id": "u", "session_id": "s", "messages": [
        {"role": "user", "content": "Considering a laptop", "timestamp": 1690000000000},
        {"role": "user", "content": "Repair succeeded; purchase cancelled",
         "timestamp": 1690000060000},
    ]}
    rt = MemoryRuntime(config, compiler=AcceptedGraphCompiler())
    try:
        rt.add(payload)
    finally:
        rt.close()
    rt = MemoryRuntime(config)  # reading must not need a configured compiler
    try:
        q = {"query": "laptop", "user_id": "u", "top_k": 1}
        group = rt.search(q)["data"][0]
        assert group["id"].startswith("line:")
        assert "Repair succeeded; purchase cancelled" in group["content"]
        assert "Recorded progression:" in group["content"]
        assert not rt.search(q, mode="baseline")["data"][0]["id"].startswith("line:")
        assert rt.search({**q, "user_id": "other"}) == {"data": []}
    finally:
        rt.close()
