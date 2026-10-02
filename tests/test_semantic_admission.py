"""Real SQLite source, lifecycle, atomic mutation, and replay invariants."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mr_mem import (
    MemoryCore,
    MemoryLifecycle,
    Scope,
    ScopeDomain,
    SemanticAdmissionService,
    SemanticMemoryCandidate,
    SourceRef,
)
from mr_mem.memory.semantic_contracts import SemanticDeltaError
from mr_mem.memory.semantic_store import SemanticSourceBinding
from tests.test_semantic_validator import delta, edge, point

NOW = datetime(2026, 10, 2, tzinfo=UTC)
SCOPE = Scope(ScopeDomain.USER, user_id="u1")


class Sources:
    def __init__(self):
        self.refs = {}

    def bind(self, interaction_id):
        ref = SourceRef("native-test", "session", interaction_id, NOW, revision="1")
        self.refs[interaction_id] = ref
        return SemanticSourceBinding(SCOPE, interaction_id, ref)

    def current_ref(self, scope, ref):
        return self.refs.get(ref.record_id) if scope == SCOPE else None

    def current_user_source(self, scope, interaction_id):
        return self.refs.get(interaction_id) if scope == SCOPE else None


class Clock:
    def now(self):
        return NOW


def service(core, sources):
    return SemanticAdmissionService(
        store=core.canonical, sources=sources, clock=Clock(), origin_runtime_id="test-host"
    )


def seed(admission, binding, content="旧 top-k=100"):
    return admission.admit(
        SemanticMemoryCandidate(
            semantic_id=binding.interaction_id,
            scope=SCOPE,
            content=content,
            source_refs=(binding.source_ref,),
            compiler_version="seed-v1",
        )
    )


def test_native_binding_multi_block_context_and_partial_correction(tmp_path):
    sources = Sources()
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        admission = service(core, sources)
        budget = seed(admission, sources.bind("budget"), "预算80万")
        suv = seed(admission, sources.bind("suv"), "车型SUV")
        old = seed(admission, sources.bind("old"), "地点北京")
        binding = sources.bind("current")
        payload = delta()
        payload["points"] = [point("p1", "用户将地点改为上海。"), point("p2", "测试已通过。")]
        payload["dependencies"] = [edge(target_id=old.memory_id)]
        receipt = admission.admit_semantic_delta(
            payload,
            binding=binding,
            activated_memory_ids=(budget.memory_id, suv.memory_id, old.memory_id),
        )
        assert len(receipt.memory_ids) == 2
        assert core.get(old.memory_id).lifecycle is MemoryLifecycle.SUPERSEDED
        assert core.get(budget.memory_id) == budget and core.get(suv.memory_id) == suv
        memories = [core.get(mid) for mid in receipt.memory_ids]
        assert all("地点北京" not in memory.content for memory in memories)
        assert all(memory.provenance.source_refs == (binding.source_ref,) for memory in memories)
        meta = core.canonical.get_semantic_metadata(receipt.memory_ids[0])
        assert meta.context_memory_ids == (old.memory_id,)
        assert meta.source_interaction_id == "current"
        assert core.canonical.semantic_relations(receipt.memory_ids[0]) == (
            (receipt.memory_ids[0], old.memory_id, "supersedes", "context", "supersede", "current"),
        )
        assert (
            admission.admit_semantic_delta(
                payload,
                binding=binding,
                activated_memory_ids=(budget.memory_id, suv.memory_id, old.memory_id),
            )
            == receipt
        )
        assert len(core.load_all()) == 5


@pytest.mark.parametrize("fault", ["metadata", "relation", "lifecycle"])
def test_atomic_failure_keeps_old_active_and_frozen_compilation(tmp_path, fault):
    sources = Sources()
    path = tmp_path / "memory.sqlite"
    with MemoryCore(path) as core:
        admission = service(core, sources)
        old = seed(admission, sources.bind("old"))
        binding = sources.bind("current")
        payload = delta()
        payload["points"].append(point("p2", "另一个独立新增。"))
        payload["dependencies"] = [edge(target_id=old.memory_id)]
        sql = {
            "metadata": "BEFORE INSERT ON semantic_block_metadata",
            "relation": "BEFORE INSERT ON semantic_relations",
            "lifecycle": "BEFORE UPDATE ON canonical_memory",
        }[fault]
        with sqlite3.connect(path) as db:
            db.execute(
                f"CREATE TRIGGER fail_semantic {sql} BEGIN SELECT RAISE(ABORT,'injected'); END"
            )
        with pytest.raises(SemanticDeltaError, match="SEMANTIC_DELTA_COMMIT_FAILED"):
            admission.admit_semantic_delta(
                payload, binding=binding, activated_memory_ids=(old.memory_id,)
            )
        assert core.load_all() == (old,)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT status FROM semantic_compilations").fetchall() == [
                ("frozen",)
            ]
            assert db.execute("SELECT count(*) FROM semantic_block_metadata").fetchone() == (0,)
            assert db.execute("SELECT count(*) FROM semantic_relations").fetchone() == (0,)
            db.execute("DROP TRIGGER fail_semantic")
        receipt = admission.admit_semantic_delta(
            payload, binding=binding, activated_memory_ids=(old.memory_id,)
        )
        assert len(receipt.memory_ids) == 2
        assert core.get(old.memory_id).lifecycle is MemoryLifecycle.SUPERSEDED


@pytest.mark.parametrize("mutation", ["revision", "interaction", "scope"])
def test_system_bound_source_is_revalidated_and_cannot_be_forged(tmp_path, mutation):
    sources = Sources()
    binding = sources.bind("current")
    if mutation == "revision":
        sources.refs["current"] = replace(binding.source_ref, revision="2")
    elif mutation == "interaction":
        binding = replace(binding, interaction_id="other")
    else:
        binding = replace(binding, scope=Scope(ScopeDomain.USER, user_id="another"))
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        with pytest.raises(SemanticDeltaError):
            service(core, sources).admit_semantic_delta(
                delta(), binding=binding, activated_memory_ids=()
            )
        assert core.load_all() == ()


def test_defer_is_audited_but_has_no_memory_or_projection(tmp_path):
    sources = Sources()
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        payload = delta()
        payload["points"][0].update(status="defer", unresolved_refs=["那个"])
        receipt = service(core, sources).admit_semantic_delta(
            payload, binding=sources.bind("current"), activated_memory_ids=()
        )
        assert receipt.memory_ids == () and receipt.status == "deferred"
        assert core.load_all() == ()
        assert core.canonical.projection_queue().pending(10) == ()
