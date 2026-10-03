"""Public projection authority and privacy over real committed SQLite state."""

from dataclasses import FrozenInstanceError, fields, replace
from datetime import timedelta

import pytest

from mr_mem import MemoryCore, MemoryLifecycle
from mr_mem.memory.semantic_projection import SqliteCanonicalSemanticBlockReader
from tests.test_semantic_admission import NOW, Sources, service
from tests.test_semantic_validator import delta, edge, point


def test_readonly_block_hides_closure_and_preserves_dual_time(tmp_path):
    path = tmp_path / "canonical.sqlite"
    sources = Sources()
    binding = sources.bind("cohabit")
    ref = replace(binding.source_ref, occurred_at=NOW - timedelta(days=10))
    sources.refs["cohabit"] = ref
    binding = replace(binding, source_ref=ref)
    payload = delta()
    payload["points"] = [point("p1", "条件"), point("p2", "结论")]
    payload["dependencies"] = [edge(from_point_id="p2", target_kind="point", target_id="p1",
                                    boundary_policy="cohabit", lifecycle_effect="none")]
    with MemoryCore(path) as core:
        receipt = service(core, sources).admit_semantic_delta(
            payload, binding=binding, activated_memory_ids=(),
        )
        assert len(receipt.memory_ids) == 1
        reader = SqliteCanonicalSemanticBlockReader(path)
        try:
            view = reader.get_semantic_block_view(receipt.memory_ids[0])
            canonical = core.get(view.memory_id)
            assert view.content == "条件\n结论"
            assert view.occurred_at == canonical.occurred_at == ref.occurred_at
            assert view.known_at == canonical.known_at == NOW
            assert view.source_refs == (ref,)
            assert view.lifecycle is MemoryLifecycle.ACTIVE
            assert view.transition_known_at is None
            assert not {"member_point_ids", "points", "dependencies"} & {
                f.name for f in fields(view)
            }
            assert not hasattr(reader, "projection_queue")
            assert not hasattr(reader, "_commit")
            with pytest.raises(FrozenInstanceError):
                view.content = "overwrite"
            assert reader.get_semantic_block_view("missing") is None
        finally:
            reader.close()


def test_canonical_incoming_outgoing_and_supersede_authority(tmp_path):
    path = tmp_path / "canonical.sqlite"
    sources = Sources()
    with MemoryCore(path) as core:
        admission = service(core, sources)
        m1 = admission.admit_semantic_delta(
            delta(), binding=sources.bind("first"), activated_memory_ids=(),
        ).memory_ids[0]
        payload = delta()
        payload["dependencies"] = [edge(target_id=m1)]
        m2 = admission.admit_semantic_delta(
            payload, binding=sources.bind("second"), activated_memory_ids=(m1,),
        ).memory_ids[0]
        reader = SqliteCanonicalSemanticBlockReader(path)
        try:
            old, new = (reader.get_semantic_block_view(mid) for mid in (m1, m2))
            assert old.lifecycle is MemoryLifecycle.SUPERSEDED
            assert new.lifecycle is MemoryLifecycle.ACTIVE
            assert old.transition_known_at == new.known_at
            assert old.incoming_relations == new.outgoing_relations
            assert new.context_memory_ids == (m1,)
            relation = new.outgoing_relations[0]
            assert (relation.from_memory_id, relation.to_memory_id) == (m2, m1)
            assert relation.relation == "supersedes"
            for lifecycle in (MemoryLifecycle.INVALIDATED, MemoryLifecycle.ARCHIVED):
                with core.canonical._transaction():
                    core.canonical._set_lifecycle(core.get(m2), lifecycle)
                view = reader.get_semantic_block_view(m2)
                assert view.lifecycle is lifecycle
                assert view.transition_known_at is None
        finally:
            reader.close()
