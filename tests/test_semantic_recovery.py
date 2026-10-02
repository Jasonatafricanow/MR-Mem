"""Restart uses the accepted job, without semantic inference or re-compilation."""

import sqlite3
from dataclasses import replace

import pytest

from mr_mem import MemoryCore
from mr_mem.memory.projection import ProjectionWorker
from mr_mem.memory.semantic_contracts import SemanticDeltaError
from tests.test_semantic_admission import Clock, Sources, service
from tests.test_semantic_validator import delta, edge, point


@pytest.mark.parametrize("mode,stale", [("resume", False), ("batch", False), ("batch", True)])
def test_restart_after_freeze_uses_accepted_job_without_compiler(
    tmp_path,
    monkeypatch,
    mode,
    stale,
):
    path = tmp_path / "memory.sqlite"
    sources = Sources()
    binding = sources.bind("current")
    with MemoryCore(path) as core:
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TRIGGER interrupt_commit BEFORE INSERT ON semantic_block_metadata "
                "BEGIN SELECT RAISE(ABORT,'crash boundary'); END"
            )
        with pytest.raises(SemanticDeltaError):
            service(core, sources).admit_semantic_delta(
                delta(), binding=binding, activated_memory_ids=()
            )
        assert core.load_all() == ()
    with sqlite3.connect(path) as db:
        frozen = db.execute("SELECT accepted FROM semantic_compilations").fetchone()[0]
        db.execute("DROP TRIGGER interrupt_commit")
    if stale:
        sources.refs["current"] = replace(binding.source_ref, revision="2")

    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must never recompile accepted semantics")

    monkeypatch.setattr("mr_mem.memory.semantic_admission.compile_semantic_delta", forbidden)
    with MemoryCore(path) as core:
        admission = core.semantic_admission(
            sources=sources, clock=Clock(), origin_runtime_id="test-host"
        )
        if stale:
            with pytest.raises(SemanticDeltaError):
                admission.recover_semantic_deltas()
            assert core.load_all() == ()
        else:
            receipt = (
                admission.resume_semantic_delta(binding)
                if mode == "resume"
                else admission.recover_semantic_deltas()[0]
            )
            assert receipt == admission.semantic_delta_receipt(binding)
            assert receipt == admission.resume_semantic_delta(binding)
            assert len(core.load_all()) == 1
            assert core.get_semantic_metadata(receipt.memory_ids[0]).member_point_ids == ("p1",)
            assert admission.recover_semantic_deltas() == ()
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT accepted FROM semantic_compilations").fetchone()[0] == frozen


def test_only_compiled_blocks_reach_projection_and_rebuild_keeps_memory_identity(tmp_path):
    sources = Sources()
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        payload = delta()
        payload["points"] = [
            point("p1", "如果服务器恢复。"),
            point("p2", "该条件下不重启数据库。"),
            point("p3", "另一独立要求。"),
            point("p4", "指代不明。"),
        ]
        payload["points"][-1].update(status="defer", unresolved_refs=["那个"])
        payload["dependencies"] = [
            edge(
                from_point_id="p2",
                target_kind="point",
                target_id="p1",
                boundary_policy="cohabit",
                lifecycle_effect="none",
            )
        ]
        binding = sources.bind("current")
        receipt = service(core, sources).admit_semantic_delta(
            payload, binding=binding, activated_memory_ids=()
        )
        texts = []

        class Writer:
            def upsert(self, memory, *, intent):
                texts.append(memory.content)
                return memory.memory_id

        queue = core.canonical.projection_queue()
        assert ProjectionWorker(queue, Writer(), target="unassigned").run_once() == (2, 0)
        assert set(texts) == {"如果服务器恢复。\n该条件下不重启数据库。", "另一独立要求。"}
        assert "指代不明。" not in texts
        original = core.load_all()
        queue.rebuild(target="embedding-revision-B")
        assert ProjectionWorker(queue, Writer(), target="embedding-revision-B").run_once() == (2, 0)
        assert core.load_all() == original
        assert receipt.memory_ids == tuple(
            m.memory_id for m in (core.get(mid) for mid in receipt.memory_ids)
        )


def test_conflicting_retry_does_not_replace_frozen_or_committed_compilation(tmp_path):
    sources = Sources()
    binding = sources.bind("current")
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        admission = service(core, sources)
        receipt = admission.admit_semantic_delta(delta(), binding=binding, activated_memory_ids=())
        changed = delta()
        changed["points"][0]["meaning"] = "不同的解释。"
        with pytest.raises(ValueError, match="conflicting accepted"):
            admission.admit_semantic_delta(changed, binding=binding, activated_memory_ids=())
        assert admission.semantic_delta_receipt(binding) == receipt
        assert len(core.load_all()) == 1
