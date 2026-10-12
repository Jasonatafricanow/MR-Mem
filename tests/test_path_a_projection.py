"""Path A contract tests: real Canonical commits, no gold or precomputed Thread plan."""
from dataclasses import replace

import pytest

from mr_mem.memory.core import MemoryCore
from mr_mem.memory.path_a import PathAAction, PathAProjector, PathAUpdate
from tests.test_native_semantics import KNOWN, SCOPE, Host, candidate, service


class Publisher:
    def __init__(self):
        self.projections = []
        self.fail = False

    def publish(self, projection):
        if self.fail:
            raise RuntimeError("LCE unavailable")
        self.projections.append(projection)
        return "baseline-1"


def put(core, host, record, identity, content):
    ref = host.put(record, "RAW-ONLY: " + record)
    return service(core, host).admit(candidate(ref, identity, content))


def update(event_id, action, ids, text=None):
    return PathAUpdate(
        event_id, "stable-decision-key", action, ids,
        open_question="Should we proceed?", understanding=text,
    )


def test_canonical_thread_evolves_then_handoffs_to_lce(tmp_path):
    path, host, publisher = tmp_path / "memory.sqlite", Host(), Publisher()
    with MemoryCore(path) as core:
        one = put(core, host, "turn-1", "sem-1", "Option under consideration")
        projector = PathAProjector(core, publisher=publisher)
        opened = projector.apply(scope=SCOPE, update=update(
            "e1", PathAAction.TRACK, (one.memory_id,), "Decision is open",
        ), at=KNOWN)
        assert opened.status == "active"
        assert core.products.get_thread(opened.thread_id).mature is False
        two = put(core, host, "turn-2", "sem-2", "New constraint")
        finished = projector.apply(scope=SCOPE, update=update(
            "e2", PathAAction.MATURE, (two.memory_id,),
            "Proceed only when the constraint is satisfied",
        ), at=KNOWN)
        assert finished.status == "published"
        assert core.products.get_thread(opened.thread_id) is None
        assert publisher.projections[0].supporting_memory_ids == (
            one.memory_id, two.memory_id,
        )
        assert core.get(one.memory_id) == one and core.get(two.memory_id) == two
        assert b"RAW-ONLY" not in path.read_bytes()
        assert projector.apply(scope=SCOPE, update=update(
            "e2", PathAAction.MATURE, (two.memory_id,),
            "Proceed only when the constraint is satisfied",
        ), at=KNOWN) == finished
        assert len(publisher.projections) == 1
        projector.close()


def test_batch_block_can_mature_by_semantics_not_turn_count(tmp_path):
    host, publisher = Host(), Publisher()
    with MemoryCore(tmp_path / "db.sqlite") as core:
        block = put(core, host, "batch", "block", "Complete decision and reasoning")
        projector = PathAProjector(core, publisher=publisher)
        result = projector.apply(scope=SCOPE, update=update(
            "done", PathAAction.MATURE, (block.memory_id,), "A settled understanding",
        ), at=KNOWN)
        assert result.status == "published"
        assert publisher.projections[0].supporting_memory_ids == (block.memory_id,)
        projector.close()


def test_handoff_failure_is_pending_and_retryable(tmp_path):
    host, publisher = Host(), Publisher()
    with MemoryCore(tmp_path / "db.sqlite") as core:
        block = put(core, host, "turn-1", "sem-1", "A decided fact")
        projector = PathAProjector(core, publisher=publisher)
        publisher.fail = True
        with pytest.raises(RuntimeError, match="unavailable"):
            projector.apply(scope=SCOPE, update=update(
                "e1", PathAAction.MATURE, (block.memory_id,), "Final understanding",
            ), at=KNOWN)
        assert projector._conn.execute(
            "SELECT status FROM patha_event_receipts WHERE event_id='e1'"
        ).fetchone()[0] == "pending"
        publisher.fail = False
        assert projector.flush(scope=SCOPE)[0].status == "published"
        projector.close()


def test_identity_conflict_and_unknown_source_fail_closed(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "db.sqlite") as core:
        block = put(core, host, "turn-1", "sem-1", "A fact")
        projector = PathAProjector(core)
        first = update("e1", PathAAction.TRACK, (block.memory_id,), "Open")
        projector.apply(scope=SCOPE, update=first, at=KNOWN)
        with pytest.raises(ValueError, match="replay conflicts"):
            projector.apply(scope=SCOPE, update=replace(
                first, understanding="Changed",
            ), at=KNOWN)
        with pytest.raises(ValueError, match="unavailable"):
            projector.apply(scope=SCOPE, update=update(
                "e2", PathAAction.TRACK, ("missing",),
            ), at=KNOWN)
        projector.close()


def test_existing_legacy_maturity_gate_remains_explicitly_opt_in(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "db.sqlite") as core:
        block = put(core, host, "turn-1", "sem-1", "Complete proposition")
        core.products.open_thread(
            thread_id="legacy", scope=SCOPE, open_question="Legacy?",
            supporting_memory_ids=(block.memory_id,), at=KNOWN,
        )
        with pytest.raises(ValueError, match="at least two"):
            core.products.update_thread(
                "legacy", supporting_memory_ids=(block.memory_id,), at=KNOWN,
                working_summary="Done", mature=True,
            )
        current = core.products.update_thread(
            "legacy", supporting_memory_ids=(block.memory_id,), at=KNOWN,
            working_summary="Done", mature=True, semantic_maturity=True,
        )
        assert current.maturity_basis == "semantic"
        assert core.products.thread_handoff("legacy") == current


def test_reopened_thread_does_not_inherit_prior_baseline_support(tmp_path):
    host, publisher = Host(), Publisher()
    with MemoryCore(tmp_path / "db.sqlite") as core:
        first = put(core, host, "t1", "first", "Original conclusion")
        second = put(core, host, "t2", "second", "Updated conclusion")
        projector = PathAProjector(core, publisher=publisher)
        projector.apply(scope=SCOPE, update=update(
            "first-maturity", PathAAction.MATURE, (first.memory_id,),
            "First settled conclusion",
        ), at=KNOWN)
        projector.apply(scope=SCOPE, update=update(
            "second-maturity", PathAAction.MATURE, (second.memory_id,),
            "New independently settled conclusion",
        ), at=KNOWN)
        assert len(publisher.projections) == 2
        assert publisher.projections[1].supporting_memory_ids == (second.memory_id,)
        projector.close()
