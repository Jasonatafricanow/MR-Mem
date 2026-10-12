"""Real LCE Draft/HEAD smoke, optional unless LCE is installed."""
import pytest

pytest.importorskip("lce")

from mr_mem.memory.core import MemoryCore
from mr_mem.memory.path_a import PathAAction, PathAProjector, PathAUpdate
from mr_mem.memory.path_a_lce import LcePathAProjection
from tests.test_native_semantics import KNOWN, SCOPE, Host, candidate, service


def test_real_lce_path_a_without_path_b(tmp_path):
    host = Host()
    ref = host.put("t1", "RAW witness")
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        memory = service(core, host).admit(candidate(
            ref, "first", "The decision includes an important constraint",
        ))
        lce = LcePathAProjection(core, scope=SCOPE, root=tmp_path / "lce")
        projector = PathAProjector(core, publisher=lce)
        result = projector.apply(
            scope=SCOPE,
            update=PathAUpdate(
                "event-1", "stable-thread-key", PathAAction.MATURE,
                (memory.memory_id,), "What is the decision?",
                "The decision remains conditional on the constraint",
            ),
            at=KNOWN,
        )
        assert result.status == "published"
        found = lce.search("decision constraint")
        assert len(found) == 1
        assert found[0].baseline_id == result.baseline_id
        assert found[0].supporting_memory_ids == (memory.memory_id,)
        assert ref.source_key in found[0].source_refs[0]
        core.canonical.invalidate_source(SCOPE, ref, deleted=True)
        assert lce.search("decision constraint") == ()
        projector.close()
        lce.close()
