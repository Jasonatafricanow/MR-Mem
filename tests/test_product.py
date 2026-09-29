from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mr_mem import MemoryCore, ThreadStatus
from mr_mem.memory.store import CanonicalMemoryStore
from tests.conftest import memory


def test_thread_product_persists_and_requires_independent_support(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    canonical = CanonicalMemoryStore(path)
    try:
        canonical._commit(
            (
                memory("memory-1", interaction_id="interaction-1"),
                memory(
                    "memory-2",
                    content="Battery pressure made the replacement urgent.",
                    interaction_id="interaction-2",
                ),
            )
        )
    finally:
        canonical.close()

    at = datetime(2026, 9, 29, tzinfo=UTC)
    with MemoryCore(path) as core:
        opened = core.products.open_thread(
            thread_id="replace-laptop",
            scope=memory().scope,
            open_question="Will the laptop be replaced?",
            supporting_memory_ids=("memory-1",),
            at=at,
            working_summary="Replacement remains undecided.",
        )
        assert opened.status is ThreadStatus.OPEN
        updated = core.products.update_thread(
            "replace-laptop",
            supporting_memory_ids=("memory-1", "memory-2"),
            at=at + timedelta(days=1),
            working_summary="Battery pressure makes replacement more likely.",
            mature=True,
        )
        assert updated.mature is True

    with MemoryCore(path, read_only=True) as reopened:
        thread = reopened.products.get_thread("replace-laptop")
        assert thread is not None
        assert thread.mature is True
        assert thread.handoff_memory_ids == ("memory-1", "memory-2")
