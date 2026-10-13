from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mr_mem import MemoryCore, ThreadStatus
from mr_mem.memory.store import CanonicalMemoryStore
from tests.conftest import memory


def test_thread_product_persists_mature_thread(tmp_path) -> None:
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


def test_one_interaction_can_raise_and_settle_a_matter(tmp_path) -> None:
    """"I want this laptop" -> "no money, not buying" in one turn: the matter has
    ended, so the line is mature without a second interaction."""
    path = tmp_path / "memory.sqlite"
    canonical = CanonicalMemoryStore(path)
    try:
        canonical._commit(
            (
                memory("want", content="The user wants to buy a powerful laptop.",
                       interaction_id="interaction-1"),
                memory("decline", content="The user decides not to buy it: no budget.",
                       interaction_id="interaction-1"),
            )
        )
    finally:
        canonical.close()
    at = datetime(2026, 9, 29, tzinfo=UTC)
    with MemoryCore(path) as core:
        core.products.open_thread(
            thread_id="laptop",
            scope=memory().scope,
            open_question="Will the user buy the laptop?",
            supporting_memory_ids=("want",),
            at=at,
            working_summary="The user wants the laptop.",
        )
        resolved = core.products.resolve_thread(
            "laptop", memory_id="decline", at=at,
            working_summary="The user decided not to buy the laptop for lack of budget.",
        )
        assert resolved.status is ThreadStatus.RESOLVED
        assert resolved.mature is True
        assert core.products.thread_handoff("laptop").thread_id == "laptop"
