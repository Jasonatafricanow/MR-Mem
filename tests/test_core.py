from __future__ import annotations

import pytest

from mr_mem import MemoryCore, MemoryCoreSelectionError, Scope, ScopeDomain
from mr_mem.memory.store import CanonicalMemoryStore
from tests.conftest import memory


def seed(path) -> tuple[str, ...]:
    store = CanonicalMemoryStore(path)
    try:
        rows = (
            memory("memory-1"),
            memory(
                "memory-2",
                content="The user is considering replacing a laptop.",
                interaction_id="interaction-2",
            ),
        )
        store._commit(rows)
        return tuple(item.memory_id for item in rows)
    finally:
        store.close()


def test_memory_core_selects_exact_order_and_scope(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    ids = seed(path)
    scope = Scope(ScopeDomain.USER, user_id="user-1")

    with MemoryCore(path, read_only=True) as core:
        selected = core.select(
            ids[::-1],
            scope=scope,
            active_only=True,
        )

    assert tuple(item.memory_id for item in selected) == ids[::-1]


def test_memory_core_rejects_partial_duplicate_and_wrong_scope(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    (memory_id, _) = seed(path)

    with MemoryCore(path, read_only=True) as core:
        with pytest.raises(MemoryCoreSelectionError):
            core.select((memory_id, "missing"))
        with pytest.raises(MemoryCoreSelectionError):
            core.select((memory_id, memory_id))
        with pytest.raises(MemoryCoreSelectionError):
            core.select(
                (memory_id,),
                scope=Scope(ScopeDomain.USER, user_id="other-user"),
            )
