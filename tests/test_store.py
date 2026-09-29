from __future__ import annotations

from dataclasses import replace

import pytest

from mr_mem.memory.store import CanonicalMemoryStore, MemoryConflict
from tests.conftest import memory


def test_canonical_memory_survives_restart_and_enqueues_projection(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    original = memory()

    store = CanonicalMemoryStore(path)
    store._commit((original,))
    store.close()

    reopened = CanonicalMemoryStore(path)
    try:
        assert reopened.get(original.memory_id) == original
        assert reopened.load_all() == (original,)
        pending = reopened.projection_queue().pending(10)
        assert len(pending) == 1
        assert pending[0].memory_id == original.memory_id
    finally:
        reopened.close()


def test_canonical_identity_is_immutable(tmp_path) -> None:
    store = CanonicalMemoryStore(tmp_path / "memory.sqlite")
    original = memory()
    try:
        store._commit((original,))
        store._commit((original,))
        with pytest.raises(MemoryConflict):
            store._commit((replace(original, content="changed"),))
        assert store.load_all() == (original,)
    finally:
        store.close()
