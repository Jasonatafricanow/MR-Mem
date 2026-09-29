"""Runtime-independent Memory core over one canonical MR-Mem database.

This module is the reusable ownership boundary for tools that need canonical
Memory plus its local derived product projections without importing RuntimeBinding,
host composition, Body agents, retrieval providers, or LCE.

It does not grant new write authority. Canonical commits still flow through the
existing admission service; callers receive the same stores and invariants that
the bound MR runtime uses.
"""

from __future__ import annotations

from pathlib import Path

from mr_mem.contracts import Scope
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.product import MemoryProductStore
from mr_mem.memory.store import CanonicalMemoryStore


class MemoryCoreSelectionError(ValueError):
    """A bounded canonical selection could not be resolved exactly."""


class MemoryCore:
    """Reusable MR-Mem core addressed by its canonical database path.

    MemoryCore owns connection lifetime only. It deliberately knows nothing
    about RuntimeBinding, storage namespaces, LCE, AML, model providers, or host
    composition. Those concerns belong to adapters outside this module.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._canonical = CanonicalMemoryStore(self.path, read_only=read_only)
        try:
            self._products = MemoryProductStore(
                self.path,
                self._canonical,
                read_only=read_only,
            )
        except BaseException:
            self._canonical.close()
            raise
        self._closed = False

    @property
    def canonical(self) -> CanonicalMemoryStore:
        """Canonical factual store; private commit seams remain unchanged."""
        return self._canonical

    @property
    def products(self) -> MemoryProductStore:
        """Derived attention/Thread projection store over canonical Memory."""
        return self._products

    def get(self, memory_id: str) -> CommittedMemory | None:
        return self._canonical.get(memory_id)

    def load_all(self) -> tuple[CommittedMemory, ...]:
        return self._canonical.load_all()

    def select(
        self,
        memory_ids: tuple[str, ...],
        *,
        scope: Scope | None = None,
        active_only: bool = False,
        max_items: int = 100,
    ) -> tuple[CommittedMemory, ...]:
        """Resolve an exact bounded selection while preserving caller order.

        Missing, duplicate, out-of-scope, or ineligible IDs reject the complete
        selection. Returning a partial set would let a projection silently
        change the evidence boundary.
        """
        if type(max_items) is not int or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        if not isinstance(memory_ids, tuple) or not 1 <= len(memory_ids) <= max_items:
            raise MemoryCoreSelectionError(
                f"expected a tuple of 1 to {max_items} stable Memory IDs"
            )
        if any(
            not isinstance(memory_id, str) or not memory_id.strip()
            for memory_id in memory_ids
        ):
            raise MemoryCoreSelectionError(
                "each Memory ID must be a non-empty string"
            )
        if len(set(memory_ids)) != len(memory_ids):
            raise MemoryCoreSelectionError("duplicate Memory IDs are not permitted")
        if scope is not None and not isinstance(scope, Scope):
            raise TypeError("scope must be Scope or None")

        memories = tuple(self._canonical.get(memory_id) for memory_id in memory_ids)
        if any(memory is None for memory in memories):
            raise MemoryCoreSelectionError(
                "selected Memory set contains unavailable IDs"
            )

        resolved = tuple(memory for memory in memories if memory is not None)
        if scope is not None and any(memory.scope != scope for memory in resolved):
            raise MemoryCoreSelectionError(
                "selected Memory set contains out-of-scope IDs"
            )
        if active_only and any(
            memory.lifecycle is not MemoryLifecycle.ACTIVE for memory in resolved
        ):
            raise MemoryCoreSelectionError(
                "selected Memory set contains inactive IDs"
            )
        return resolved

    def close(self) -> None:
        if self._closed:
            return
        self._products.close()
        self._canonical.close()
        self._closed = True

    def __enter__(self) -> MemoryCore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
