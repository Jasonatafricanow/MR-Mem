"""Read-only canonical cognition contract for derived consumers.

Native source references are metadata, never transcript access. Compiler
intermediates stay private to admission. Unknown lifecycle times stay unknown.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from mr_mem.contracts import Scope
from mr_mem.memory.contracts import MemoryLifecycle
from mr_mem.memory.semantic_store import read_metadata
from mr_mem.memory.source import SourceRef
from mr_mem.memory.store import _decode


@dataclass(frozen=True, slots=True)
class CanonicalSemanticRelationView:
    from_memory_id: str
    to_memory_id: str
    relation: str
    boundary_policy: str
    lifecycle_effect: str
    source_interaction_id: str
    known_at: datetime


@dataclass(frozen=True, slots=True)
class CanonicalSemanticBlockView:
    memory_id: str
    scope: Scope
    content: str
    occurred_at: datetime
    known_at: datetime
    lifecycle: MemoryLifecycle
    schema_version: str | None
    compiler_version: str
    source_refs: tuple[SourceRef, ...]
    context_memory_ids: tuple[str, ...]
    outgoing_relations: tuple[CanonicalSemanticRelationView, ...]
    incoming_relations: tuple[CanonicalSemanticRelationView, ...]
    source_interaction_id: str | None = None
    transition_known_at: datetime | None = None


class CanonicalSemanticBlockReader(Protocol):
    def get_semantic_block_view(self, memory_id: str) -> CanonicalSemanticBlockView | None: ...

    def semantic_relations(self, memory_id: str) -> tuple[CanonicalSemanticRelationView, ...]: ...


class SqliteCanonicalSemanticBlockReader:
    """A separate query-only connection with no admission or mutation capability."""

    def __init__(self, path: str | Path) -> None:
        self.__conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.__conn.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        self.__conn.close()

    def semantic_relations(self, memory_id: str) -> tuple[CanonicalSemanticRelationView, ...]:
        return tuple(
            CanonicalSemanticRelationView(*row[:6], known_at=_decode(row[6]).known_at)
            for row in self.__conn.execute(
                "SELECT r.*,m.payload FROM semantic_relations r "
                "JOIN canonical_memory m ON m.memory_id=r.from_memory_id "
                "WHERE r.from_memory_id=? OR r.to_memory_id=? "
                "ORDER BY r.from_memory_id,r.to_memory_id,r.relation,r.boundary_policy,"
                "r.lifecycle_effect,r.interaction_id",
                (memory_id, memory_id),
            )
        )

    def _recorded_transition(
        self, memory_id: str, lifecycle: MemoryLifecycle
    ) -> datetime | None:
        try:
            row = self.__conn.execute(
                "SELECT transitioned_at FROM lifecycle_transitions "
                "WHERE memory_id=? AND lifecycle=?", (memory_id, lifecycle.value),
            ).fetchone()
        except sqlite3.OperationalError:  # database predates the table
            return None
        return datetime.fromisoformat(row[0]) if row else None

    def get_semantic_block_view(self, memory_id: str) -> CanonicalSemanticBlockView | None:
        self.__conn.execute("BEGIN")
        try:
            row = self.__conn.execute(
                "SELECT m.payload,s.payload FROM canonical_memory m "
                "LEFT JOIN semantic_block_metadata s ON s.memory_id=m.memory_id "
                "WHERE m.memory_id=?", (memory_id,),
            ).fetchone()
            if row is None:
                return None
            memory = _decode(row[0])
            # Legacy factual admission has no native occurrence authority.
            if memory.occurred_at is None:
                return None
            metadata = read_metadata(row[1]) if row[1] is not None else None
            relations = self.semantic_relations(memory_id)
            outgoing = tuple(r for r in relations if r.from_memory_id == memory_id)
            incoming = tuple(r for r in relations if r.to_memory_id == memory_id)
            transitions = [r.known_at for r in incoming if r.lifecycle_effect == "supersede"]
            transition = (
                min(transitions)
                if memory.lifecycle is MemoryLifecycle.SUPERSEDED and transitions
                else None
            )
            if transition is None and memory.lifecycle is MemoryLifecycle.INVALIDATED:
                transition = self._recorded_transition(memory_id, memory.lifecycle)
            return CanonicalSemanticBlockView(
                memory.memory_id, memory.scope, memory.content,
                memory.occurred_at, memory.known_at, memory.lifecycle,
                metadata.schema_version if metadata else None,
                metadata.compiler_version if metadata else memory.provenance.extractor_version,
                memory.provenance.source_refs,
                metadata.context_memory_ids if metadata else (),
                outgoing, incoming,
                metadata.source_interaction_id if metadata else None,
                transition,
            )
        finally:
            self.__conn.rollback()
