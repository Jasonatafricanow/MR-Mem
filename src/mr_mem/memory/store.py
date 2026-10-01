"""Canonical Memory persistence. Only admission calls the private commit seam."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mr_mem.contracts import Scope, ScopeDomain, SyncFields
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle, MemoryProvenance
from mr_mem.memory.source import SourceRef

if TYPE_CHECKING:
    from mr_mem.memory.projection import ProjectionQueue


class MemoryConflict(ValueError):
    """Immutable identity reuse; never last-write-wins."""


def scope_json(scope: Scope) -> str:
    return json.dumps(asdict(scope), sort_keys=True, ensure_ascii=False)


def _encode(memory: CommittedMemory) -> str:
    data = asdict(memory)
    data["committed_at"] = memory.committed_at.isoformat()
    for ref in data["provenance"]["source_refs"]:
        ref["occurred_at"] = ref["occurred_at"].isoformat()
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _decode(payload: str) -> CommittedMemory:
    data = json.loads(payload)
    scope_data = data.pop("scope")
    scope_data["domain"] = ScopeDomain(scope_data["domain"])
    scope = Scope(**scope_data)
    sync_data = data.pop("sync")
    if sync_data.pop("scope") != json.loads(scope_json(scope)):
        raise MemoryConflict("corrupt sync scope")
    provenance = data.pop("provenance")
    provenance["evidence_refs"] = tuple(provenance["evidence_refs"])
    provenance["source_refs"] = tuple(
        SourceRef(**{**ref, "occurred_at": datetime.fromisoformat(ref["occurred_at"])})
        for ref in provenance.get("source_refs", ())
    )
    data["attributes"] = tuple(tuple(pair) for pair in data.get("attributes", ()))
    for name in ("supports_memory_ids", "contradicts_memory_ids"):
        data[name] = tuple(data.get(name, ()))
    data["lifecycle"] = MemoryLifecycle(data["lifecycle"])
    data["committed_at"] = datetime.fromisoformat(data["committed_at"])
    return CommittedMemory(
        scope=scope,
        sync=SyncFields(scope=scope, **sync_data),
        provenance=MemoryProvenance(**provenance),
        **data,
    )


class CanonicalMemoryStore:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        if read_only:
            self._conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
            self._conn.execute("PRAGMA query_only=ON")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS admission_jobs (
                source_key TEXT PRIMARY KEY, registered_at TEXT NOT NULL,
                candidates TEXT, done INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS canonical_memory (
                memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS semantic_sources (
                scope TEXT NOT NULL, source_key TEXT NOT NULL, version_key TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES canonical_memory(memory_id),
                PRIMARY KEY(scope, source_key, memory_id));
            CREATE INDEX IF NOT EXISTS semantic_sources_lookup
                ON semantic_sources(scope, source_key, version_key);
            CREATE TABLE IF NOT EXISTS projection_intents (
                intent_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES canonical_memory(memory_id),
                target TEXT NOT NULL, operation TEXT NOT NULL DEFAULT 'upsert',
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                provider_ref TEXT, last_error TEXT,
                UNIQUE(memory_id, target));
        """)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    def get(self, memory_id: str) -> CommittedMemory | None:
        row = self._conn.execute(
            "SELECT payload FROM canonical_memory WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return None if row is None else _decode(row[0])

    def load_all(self) -> tuple[CommittedMemory, ...]:
        return tuple(
            _decode(row[0])
            for row in self._conn.execute("SELECT payload FROM canonical_memory ORDER BY memory_id")
        )

    def _insert(self, memory: CommittedMemory) -> None:
        payload = _encode(memory)
        row = self._conn.execute(
            "SELECT payload FROM canonical_memory WHERE memory_id=?", (memory.memory_id,)
        ).fetchone()
        if row is not None:
            if _decode(row[0]) != memory:
                raise MemoryConflict(f"immutable Memory conflict: {memory.memory_id}")
            return
        self._conn.execute(
            "INSERT INTO canonical_memory VALUES (?, ?)", (memory.memory_id, payload)
        )
        self.projection_queue()._enqueue(memory.memory_id, "unassigned")
        for ref in memory.provenance.source_refs:
            self._conn.execute(
                "INSERT INTO semantic_sources VALUES (?,?,?,?)",
                (scope_json(memory.scope), ref.source_key, ref.version_key, memory.memory_id),
            )

    def _set_lifecycle(self, memory: CommittedMemory, lifecycle: MemoryLifecycle) -> None:
        self._conn.execute(
            "UPDATE canonical_memory SET payload=? WHERE memory_id=?",
            (_encode(replace(memory, lifecycle=lifecycle)), memory.memory_id),
        )
        # Consumers must revalidate canonical lifecycle; stale derived text is never authority.
        self._conn.execute(
            "UPDATE projection_intents SET status='pending',last_error=NULL WHERE memory_id=?",
            (memory.memory_id,),
        )

    def invalidate_source(
        self, scope: Scope, ref: SourceRef, *, deleted: bool = False,
    ) -> tuple[str, ...]:
        """Invalidate only support with this identity and stale revision, or a tombstone."""
        if type(deleted) is not bool:
            raise ValueError("deleted must be bool")
        with self._transaction():
            ids = self._conn.execute(
                "SELECT memory_id FROM semantic_sources WHERE scope=? AND source_key=? "
                "AND (? OR version_key!=?) ORDER BY memory_id",
                (scope_json(scope), ref.source_key, deleted, ref.version_key),
            ).fetchall()
            changed = []
            for (memory_id,) in ids:
                memory = self.get(memory_id)
                if memory is not None and memory.lifecycle is MemoryLifecycle.ACTIVE:
                    self._set_lifecycle(memory, MemoryLifecycle.INVALIDATED)
                    changed.append(memory_id)
        return tuple(changed)

    def _commit_semantic(self, memory: CommittedMemory) -> None:
        with self._transaction():
            previous = None
            if memory.supersedes_memory_id is not None:
                previous = self.get(memory.supersedes_memory_id)
                if previous is None or previous.scope != memory.scope:
                    raise MemoryConflict("supersession requires same-scope canonical support")
            for memory_id in (*memory.supports_memory_ids, *memory.contradicts_memory_ids):
                support = self.get(memory_id)
                if support is None or support.scope != memory.scope:
                    raise MemoryConflict("semantic relation exceeds canonical authority")
            self._insert(memory)
            if previous is not None and previous.lifecycle is not MemoryLifecycle.SUPERSEDED:
                self._set_lifecycle(previous, MemoryLifecycle.SUPERSEDED)

    def _commit(self, memories: tuple[CommittedMemory, ...]) -> None:
        with self._transaction():
            for memory in memories:
                self._insert(memory)

    def _register_job(self, source_key: str, registered_at: datetime) -> None:
        with self._transaction():
            self._conn.execute(
                "INSERT OR IGNORE INTO admission_jobs(source_key,registered_at) VALUES (?,?)",
                (source_key, registered_at.isoformat()),
            )

    def _job(
        self, source_key: str
    ) -> tuple[datetime, tuple[CommittedMemory, ...] | None, bool] | None:
        row = self._conn.execute(
            "SELECT registered_at,candidates,done FROM admission_jobs WHERE source_key=?",
            (source_key,),
        ).fetchone()
        if row is None:
            return None
        candidates = None if row[1] is None else tuple(_decode(p) for p in json.loads(row[1]))
        return datetime.fromisoformat(row[0]), candidates, bool(row[2])

    def _freeze_job(self, source_key: str, memories: tuple[CommittedMemory, ...]) -> None:
        payload = json.dumps([_encode(m) for m in memories])
        with self._transaction():
            row = self._conn.execute(
                "SELECT candidates FROM admission_jobs WHERE source_key=?", (source_key,)
            ).fetchone()
            if row is None or (row[0] is not None and row[0] != payload):
                raise MemoryConflict("missing job or conflicting extraction replay")
            self._conn.execute(
                "UPDATE admission_jobs SET candidates=? WHERE source_key=?", (payload, source_key)
            )

    def _complete_job(self, source_key: str) -> None:
        with self._transaction():
            job = self._job(source_key)
            if job is None or job[1] is None:
                raise MemoryConflict("no frozen candidates")
            if job[2]:
                return
            for memory in job[1]:
                self._insert(memory)
            self._conn.execute("UPDATE admission_jobs SET done=1 WHERE source_key=?", (source_key,))

    def projection_queue(self) -> "ProjectionQueue":
        """A capability exposing only canonical reads and derived projection writes."""
        from mr_mem.memory.projection import ProjectionQueue

        return ProjectionQueue(self._conn)
