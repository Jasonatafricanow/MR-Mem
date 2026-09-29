"""Canonical Memory persistence. Only admission calls the private commit seam."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mr_mem.contracts import Scope, ScopeDomain, SyncFields
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle, MemoryProvenance

if TYPE_CHECKING:
    from mr_mem.memory.projection import ProjectionQueue


class MemoryConflict(ValueError):
    """Immutable identity reuse; never last-write-wins."""


def scope_json(scope: Scope) -> str:
    return json.dumps(asdict(scope), sort_keys=True, ensure_ascii=False)


def _encode(memory: CommittedMemory) -> str:
    data = asdict(memory)
    data["committed_at"] = memory.committed_at.isoformat()
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
            if row[0] != payload:
                raise MemoryConflict(f"immutable Memory conflict: {memory.memory_id}")
            return
        self._conn.execute(
            "INSERT INTO canonical_memory VALUES (?, ?)", (memory.memory_id, payload)
        )
        self.projection_queue()._enqueue(memory.memory_id, "unassigned")

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
