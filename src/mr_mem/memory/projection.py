"""Rebuildable index state; this capability cannot admit canonical Memory."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mr_mem.contracts.common import require_non_empty
from mr_mem.memory.contracts import CommittedMemory
from mr_mem.memory.store import _decode


@dataclass(frozen=True, slots=True)
class ProjectionIntent:
    intent_id: str
    memory_id: str
    target: str
    operation: str
    status: str
    attempts: int
    provider_ref: str | None
    last_error: str | None
    generation: int = 0


class ProjectionQueue:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.__conn = connection

    @property
    def database_path(self) -> Path | None:
        """Physical storage provenance for bound provider composition, never cognition."""
        for _, name, path in self.__conn.execute("PRAGMA database_list"):
            if name == "main" and path:
                return Path(path).resolve()
        return None

    def _enqueue(self, memory_id: str, target: str) -> None:
        require_non_empty(target, "target")
        identity = hashlib.sha256(json.dumps([memory_id, target]).encode()).hexdigest()
        self.__conn.execute(
            "INSERT OR IGNORE INTO projection_intents(intent_id,memory_id,target) VALUES(?,?,?)",
            (f"projection-{identity}", memory_id, target),
        )

    def pending(self, limit: int, *, target: str | None = None) -> tuple[ProjectionIntent, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if target is not None:
            require_non_empty(target, "target")
        return tuple(
            ProjectionIntent(*row)
            for row in self.__conn.execute(
                "SELECT * FROM projection_intents WHERE status != 'succeeded' "
                "AND (? IS NULL OR target=?) "
                "ORDER BY intent_id LIMIT ?",
                (target, target, limit),
            )
        )

    def register_projection_target(self, target: str) -> int:
        """Atomically persist a consumer target and backfill existing canonical IDs.

        Registration survives restart. Future canonical inserts enqueue every
        registered target in their admission transaction. Re-registration does
        not reset successful intents; rebuild(reset=True) explicitly does.
        """
        require_non_empty(target, "target")
        self.__conn.execute("BEGIN IMMEDIATE")
        try:
            self.__conn.execute("INSERT OR IGNORE INTO projection_targets VALUES (?)", (target,))
            before = self.__conn.total_changes
            for (memory_id,) in self.__conn.execute("SELECT memory_id FROM canonical_memory"):
                self._enqueue(memory_id, target)
            count = self.__conn.total_changes - before
            self.__conn.commit()
            return count
        except BaseException:
            self.__conn.rollback()
            raise

    def get(self, memory_id: str) -> CommittedMemory | None:
        row = self.__conn.execute(
            "SELECT payload FROM canonical_memory WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return None if row is None else _decode(row[0])

    def begin_attempt(self, intent_id: str) -> None:
        with self.__conn:
            self.__conn.execute(
                "UPDATE projection_intents SET attempts=attempts+1 WHERE intent_id=?", (intent_id,)
            )

    def succeed(
        self, intent_id: str, provider_ref: str, *, expected_generation: int | None = None,
    ) -> None:
        require_non_empty(provider_ref, "provider_ref")
        with self.__conn:
            self.__conn.execute(
                "UPDATE projection_intents SET status='succeeded', "
                "provider_ref=?,last_error=NULL WHERE intent_id=? "
                "AND (? IS NULL OR generation=?)",
                (provider_ref, intent_id, expected_generation, expected_generation),
            )

    def fail(self, intent_id: str, error: str) -> None:
        with self.__conn:
            self.__conn.execute(
                "UPDATE projection_intents SET status='failed_retryable', "
                "last_error=? WHERE intent_id=?",
                (error[:1024], intent_id),
            )

    def wipe(self) -> None:
        """Drop only local derived index state, never canonical records."""
        with self.__conn:
            self.__conn.execute("DELETE FROM projection_intents")

    def rebuild(self, *, target: str = "unassigned", reset: bool = False) -> int:
        require_non_empty(target, "target")
        if type(reset) is not bool:
            raise ValueError("reset must be bool")
        self.__conn.execute("BEGIN IMMEDIATE")
        try:
            if reset:
                self.__conn.execute("DELETE FROM projection_intents WHERE target=?", (target,))
            before = self.__conn.total_changes
            for (memory_id,) in self.__conn.execute("SELECT memory_id FROM canonical_memory"):
                self._enqueue(memory_id, target)
            count = self.__conn.total_changes - before
            self.__conn.commit()
            return count
        except BaseException:
            self.__conn.rollback()
            raise


class ProjectionWriter(Protocol):
    """Idempotent upsert by stable intent identity; returns an opaque provider ref.

    Provider/embedding configuration belongs to the injected implementation.
    Target names identify fixed composition-owned projection revisions. A new
    provider/model configuration uses a new target via rebuild, not a new Memory ID.
    A crash may repeat an upsert with the identical intent_id and immutable Memory.
    """

    def upsert(self, memory: CommittedMemory, *, intent: ProjectionIntent) -> str: ...


class ProjectionWorker:
    def __init__(
        self, queue: ProjectionQueue, writer: ProjectionWriter, *, target: str | None = None,
    ) -> None:
        if target is not None:
            require_non_empty(target, "target")
        self._queue, self._writer = queue, writer
        validate_queue = getattr(writer, "validate_queue", None)
        if validate_queue is not None:
            validate_queue(queue)
        self._target = target

    def run_once(self, *, limit: int = 32) -> tuple[int, int]:
        """Return success/failure counts; each pending intent attempted at most once."""
        succeeded = failed = 0
        for intent in self._queue.pending(limit, target=self._target):
            self._queue.begin_attempt(intent.intent_id)
            memory = self._queue.get(intent.memory_id)
            if memory is None:
                self._queue.fail(intent.intent_id, "canonical Memory missing")
                failed += 1
                continue
            try:
                ref = self._writer.upsert(memory, intent=intent)
                self._queue.succeed(
                    intent.intent_id, ref, expected_generation=intent.generation,
                )
            except Exception as exc:
                self._queue.fail(intent.intent_id, f"{type(exc).__name__}: {exc}")
                failed += 1
            else:
                succeeded += 1
        return succeeded, failed
