"""Path A over *committed* Canonical SemanticBlocks, not raw or SemanticPoints.

The host semantic compiler supplies explicit thread identity, updates and
maturity decisions. This projection neither infers meaning nor re-reads raw
transcripts. LCE alone owns accepted Baseline revisions via a publisher port.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from mr_mem.contracts import Scope
from mr_mem.contracts.common import require_aware_utc, require_non_empty
from mr_mem.memory.core import MemoryCore
from mr_mem.memory.product import MAX_THREAD_CURRENT_SUPPORT, MemoryThread, ThreadStatus
from mr_mem.memory.store import scope_json


class PathAAction(StrEnum):
    TRACK = "track"
    MATURE = "mature"
    ABANDON = "abandon"


@dataclass(frozen=True, slots=True)
class PathAUpdate:
    """Host-authorized semantic verdict over committed Blocks.

    thread_key is stable semantic identity supplied by the compiler;
    maturity is an explicit semantic decision, never inferred from turn count.
    """
    event_id: str
    thread_key: str
    action: PathAAction
    memory_ids: tuple[str, ...]
    open_question: str | None = None
    understanding: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.event_id, "event_id")
        require_non_empty(self.thread_key, "thread_key")
        if not isinstance(self.action, PathAAction):
            raise TypeError("action must be PathAAction")
        if not isinstance(self.memory_ids, tuple) or not self.memory_ids:
            raise ValueError("at least one canonical Memory ID is required")
        if len(self.memory_ids) > 64 or len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("memory IDs must be unique and bounded")
        for mid in self.memory_ids:
            require_non_empty(mid, "memory_id")
        if self.open_question is not None:
            require_non_empty(self.open_question, "open_question")
        if self.understanding is not None:
            require_non_empty(self.understanding, "understanding")
        if self.action is PathAAction.MATURE and not self.understanding:
            raise ValueError("mature requires a completed understanding")


@dataclass(frozen=True, slots=True)
class MatureProjection:
    thread_id: str
    scope: Scope
    understanding: str
    supporting_memory_ids: tuple[str, ...]
    processing_input_id: str


class LceBaselinePublisher(Protocol):
    def publish(self, projection: MatureProjection) -> str:
        """Return durable LCE Baseline ID, idempotent by processing_input_id."""


@dataclass(frozen=True, slots=True)
class PathAReceipt:
    event_id: str
    thread_id: str
    status: str
    baseline_id: str | None = None


def _thread_id(scope: Scope, key: str) -> str:
    data = ["mr-mem-patha-v1", scope_json(scope), key]
    return "patha-" + hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()[:32]


def _event_hash(scope: Scope, event: PathAUpdate) -> str:
    payload = [scope_json(scope), event.thread_key, event.action.value,
               event.memory_ids, event.open_question, event.understanding]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class PathAProjector:
    """Derived Thread projection over Canonical, with optional LCE handoff.

    Stores reference IDs and idempotency receipts, not another factual corpus.
    A publisher failure leaves a mature Thread pending for flush; it never
    fabricates a Baseline.
    """

    def __init__(
        self, core: MemoryCore, *, publisher: LceBaselinePublisher | None = None,
        max_support: int = 256,
    ) -> None:
        if core.read_only:
            raise ValueError("Path A requires writable MemoryCore")
        if type(max_support) is not int or not 1 <= max_support <= 256:
            raise ValueError("max_support must be in 1..256")
        self.core, self.publisher, self.max_support = core, publisher, max_support
        self._conn = sqlite3.connect(Path(core.path))
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS patha_event_receipts (
                event_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                status TEXT NOT NULL,
                baseline_id TEXT
            );
            CREATE TABLE IF NOT EXISTS patha_thread_support (
                thread_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY(thread_id, memory_id),
                UNIQUE(thread_id, position)
            );
        """)

    def close(self) -> None:
        self._conn.close()

    def apply(
        self, *, scope: Scope, update: PathAUpdate, at: datetime,
    ) -> PathAReceipt:
        require_aware_utc(at, "at")
        if not isinstance(scope, Scope) or not isinstance(update, PathAUpdate):
            raise TypeError("structured Scope and PathAUpdate required")
        digest, tid = _event_hash(scope, update), _thread_id(scope, update.thread_key)
        old = self._conn.execute(
            "SELECT payload_hash,thread_id,status,baseline_id "
            "FROM patha_event_receipts WHERE event_id=?", (update.event_id,),
        ).fetchone()
        if old is not None:
            if old[0] != digest or old[1] != tid:
                raise ValueError("event_id replay conflicts with accepted semantic event")
            return PathAReceipt(update.event_id, tid, old[2], old[3])

        # No raw-text input, no invented source keys, no cross-scope or stale support.
        memories = self.core.select(update.memory_ids, scope=scope,
                                    active_only=True, max_items=64)
        if any(not memory.provenance.source_refs for memory in memories):
            raise ValueError("Path A support requires native Canonical SourceRefs")
        thread = self.core.products.get_thread(tid)
        if thread is None:
            if update.action is PathAAction.ABANDON:
                raise ValueError("cannot abandon unknown Thread")
            if not update.open_question:
                raise ValueError("a new Thread needs an explicit open question")
            thread = self.core.products.open_thread(
                thread_id=tid, scope=scope, open_question=update.open_question,
                supporting_memory_ids=update.memory_ids[:8], at=at,
                working_summary=update.understanding,
            )
        else:
            if thread.mature:
                raise ValueError("Thread awaits pending LCE handoff; flush first")
            if thread.scope != scope or thread.status is not ThreadStatus.OPEN:
                raise ValueError("Thread scope/lifecycle mismatch")
            if update.open_question and update.open_question != thread.open_question:
                raise ValueError("thread identity reused with another open question")
            if at < thread.updated_at:
                raise ValueError("out-of-order Thread update")

        existing_support = tuple(row[0] for row in self._conn.execute(
            "SELECT memory_id FROM patha_thread_support WHERE thread_id=? ORDER BY position",
            (tid,),
        ))
        support = tuple(dict.fromkeys((*existing_support, *thread.handoff_memory_ids,
                                       *update.memory_ids)))
        if len(support) > self.max_support:
            raise ValueError("Path A support exceeds bounded projection capacity")
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO patha_thread_support(thread_id,memory_id,position) "
                "VALUES(?,?,?)",
                [(tid, mid, i) for i, mid in enumerate(support)],
            )

        if update.action is PathAAction.ABANDON:
            self.core.products.abandon_thread(tid, at=at)
            status, baseline_id = "abandoned", None
        else:
            current = support[-MAX_THREAD_CURRENT_SUPPORT:]
            thread = self.core.products.update_thread(
                tid, supporting_memory_ids=current, at=at,
                working_summary=update.understanding,
                mature=update.action is PathAAction.MATURE,
                semantic_maturity=update.action is PathAAction.MATURE,
            )
            status, baseline_id = ("pending" if thread.mature else "active"), None
        with self._conn:
            self._conn.execute(
                "INSERT INTO patha_event_receipts VALUES(?,?,?,?,?)",
                (update.event_id, digest, tid, status, baseline_id),
            )
        if status == "pending" and self.publisher is not None:
            baseline_id = self._publish(thread, support)
            status = "published"
            with self._conn:
                self._conn.execute(
                    "UPDATE patha_event_receipts SET status=?,baseline_id=? WHERE event_id=?",
                    (status, baseline_id, update.event_id),
                )
            self.core.products.retire_compiled_thread(tid, baseline_id=baseline_id)
        return PathAReceipt(update.event_id, tid, status, baseline_id)

    def flush(self, *, scope: Scope) -> tuple[PathAReceipt, ...]:
        if self.publisher is None:
            return ()
        receipts: list[PathAReceipt] = []
        # Reconcile a crash after LCE commit and receipt, before retirement.
        for tid, baseline_id in self._conn.execute(
            "SELECT DISTINCT thread_id,baseline_id FROM patha_event_receipts "
            "WHERE status='published' AND baseline_id IS NOT NULL"
        ).fetchall():
            thread = self.core.products.get_thread(tid)
            if thread is not None and thread.scope == scope and thread.mature:
                self.core.products.retire_compiled_thread(tid, baseline_id=baseline_id)
        for event_id, tid in self._conn.execute(
            "SELECT event_id,thread_id FROM patha_event_receipts "
            "WHERE status='pending' ORDER BY rowid"
        ).fetchall():
            thread = self.core.products.get_thread(tid)
            if thread is None or thread.scope != scope or not thread.mature:
                continue
            support = tuple(row[0] for row in self._conn.execute(
                "SELECT memory_id FROM patha_thread_support "
                "WHERE thread_id=? ORDER BY position", (tid,),
            ))
            baseline_id = self._publish(thread, support)
            with self._conn:
                self._conn.execute(
                    "UPDATE patha_event_receipts SET status='published',baseline_id=? "
                    "WHERE event_id=?", (baseline_id, event_id),
                )
            self.core.products.retire_compiled_thread(tid, baseline_id=baseline_id)
            receipts.append(PathAReceipt(event_id, tid, "published", baseline_id))
        return tuple(receipts)

    def _publish(self, thread: MemoryThread, support: tuple[str, ...]) -> str:
        assert self.publisher is not None
        checked = self.core.select(support, scope=thread.scope, active_only=True,
                                   max_items=self.max_support)
        if any(not item.provenance.source_refs for item in checked):
            raise ValueError("support lost native provenance before handoff")
        if not thread.working_summary:
            raise ValueError("mature Thread lacks understanding")
        key = hashlib.sha256(json.dumps(
            [thread.thread_id, thread.working_summary, support],
            ensure_ascii=False,
        ).encode()).hexdigest()
        projection = MatureProjection(
            thread.thread_id, thread.scope, thread.working_summary, support,
            "patha:" + key,
        )
        baseline_id = self.publisher.publish(projection)
        require_non_empty(baseline_id, "baseline_id")
        return baseline_id
