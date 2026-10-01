"""Product-level memory governance over canonical MR-Mem.

Canonical Memory remains semantic authority. This module stores only derived
attention state and short-lived logical projections in the same memory.sqlite.
Neither attention nor threads can create Evidence, Observation, or Memory.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from mr_mem.contracts import Scope, ScopeDomain
from mr_mem.contracts.common import require_aware_utc, require_non_empty
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle


class CanonicalMemoryReader(Protocol):
    def get(self, memory_id: str) -> CommittedMemory | None: ...


@dataclass(frozen=True, slots=True)
class MemoryAttention:
    """Derived product state. It never changes canonical Memory authority."""

    memory_id: str
    importance: int = 5
    activation_count: int = 0
    last_reinforced_at: datetime | None = None
    resolved: bool = False
    digested: bool = False
    dont_surface: bool = False

    def __post_init__(self) -> None:
        require_non_empty(self.memory_id, "memory_id")
        if type(self.importance) is not int or not 1 <= self.importance <= 10:
            raise ValueError("importance must be an integer in [1, 10]")
        if type(self.activation_count) is not int or self.activation_count < 0:
            raise ValueError("activation_count must be a non-negative integer")
        if self.last_reinforced_at is not None:
            require_aware_utc(self.last_reinforced_at, "last_reinforced_at")
        for name in ("resolved", "digested", "dont_surface"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")


class SurfaceMode(StrEnum):
    EXPLICIT_SEARCH = "explicit_search"
    AUTOMATIC = "automatic"
    SPONTANEOUS = "spontaneous"
    LCE = "lce"


@dataclass(frozen=True, slots=True)
class SurfaceDecision:
    allowed: bool
    score: float
    reasons: tuple[str, ...] = ()


class MemorySurfacePolicy:
    """Read-side visibility policy; decay changes attention, never truth."""

    def __init__(self, *, decay_lambda: float = 0.03, resolved_factor: float = 0.25) -> None:
        if not math.isfinite(decay_lambda) or decay_lambda < 0:
            raise ValueError("decay_lambda must be finite and non-negative")
        if not math.isfinite(resolved_factor) or not 0 <= resolved_factor <= 1:
            raise ValueError("resolved_factor must be finite and in [0, 1]")
        self._decay_lambda = decay_lambda
        self._resolved_factor = resolved_factor

    def evaluate(
        self,
        memory: CommittedMemory,
        attention: MemoryAttention,
        *,
        mode: SurfaceMode,
        now: datetime,
    ) -> SurfaceDecision:
        require_aware_utc(now, "now")
        if memory.memory_id != attention.memory_id:
            raise ValueError("attention must belong to the evaluated Memory")
        if memory.lifecycle is not MemoryLifecycle.ACTIVE:
            return SurfaceDecision(False, 0.0, ("inactive_memory",))
        reasons: list[str] = []
        if mode in (SurfaceMode.AUTOMATIC, SurfaceMode.SPONTANEOUS):
            if attention.dont_surface:
                reasons.append("dont_surface")
            if attention.digested:
                reasons.append("digested")
        if reasons:
            return SurfaceDecision(False, 0.0, tuple(reasons))
        age_days = max(0.0, (now - memory.committed_at).total_seconds() / 86400.0)
        score = (
            attention.importance
            * (1.0 + math.log1p(attention.activation_count))
            * math.exp(-self._decay_lambda * age_days)
        )
        if attention.resolved:
            score *= self._resolved_factor
        return SurfaceDecision(True, round(score, 6))


class ThreadStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


MAX_THREAD_ORIGIN_SUPPORT = 8
MAX_THREAD_CURRENT_SUPPORT = 8


@dataclass(frozen=True, slots=True)
class MemoryThread:
    """Bounded medium-term working structure over canonical Memory.

    A Thread owns only the fact that one explicit line is still being worked,
    its bounded current support, and any already-reasoned online summary that
    is ready for longitudinal compilation. It is not a second history store
    and does not classify trajectory transitions such as progress/reversal.
    """

    thread_id: str
    scope: Scope
    open_question: str
    status: ThreadStatus
    importance: int
    created_at: datetime
    updated_at: datetime
    touch_count: int
    suppressed: bool
    origin_memory_ids: tuple[str, ...]
    current_support_ids: tuple[str, ...]
    working_summary: str | None = None
    mature: bool = False

    def __post_init__(self) -> None:
        require_non_empty(self.thread_id, "thread_id")
        if not isinstance(self.scope, Scope):
            raise ValueError("scope must be Scope")
        if not isinstance(self.open_question, str) or not self.open_question.strip():
            raise ValueError("open_question must be nonempty")
        if len(self.open_question) > 2048:
            raise ValueError("open_question exceeds 2048 characters")
        if not isinstance(self.status, ThreadStatus):
            raise ValueError("status must be ThreadStatus")
        if type(self.importance) is not int or not 1 <= self.importance <= 10:
            raise ValueError("importance must be an integer in [1, 10]")
        require_aware_utc(self.created_at, "created_at")
        require_aware_utc(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if type(self.touch_count) is not int or self.touch_count < 0:
            raise ValueError("touch_count must be a non-negative integer")
        if type(self.suppressed) is not bool:
            raise ValueError("suppressed must be bool")
        if (
            not self.origin_memory_ids
            or len(self.origin_memory_ids) > MAX_THREAD_ORIGIN_SUPPORT
            or len(set(self.origin_memory_ids)) != len(self.origin_memory_ids)
        ):
            raise ValueError("origin_memory_ids must contain 1..8 unique items")
        if (
            not self.current_support_ids
            or len(self.current_support_ids) > MAX_THREAD_CURRENT_SUPPORT
            or len(set(self.current_support_ids)) != len(self.current_support_ids)
        ):
            raise ValueError("current_support_ids must contain 1..8 unique items")
        for memory_id in (*self.origin_memory_ids, *self.current_support_ids):
            require_non_empty(memory_id, "thread support memory_id")
        if self.working_summary is not None:
            if not isinstance(self.working_summary, str) or not self.working_summary.strip():
                raise ValueError("working_summary must be nonempty when supplied")
            if len(self.working_summary) > 4096:
                raise ValueError("working_summary exceeds 4096 characters")
        if type(self.mature) is not bool:
            raise ValueError("mature must be bool")
        if self.mature and self.working_summary is None:
            raise ValueError("mature Thread requires working_summary")

    @property
    def handoff_memory_ids(self) -> tuple[str, ...]:
        """Bounded canonical support carried into longitudinal compilation."""
        return tuple(dict.fromkeys((*self.origin_memory_ids, *self.current_support_ids)))


class MemoryProductConflict(ValueError):
    """A stable product identity was reused with conflicting content."""


class MemoryProductStore:
    """Derived product state stored beside canonical rows in memory.sqlite."""

    def __init__(
        self,
        path: str | Path,
        canonical: CanonicalMemoryReader,
        *,
        read_only: bool = False,
    ) -> None:
        self._canonical = canonical
        self._read_only = read_only
        if read_only:
            self._conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
            self._conn.execute("PRAGMA query_only=ON")
        else:
            self._conn = sqlite3.connect(path)
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_attention (
                    memory_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_threads (
                    thread_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        self._conn.close()

    def _writable(self) -> None:
        if self._read_only:
            raise sqlite3.OperationalError("MemoryProductStore is read-only")

    def _table_exists(self, table: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    def _memory(self, memory_id: str, *, active: bool = False) -> CommittedMemory:
        require_non_empty(memory_id, "memory_id")
        memory = self._canonical.get(memory_id)
        if memory is None:
            raise ValueError(f"unknown canonical Memory: {memory_id}")
        if active and memory.lifecycle is not MemoryLifecycle.ACTIVE:
            raise ValueError(f"canonical Memory is not ACTIVE: {memory_id}")
        return memory

    def attention(self, memory_id: str) -> MemoryAttention:
        self._memory(memory_id)
        if not self._table_exists("memory_attention"):
            return MemoryAttention(memory_id)
        row = self._conn.execute(
            "SELECT payload FROM memory_attention WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return MemoryAttention(memory_id) if row is None else _decode_attention(row[0])

    def set_attention(
        self,
        memory_id: str,
        *,
        importance: int | None = None,
        resolved: bool | None = None,
        digested: bool | None = None,
        dont_surface: bool | None = None,
    ) -> MemoryAttention:
        self._writable()
        current = self.attention(memory_id)
        updated = replace(
            current,
            importance=current.importance if importance is None else importance,
            resolved=current.resolved if resolved is None else resolved,
            digested=current.digested if digested is None else digested,
            dont_surface=current.dont_surface if dont_surface is None else dont_surface,
        )
        self._put_attention(updated)
        return updated

    def reinforce(self, memory_id: str, *, at: datetime) -> MemoryAttention:
        self._writable()
        require_aware_utc(at, "at")
        current = self.attention(memory_id)
        updated = replace(
            current,
            activation_count=current.activation_count + 1,
            last_reinforced_at=at,
        )
        self._put_attention(updated)
        return updated

    def _put_attention(self, attention: MemoryAttention) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO memory_attention(memory_id,payload) VALUES(?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET payload=excluded.payload",
                (attention.memory_id, _encode_attention(attention)),
            )

    def open_thread(
        self,
        *,
        thread_id: str,
        scope: Scope,
        open_question: str,
        supporting_memory_ids: tuple[str, ...],
        at: datetime,
        importance: int = 5,
        working_summary: str | None = None,
        mature: bool = False,
    ) -> MemoryThread:
        self._writable()
        require_non_empty(thread_id, "thread_id")
        require_aware_utc(at, "at")
        if mature:
            raise ValueError("a new Thread cannot start mature")
        if (
            not supporting_memory_ids
            or len(supporting_memory_ids) > MAX_THREAD_ORIGIN_SUPPORT
            or len(set(supporting_memory_ids)) != len(supporting_memory_ids)
        ):
            raise ValueError("supporting_memory_ids must contain 1..8 unique items")
        memories = tuple(self._memory(mid, active=True) for mid in supporting_memory_ids)
        if any(memory.scope != scope for memory in memories):
            raise ValueError("thread support must exactly match thread Scope")
        thread = MemoryThread(
            thread_id=thread_id,
            scope=scope,
            open_question=open_question,
            status=ThreadStatus.OPEN,
            importance=importance,
            created_at=at,
            updated_at=at,
            touch_count=0,
            suppressed=False,
            origin_memory_ids=supporting_memory_ids,
            current_support_ids=supporting_memory_ids,
            working_summary=working_summary,
            mature=mature,
        )
        existing = self.get_thread(thread_id)
        if existing is not None:
            if existing != thread:
                raise MemoryProductConflict(f"conflicting thread identity: {thread_id}")
            return existing
        self._put_thread(thread)
        return thread

    def get_thread(self, thread_id: str) -> MemoryThread | None:
        require_non_empty(thread_id, "thread_id")
        if not self._table_exists("memory_threads"):
            return None
        row = self._conn.execute(
            "SELECT payload FROM memory_threads WHERE thread_id=?", (thread_id,)
        ).fetchone()
        return None if row is None else _decode_thread(row[0])

    def list_threads(
        self,
        scope: Scope,
        *,
        status: ThreadStatus | None = None,
        include_suppressed: bool = True,
    ) -> tuple[MemoryThread, ...]:
        """List product Threads without ranking or semantic interpretation."""
        if status is not None and not isinstance(status, ThreadStatus):
            raise ValueError("status must be ThreadStatus or None")
        if type(include_suppressed) is not bool:
            raise ValueError("include_suppressed must be bool")
        if not self._table_exists("memory_threads"):
            return ()
        threads: list[MemoryThread] = []
        for (payload,) in self._conn.execute(
            "SELECT payload FROM memory_threads ORDER BY thread_id"
        ):
            thread = _decode_thread(payload)
            if thread.scope != scope:
                continue
            if status is not None and thread.status is not status:
                continue
            if not include_suppressed and thread.suppressed:
                continue
            if not self._thread_support_is_current(thread):
                continue
            threads.append(thread)
        return tuple(threads)

    def update_thread(
        self,
        thread_id: str,
        *,
        supporting_memory_ids: tuple[str, ...],
        at: datetime,
        working_summary: str | None = None,
        mature: bool | None = None,
    ) -> MemoryThread:
        """Replace the bounded current working set; do not append history."""
        self._writable()
        require_aware_utc(at, "at")
        thread = self._require_open_thread(thread_id)
        if (
            not supporting_memory_ids
            or len(supporting_memory_ids) > MAX_THREAD_CURRENT_SUPPORT
            or len(set(supporting_memory_ids)) != len(supporting_memory_ids)
        ):
            raise ValueError("supporting_memory_ids must contain 1..8 unique items")
        memories = tuple(self._memory(mid, active=True) for mid in supporting_memory_ids)
        if any(memory.scope != thread.scope for memory in memories):
            raise ValueError("thread support must exactly match thread Scope")
        next_summary = thread.working_summary if working_summary is None else working_summary
        next_mature = thread.mature if mature is None else mature
        maturity_support = tuple(
            dict.fromkeys((*thread.origin_memory_ids, *supporting_memory_ids))
        )
        if next_mature and not self._has_independent_support(maturity_support):
            if mature is True:
                raise ValueError(
                    "mature Thread requires support from at least two interactions"
                )
            next_mature = False
        if (
            thread.current_support_ids == supporting_memory_ids
            and thread.working_summary == next_summary
            and thread.mature == next_mature
        ):
            return thread
        updated = replace(
            thread,
            updated_at=at,
            touch_count=thread.touch_count + 1,
            current_support_ids=supporting_memory_ids,
            working_summary=next_summary,
            mature=next_mature,
        )
        self._put_thread(updated)
        return updated

    def resolve_thread(
        self,
        thread_id: str,
        *,
        memory_id: str,
        at: datetime,
        working_summary: str | None = None,
    ) -> MemoryThread:
        self._writable()
        require_aware_utc(at, "at")
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        if thread.status is ThreadStatus.RESOLVED:
            return thread
        if thread.status is not ThreadStatus.OPEN:
            raise ValueError("only OPEN threads can be resolved")
        memory = self._memory(memory_id, active=True)
        if memory.scope != thread.scope:
            raise ValueError("resolution Memory must exactly match thread Scope")
        support = tuple(
            dict.fromkeys((*thread.current_support_ids, memory_id))
        )[-MAX_THREAD_CURRENT_SUPPORT:]
        summary = thread.working_summary if working_summary is None else working_summary
        maturity_support = tuple(
            dict.fromkeys((*thread.origin_memory_ids, *support))
        )
        updated = replace(
            thread,
            status=ThreadStatus.RESOLVED,
            updated_at=at,
            touch_count=thread.touch_count + 1,
            current_support_ids=support,
            working_summary=summary,
            mature=(
                thread.mature
                and self._has_independent_support(thread.handoff_memory_ids)
            )
            or (
                summary is not None
                and self._has_independent_support(maturity_support)
            ),
        )
        self._put_thread(updated)
        return updated

    def thread_handoff(self, thread_id: str) -> MemoryThread:
        """Return one mature bounded Thread after revalidating canonical support."""
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        if thread.status is ThreadStatus.ABANDONED:
            raise ValueError("abandoned Thread cannot be handed off")
        if not thread.mature or thread.working_summary is None:
            raise ValueError("Thread is not mature for LCE handoff")
        if not self._thread_support_is_current(thread):
            raise ValueError("Thread support is not current")
        if not self._has_independent_support(thread.handoff_memory_ids):
            raise ValueError("Thread maturity lacks independent interaction support")
        return thread

    def retire_compiled_thread(
        self,
        thread_id: str,
        *,
        baseline_id: str,
    ) -> bool:
        """Remove a temporary Thread after LCE accepted its higher projection.

        The accepted Baseline owns durable cognition lineage under the stable
        mr-thread:<thread_id> region and canonical Memory IDs keep factual
        provenance. Keeping the Thread row after that point would duplicate the
        same logical product. Missing rows are an idempotent no-op.
        """
        self._writable()
        require_non_empty(baseline_id, "baseline_id")
        thread = self.get_thread(thread_id)
        if thread is None:
            return False
        if thread.status is ThreadStatus.ABANDONED:
            raise ValueError("abandoned Thread cannot be retired as compiled")
        if not thread.mature or thread.working_summary is None:
            raise ValueError("Thread is not mature for compilation retirement")
        if not self._thread_support_is_current(thread):
            raise ValueError("Thread support is not current")
        if not self._has_independent_support(thread.handoff_memory_ids):
            raise ValueError("Thread maturity lacks independent interaction support")
        with self._conn:
            self._conn.execute(
                "DELETE FROM memory_threads WHERE thread_id=?",
                (thread_id,),
            )
        return True

    def abandon_thread(self, thread_id: str, *, at: datetime) -> MemoryThread:
        self._writable()
        require_aware_utc(at, "at")
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        if thread.status is ThreadStatus.ABANDONED:
            return thread
        if thread.status is not ThreadStatus.OPEN:
            raise ValueError("only OPEN threads can be abandoned")
        updated = replace(thread, status=ThreadStatus.ABANDONED, updated_at=at)
        self._put_thread(updated)
        return updated

    def suppress_thread(self, thread_id: str, suppressed: bool = True) -> MemoryThread:
        self._writable()
        if type(suppressed) is not bool:
            raise ValueError("suppressed must be bool")
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        updated = replace(thread, suppressed=suppressed)
        self._put_thread(updated)
        return updated

    def surface_threads(
        self,
        scope: Scope,
        *,
        now: datetime,
        limit: int = 5,
        decay_lambda: float = 0.01,
    ) -> tuple[MemoryThread, ...]:
        require_aware_utc(now, "now")
        if type(limit) is not int or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer in [0, 100]")
        if not math.isfinite(decay_lambda) or decay_lambda < 0:
            raise ValueError("decay_lambda must be finite and non-negative")
        if limit == 0 or not self._table_exists("memory_threads"):
            return ()
        candidates = []
        for (payload,) in self._conn.execute("SELECT payload FROM memory_threads"):
            thread = _decode_thread(payload)
            if (
                thread.scope != scope
                or thread.status is not ThreadStatus.OPEN
                or thread.suppressed
                or not self._thread_support_is_current(thread)
            ):
                continue
            age_days = max(0.0, (now - thread.updated_at).total_seconds() / 86400.0)
            score = (
                thread.importance
                * (1.0 + math.log1p(thread.touch_count))
                * math.exp(-decay_lambda * age_days)
            )
            candidates.append((score, thread.updated_at, thread.thread_id, thread))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return tuple(item[3] for item in candidates[:limit])

    def _has_independent_support(self, memory_ids: tuple[str, ...]) -> bool:
        event_ids = {
            event_id
            for memory_id in memory_ids
            if (memory := self._canonical.get(memory_id)) is not None
            and memory.lifecycle is MemoryLifecycle.ACTIVE
            for event_id in memory.provenance.support_event_ids
        }
        return len(event_ids) >= 2

    def _thread_support_is_current(self, thread: MemoryThread) -> bool:
        for memory_id in thread.handoff_memory_ids:
            memory = self._canonical.get(memory_id)
            if (
                memory is None
                or memory.scope != thread.scope
                or memory.lifecycle is not MemoryLifecycle.ACTIVE
            ):
                return False
        return True

    def _require_open_thread(self, thread_id: str) -> MemoryThread:
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        if thread.status is not ThreadStatus.OPEN:
            raise ValueError("thread is not OPEN")
        return thread

    def _put_thread(self, thread: MemoryThread) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO memory_threads(thread_id,payload) VALUES(?,?) "
                "ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload",
                (thread.thread_id, _encode_thread(thread)),
            )


def _encode_attention(attention: MemoryAttention) -> str:
    data = asdict(attention)
    if attention.last_reinforced_at is not None:
        data["last_reinforced_at"] = attention.last_reinforced_at.isoformat()
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _decode_attention(payload: str) -> MemoryAttention:
    data = json.loads(payload)
    if data["last_reinforced_at"] is not None:
        data["last_reinforced_at"] = datetime.fromisoformat(data["last_reinforced_at"])
    return MemoryAttention(**data)


def _encode_thread(thread: MemoryThread) -> str:
    data = asdict(thread)
    data["scope"]["domain"] = thread.scope.domain.value
    data["status"] = thread.status.value
    data["created_at"] = thread.created_at.isoformat()
    data["updated_at"] = thread.updated_at.isoformat()
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


def _decode_thread(payload: str) -> MemoryThread:
    data = json.loads(payload)
    scope_data = data.pop("scope")
    scope_data["domain"] = ScopeDomain(scope_data["domain"])
    data["scope"] = Scope(**scope_data)
    data["status"] = ThreadStatus(data["status"])
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["updated_at"] = datetime.fromisoformat(data["updated_at"])

    # Legacy migration: earlier product code stored an append-only event log
    # with PROGRESS/REVERSAL labels. Collapse it into bounded support only;
    # trajectory interpretation belongs to LCE.
    legacy_events = data.pop("events", None)
    if legacy_events is not None:
        opened = [
            str(event["memory_id"])
            for event in legacy_events
            if event.get("transition") == "opened"
        ]
        all_ids = [str(event["memory_id"]) for event in legacy_events if event.get("memory_id")]
        unique_all = tuple(dict.fromkeys(all_ids))
        data["origin_memory_ids"] = (
            tuple(dict.fromkeys(opened))[:MAX_THREAD_ORIGIN_SUPPORT] or unique_all[:1]
        )
        data["current_support_ids"] = unique_all[-MAX_THREAD_CURRENT_SUPPORT:]
        data["working_summary"] = None
        data["mature"] = False

    data["origin_memory_ids"] = tuple(data["origin_memory_ids"])
    data["current_support_ids"] = tuple(data["current_support_ids"])
    return MemoryThread(**data)
