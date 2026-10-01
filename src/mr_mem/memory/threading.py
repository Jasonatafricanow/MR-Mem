"""Automatic bounded Thread maintenance from already-accepted semantic events.

This module does not call a model. It reuses the semantic event already produced
for the current turn, resolves that event back to canonical Memory, and
updates the medium-term Thread working set deterministically.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from mr_mem.contracts import Scope
from mr_mem.contracts.common import require_aware_utc
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.product import (
    MAX_THREAD_CURRENT_SUPPORT,
    MemoryProductStore,
    MemoryThread,
    ThreadStatus,
)
from mr_mem.memory.providers.bm25 import lexical_tokens
from mr_mem.memory.store import CanonicalMemoryStore, scope_json


@runtime_checkable
class ThreadSemanticEvent(Protocol):
    """Minimum accepted semantic signal shape consumed by Thread maintenance."""

    @property
    def scope(self) -> Scope: ...

    @property
    def attributes(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def evidence_refs(self) -> tuple[str, ...]: ...


class ThreadSignalAction(StrEnum):
    TRACK = "track"
    RESOLVE = "resolve"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ThreadSignal:
    action: ThreadSignalAction
    open_question: str | None
    summary: str | None
    mature: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action, ThreadSignalAction):
            raise ValueError("action must be ThreadSignalAction")
        if self.open_question is not None:
            if not isinstance(self.open_question, str) or not self.open_question.strip():
                raise ValueError("open_question must be nonempty when supplied")
            if len(self.open_question) > 2048:
                raise ValueError("open_question exceeds 2048 characters")
        if self.summary is not None:
            if not isinstance(self.summary, str) or not self.summary.strip():
                raise ValueError("summary must be nonempty when supplied")
            if len(self.summary) > 4096:
                raise ValueError("summary exceeds 4096 characters")
        if type(self.mature) is not bool:
            raise ValueError("mature must be bool")


@runtime_checkable
class ThreadUpdatePort(Protocol):
    def apply(
        self,
        *,
        scope: Scope,
        accepted_events: tuple[ThreadSemanticEvent, ...],
        at: datetime,
    ) -> tuple[MemoryThread, ...]:
        ...


@runtime_checkable
class ThreadProjectionCompiler(Protocol):
    """Compile a mature Thread into a higher-level cognition projection.

    Returning a Baseline ID means the projection was accepted and the Thread
    may leave the active working set. Returning None leaves the Thread active.
    """

    def compile(self, thread: MemoryThread) -> str | None:
        ...


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _bool_text(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().casefold() in {"true", "yes", "1"}


def _signal(event: ThreadSemanticEvent) -> ThreadSignal | None:
    attrs = dict(event.attributes)
    raw_action = attrs.get("thread_action")
    if raw_action is None:
        return None
    try:
        action = ThreadSignalAction(raw_action.strip().casefold())
    except ValueError:
        return None
    if action is ThreadSignalAction.NONE:
        return None
    question = attrs.get("thread_question")
    summary = attrs.get("thread_summary")
    question = question.strip() if isinstance(question, str) and question.strip() else None
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None
    if action is ThreadSignalAction.TRACK and question is None:
        return None
    if action is ThreadSignalAction.RESOLVE and question is None and summary is None:
        return None
    return ThreadSignal(
        action=action,
        open_question=question,
        summary=summary,
        mature=_bool_text(attrs.get("thread_mature")),
    )


def _thread_identity(scope: Scope, question: str) -> str:
    payload = json.dumps(
        ["mr-thread-v1", json.loads(scope_json(scope)), _normalized(question)],
        sort_keys=True,
        ensure_ascii=False,
    )
    return "thread-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ThreadAutoUpdateService:
    """Maintain Threads from accepted semantic output without duplicate reasoning."""

    def __init__(
        self,
        *,
        canonical: CanonicalMemoryStore,
        product: MemoryProductStore,
        minimum_match: float = 0.30,
        projection_compiler: ThreadProjectionCompiler | None = None,
    ) -> None:
        if (
            isinstance(minimum_match, bool)
            or not isinstance(minimum_match, (int, float))
            or not 0 <= minimum_match <= 1
        ):
            raise ValueError("minimum_match must be in [0, 1]")
        self._canonical = canonical
        self._product = product
        self._minimum_match = float(minimum_match)
        self._projection_compiler = projection_compiler

    def close(self) -> None:
        self._product.close()
        self._canonical.close()

    def apply(
        self,
        *,
        scope: Scope,
        accepted_events: tuple[ThreadSemanticEvent, ...],
        at: datetime,
    ) -> tuple[MemoryThread, ...]:
        require_aware_utc(at, "at")
        changed_by_id: dict[str, MemoryThread] = {}
        for event in accepted_events:
            if event.scope != scope:
                continue
            signal = _signal(event)
            if signal is None:
                continue
            support = self._supporting_memories(scope, event.evidence_refs)
            if not support:
                continue
            updated: MemoryThread | None
            if signal.action is ThreadSignalAction.TRACK:
                updated = self._track(scope, signal, support, at)
            else:
                updated = self._resolve(scope, signal, support, at)
            if updated is not None:
                changed_by_id[updated.thread_id] = updated

        # Thread is a temporary projection. Once the same logical line has an
        # accepted higher-level projection, delete the lower projection instead
        # of maintaining two copies of the same logical product.
        if self._projection_compiler is not None:
            for thread in self._product.list_threads(scope, include_suppressed=True):
                if (
                    thread.status in (ThreadStatus.OPEN, ThreadStatus.RESOLVED)
                    and thread.mature
                    and thread.working_summary is not None
                    and self._product._has_independent_support(
                        thread.handoff_memory_ids
                    )
                ):
                    baseline_id = self._projection_compiler.compile(thread)
                    if baseline_id is None:
                        continue
                    self._product.retire_compiled_thread(
                        thread.thread_id,
                        baseline_id=baseline_id,
                    )
                    # Keep the affected snapshot in the return value for turn
                    # tracing; the projection itself no longer exists in the
                    # active/product store after successful compilation.
                    changed_by_id[thread.thread_id] = thread

        return tuple(changed_by_id.values())

    def _supporting_memories(
        self, scope: Scope, refs: tuple[str, ...]
    ) -> tuple[CommittedMemory, ...]:
        allowed = set(refs)
        if not allowed:
            return ()
        matches = [
            memory
            for memory in self._canonical.load_all()
            if (
                memory.scope == scope
                and memory.lifecycle is MemoryLifecycle.ACTIVE
                and (
                    memory.memory_id in allowed
                    or memory.provenance.observation_id in allowed
                    or bool(set(memory.provenance.evidence_refs) & allowed)
                )
            )
        ]
        matches.sort(key=lambda item: (item.committed_at, item.memory_id))
        return tuple(matches[-MAX_THREAD_CURRENT_SUPPORT:])

    def _track(
        self,
        scope: Scope,
        signal: ThreadSignal,
        support: tuple[CommittedMemory, ...],
        at: datetime,
    ) -> MemoryThread:
        assert signal.open_question is not None
        existing = self._best_match(
            scope,
            question=signal.open_question,
            summary=signal.summary,
        )
        support_ids = tuple(memory.memory_id for memory in support)
        if existing is None:
            return self._product.open_thread(
                thread_id=_thread_identity(scope, signal.open_question),
                scope=scope,
                open_question=signal.open_question,
                supporting_memory_ids=support_ids,
                at=at,
                working_summary=signal.summary,
                mature=False,
            )

        merged_support = tuple(
            dict.fromkeys((*existing.current_support_ids, *support_ids))
        )[-MAX_THREAD_CURRENT_SUPPORT:]
        next_summary = signal.summary or existing.working_summary
        all_support = tuple(
            dict.fromkeys((*existing.origin_memory_ids, *merged_support))
        )
        # Multiple Memory rows from one turn are still one supporting event.
        # Maturity requires support from at least two admitted interactions.
        mature = (
            existing.mature
            and self._product._has_independent_support(existing.handoff_memory_ids)
        ) or (
            next_summary is not None
            and self._product._has_independent_support(all_support)
            and (signal.mature or merged_support != existing.current_support_ids)
        )
        return self._product.update_thread(
            existing.thread_id,
            supporting_memory_ids=merged_support,
            at=at,
            working_summary=next_summary,
            mature=mature,
        )

    def _resolve(
        self,
        scope: Scope,
        signal: ThreadSignal,
        support: tuple[CommittedMemory, ...],
        at: datetime,
    ) -> MemoryThread | None:
        existing = self._best_match(
            scope,
            question=signal.open_question,
            summary=signal.summary,
        )
        if existing is None:
            return None
        support_ids = tuple(memory.memory_id for memory in support)
        merged_support = tuple(
            dict.fromkeys((*existing.current_support_ids, *support_ids))
        )[-MAX_THREAD_CURRENT_SUPPORT:]
        maturity_support = tuple(
            dict.fromkeys((*existing.origin_memory_ids, *merged_support))
        )
        current = self._product.update_thread(
            existing.thread_id,
            supporting_memory_ids=merged_support,
            at=at,
            working_summary=signal.summary or existing.working_summary,
            mature=(
                existing.mature
                and self._product._has_independent_support(
                    existing.handoff_memory_ids
                )
            )
            or (
                self._product._has_independent_support(maturity_support)
                and (signal.summary is not None or signal.mature)
            ),
        )
        return self._product.resolve_thread(
            current.thread_id,
            memory_id=support_ids[-1],
            at=at,
            working_summary=signal.summary or current.working_summary,
        )

    def _best_match(
        self,
        scope: Scope,
        *,
        question: str | None,
        summary: str | None,
    ) -> MemoryThread | None:
        threads = self._product.list_threads(
            scope,
            status=ThreadStatus.OPEN,
            include_suppressed=True,
        )
        if not threads:
            return None
        query_text = " ".join(value for value in (question, summary) if value)
        normalized_question = _normalized(question) if question else None
        query_tokens = set(lexical_tokens(query_text))
        ranked: list[tuple[float, datetime, str, MemoryThread]] = []
        for thread in threads:
            if normalized_question == _normalized(thread.open_question):
                score = 1.0
            else:
                supporting_text = " ".join(
                    memory.content
                    for memory_id in thread.current_support_ids
                    if (memory := self._canonical.get(memory_id)) is not None
                )
                candidate_text = " ".join(
                    value
                    for value in (
                        thread.open_question,
                        thread.working_summary,
                        supporting_text,
                    )
                    if value
                )
                candidate_tokens = set(lexical_tokens(candidate_text))
                smallest = min(len(query_tokens), len(candidate_tokens))
                if smallest < 2:
                    score = 0.0
                else:
                    score = len(query_tokens & candidate_tokens) / smallest
            if score >= self._minimum_match:
                ranked.append((score, thread.updated_at, thread.thread_id, thread))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return ranked[0][3]
