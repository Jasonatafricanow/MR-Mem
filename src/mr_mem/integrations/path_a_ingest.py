"""Route compiled Block signals into Thread maintenance and LCE Path A.

A host's semantic compiler emits, per admitted canonical Block, at most one
Thread signal and at most one first-appearance commit. This module applies them
in one place so every host (AML adapter, agent integration) shares the same
Path A semantics:

- a settling signal (``resolve`` or ``mature``) on a line that already has an
  accepted LCE HEAD continues that line (MEMORY_ARCHITECTURE_V1 section 7,
  "B1 + M4 -> B2") instead of reopening a Thread that would need new
  independent support;
- every other Thread signal goes to ``ThreadAutoUpdateService``, whose stateless
  rescan hands mature Threads to LCE (ADR-0033, two-interaction rule unchanged);
- a first-appearance commit becomes ``commit`` on a new line, or ``revise`` when
  the line already exists.

Thread and LCE are derived projections: failures here never undo canonical
Memory. Thread failures are counted; promotion failures are returned so the
host can retry the same input (all steps are idempotent).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from mr_mem.contracts import Scope
from mr_mem.memory.store import CanonicalMemoryStore
from mr_mem.memory.threading import ThreadAutoUpdateService

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ThreadSignal:
    action: str  # track | resolve | ...
    question: str
    summary: str | None = None
    mature: bool = False

    @property
    def settles(self) -> bool:
        return self.action == "resolve" or self.mature


@dataclass(frozen=True, slots=True)
class CommitSignal:
    question: str
    summary: str


@dataclass(frozen=True, slots=True)
class BlockSignals:
    memory_id: str
    content: str
    thread: ThreadSignal | None = None
    commit: CommitSignal | None = None


@dataclass(frozen=True, slots=True)
class ThreadEvent:
    scope: Scope
    attributes: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IngestResult:
    questions: tuple[str, ...]  # lines touched, for the compiler's known-lines list
    thread_failed: bool
    promotion_failures: int


class SafeProjectionCompiler:
    """A Path A failure leaves the Thread intact for the next rescan (ADR-0033)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.failures = 0

    def compile(self, thread):
        try:
            return self.inner.compile(thread)
        except Exception:  # noqa: BLE001 - derived projection boundary
            self.failures += 1
            log.exception("Path A handoff failed for %s; Thread kept", thread.thread_id)
            return None


def apply_block_signals(
    *,
    canonical: CanonicalMemoryStore,
    products,
    lce,
    scope: Scope,
    blocks: Sequence[BlockSignals],
    at: datetime,
) -> IngestResult:
    """Apply one admitted batch. ``lce=None`` maintains Threads only."""
    questions: list[str] = []
    events: list[ThreadEvent] = []
    revisions: list[tuple[str, str, str]] = []
    for block in blocks:
        signal = block.thread
        if signal is None:
            continue
        questions.append(signal.question)
        if signal.settles and lce is not None and lce.lineage_head(signal.question):
            revisions.append((signal.question, signal.summary or block.content, block.memory_id))
            continue
        attrs = [("thread_action", signal.action), ("thread_question", signal.question)]
        if signal.summary:
            attrs.append(("thread_summary", signal.summary))
        if signal.mature:
            attrs.append(("thread_mature", "true"))
        events.append(ThreadEvent(scope, tuple(attrs), (block.memory_id,)))
    thread_failed = False
    try:
        ThreadAutoUpdateService(
            canonical=canonical, product=products,
            projection_compiler=None if lce is None else SafeProjectionCompiler(lce),
        ).apply(scope=scope, accepted_events=tuple(events), at=at)
    except Exception:  # noqa: BLE001 - Thread is a derived projection
        thread_failed = True
        log.exception("Thread maintenance failed; canonical Memory is unaffected")
    failures = 0
    if lce is not None:
        for block in blocks:
            if block.commit is not None:
                questions.append(block.commit.question)
                revisions.append((block.commit.question, block.commit.summary, block.memory_id))
        for question, summary, memory_id in revisions:
            try:
                if lce.lineage_head(question) is not None:
                    lce.revise(question=question, summary=summary, support=(memory_id,))
                else:
                    lce.commit(question=question, summary=summary, support=(memory_id,))
            except Exception:  # noqa: BLE001 - derived projection boundary
                failures += 1
                log.exception("Path A promotion failed for %s", memory_id)
    return IngestResult(tuple(dict.fromkeys(questions)), thread_failed, failures)
