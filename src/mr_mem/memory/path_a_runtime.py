"""Canonical-first Add/Search composition seam for Path A; no model in MR-Mem.

A host may compile turns or a whole batch into SemanticMemoryCandidates.
Native admission validates provenance and commits SemanticBlocks before any
derived Thread change. No SemanticPoint or Path B dependency is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mr_mem.contracts import Scope
from mr_mem.memory.contracts import CommittedMemory, SemanticMemoryCandidate
from mr_mem.memory.core import MemoryCore
from mr_mem.memory.path_a import PathAAction, PathAProjector, PathAReceipt, PathAUpdate
from mr_mem.memory.path_a_lce import LcePathAProjection, ReadableBaseline
from mr_mem.memory.retrieval import (
    DEFAULT_SURFACE_BUDGET,
    MemoryRetrievalQuery,
    MemoryRetrievalService,
    MemorySurfaceBudget,
    ResolvedMemory,
)


@dataclass(frozen=True, slots=True)
class ThreadVerdict:
    """Host semantic compiler's decision about one admitted SemanticBlock."""
    event_id: str
    thread_key: str
    action: PathAAction
    open_question: str | None = None
    understanding: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticBlockInput:
    candidate: SemanticMemoryCandidate
    thread: ThreadVerdict | None = None


@dataclass(frozen=True, slots=True)
class PathAAddReceipt:
    committed: tuple[CommittedMemory, ...]
    thread_receipts: tuple[PathAReceipt, ...]


class PathASemanticWriter:
    """Real Canonical admission followed by explicit derived projection updates.

    The host provides semantics; this service does not embed a second LLM.
    The same source version cannot silently be reinterpreted on retry.
    """

    def __init__(self, core: MemoryCore, projector: PathAProjector,
                 *, sources, clock, origin_runtime_id: str) -> None:
        self.core, self.projector = core, projector
        self.admission = core.semantic_admission(
            sources=sources, clock=clock, origin_runtime_id=origin_runtime_id,
        )

    def add_blocks(self, *, scope: Scope, blocks: tuple[SemanticBlockInput, ...],
                   at: datetime) -> PathAAddReceipt:
        if not isinstance(blocks, tuple):
            raise TypeError("blocks must be a tuple")
        committed: list[CommittedMemory] = []
        verdicts: list[tuple[ThreadVerdict, CommittedMemory]] = []
        # Canonical first: projection updates can be retried after a failure.
        for block in blocks:
            if not isinstance(block, SemanticBlockInput) or block.candidate.scope != scope:
                raise ValueError("invalid or out-of-scope SemanticBlock input")
            memory = self.admission.admit(block.candidate)
            committed.append(memory)
            if block.thread is not None:
                verdicts.append((block.thread, memory))
        receipts = tuple(
            self.projector.apply(
                scope=scope,
                update=PathAUpdate(
                    event_id=verdict.event_id, thread_key=verdict.thread_key,
                    action=verdict.action, memory_ids=(memory.memory_id,),
                    open_question=verdict.open_question,
                    understanding=verdict.understanding,
                ),
                at=at,
            )
            for verdict, memory in verdicts
        )
        return PathAAddReceipt(tuple(committed), receipts)


@dataclass(frozen=True, slots=True)
class PathASearchResponse:
    canonical: tuple[ResolvedMemory, ...]
    baselines: tuple[ReadableBaseline, ...]


class PathASearch:
    """Canonical retrieval and LCE Baseline as *separate* result channels.

    Deliberately no RRF tuned on 22 questions, no entire-support fan-out.
    An Answer adapter may inspect both channels and their exact provenance.
    """

    def __init__(self, retrieval: MemoryRetrievalService,
                 lce: LcePathAProjection) -> None:
        self.retrieval, self.lce = retrieval, lce

    def search(self, query: MemoryRetrievalQuery, *,
               budget: MemorySurfaceBudget = DEFAULT_SURFACE_BUDGET,
               baseline_limit: int = 10) -> PathASearchResponse:
        if query.scope != self.lce.scope:
            raise ValueError("search scope mismatch")
        return PathASearchResponse(
            canonical=self.retrieval.search(query, budget=budget),
            baselines=self.lce.search(query.text, limit=baseline_limit),
        )
