"""Bounded discovery followed by canonical Memory authority revalidation."""

from dataclasses import dataclass
from itertools import islice
from math import isfinite
from typing import Protocol

from mr_mem.contracts import Scope
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.rerank import MemoryRerankCandidate, MemoryReranker


def _bound(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")


@dataclass(frozen=True, slots=True)
class MemoryRetrievalQuery:
    scope: Scope
    text: str
    limit: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.scope, Scope):
            raise ValueError("structured Scope required")
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 4096:
            raise ValueError("query text must contain 1..4096 characters")
        _bound(self.limit, "limit", 100)


@dataclass(frozen=True, slots=True)
class RetrievedMemoryCandidate:
    memory_id: str
    provider: str
    provider_ref: str | None = None
    score: float | None = None
    provider_text: str | None = None

    def __post_init__(self) -> None:
        for value in (self.memory_id, self.provider):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("memory_id and provider must be nonempty")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite")


class RetrievalProviderUnavailable(RuntimeError):
    """An enabled discovery provider could not complete a query."""


class RetrievalProvider(Protocol):
    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]: ...


class CanonicalMemoryReader(Protocol):
    def get(self, memory_id: str) -> CommittedMemory | None: ...


class NullRetrievalProvider:
    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class MemorySurfaceBudget:
    max_items: int = 5
    max_characters: int = 4096

    def __post_init__(self) -> None:
        _bound(self.max_items, "max_items", 100)
        _bound(self.max_characters, "max_characters", 65536)


DEFAULT_SURFACE_BUDGET = MemorySurfaceBudget()


@dataclass(frozen=True, slots=True)
class ResolvedMemory:
    memory: CommittedMemory
    candidate: RetrievedMemoryCandidate
    rank: int


class MemoryRetrievalService:
    def __init__(
        self,
        *,
        store: CanonicalMemoryReader,
        provider: RetrievalProvider | None = None,
        reranker: MemoryReranker | None = None,
    ) -> None:
        self._store = store
        self._provider = provider if provider is not None else NullRetrievalProvider()
        self._reranker = reranker

    def _canonical_candidates(
        self,
        query: MemoryRetrievalQuery,
        hits: tuple[RetrievedMemoryCandidate, ...],
    ) -> tuple[ResolvedMemory, ...]:
        resolved: list[ResolvedMemory] = []
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if not isinstance(hit, RetrievedMemoryCandidate) or hit.memory_id in seen:
                continue
            memory = self._store.get(hit.memory_id)
            if (
                memory is None
                or memory.scope != query.scope
                or memory.lifecycle != MemoryLifecycle.ACTIVE
            ):
                continue
            seen.add(memory.memory_id)
            resolved.append(ResolvedMemory(memory, hit, rank))
        return tuple(resolved)

    @staticmethod
    def _apply_budget(
        candidates: tuple[ResolvedMemory, ...],
        budget: MemorySurfaceBudget,
    ) -> tuple[ResolvedMemory, ...]:
        result: list[ResolvedMemory] = []
        remaining = budget.max_characters
        for candidate in candidates:
            if len(candidate.memory.content) > remaining:
                continue
            remaining -= len(candidate.memory.content)
            result.append(candidate)
            if len(result) >= budget.max_items:
                break
        return tuple(result)

    def search(
        self,
        query: MemoryRetrievalQuery,
        *,
        budget: MemorySurfaceBudget = DEFAULT_SURFACE_BUDGET,
    ) -> tuple[ResolvedMemory, ...]:
        if not query.limit or not budget.max_items or not budget.max_characters:
            return ()
        try:
            hits = tuple(islice(self._provider.search(query), query.limit))
        except Exception as exc:
            raise RetrievalProviderUnavailable("semantic Memory provider unavailable") from exc

        canonical = self._canonical_candidates(query, hits)
        reranker = self._reranker
        if reranker is None or not reranker.available or len(canonical) < 2:
            return self._apply_budget(canonical, budget)

        ordered_ids = reranker.rerank(
            query=query.text,
            candidates=tuple(
                MemoryRerankCandidate(
                    memory_id=item.memory.memory_id,
                    content=item.memory.content,
                )
                for item in canonical
            ),
        )
        by_id = {item.memory.memory_id: item for item in canonical}
        ordered = tuple(
            by_id[memory_id]
            for memory_id in ordered_ids
            if memory_id in by_id
        )
        if len(ordered) != len(canonical):
            ordered = canonical
        return self._apply_budget(ordered, budget)
