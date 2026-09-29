"""Hybrid retrieval and optional HyDE query expansion.

These providers only rank stable Memory IDs. Canonical validation remains in
MemoryRetrievalService.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from mr_mem.memory.retrieval import (
    MemoryRetrievalQuery,
    RetrievalProvider,
    RetrievalProviderUnavailable,
    RetrievedMemoryCandidate,
)


@dataclass(frozen=True, slots=True)
class RetrievalArm:
    name: str
    provider: RetrievalProvider
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("retrieval arm name must be nonempty")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("retrieval arm weight must be finite and positive")


def _candidate_limit(limit: int, multiplier: int) -> int:
    return min(100, max(limit, limit * multiplier))


def _fuse(
    ranked_lists: tuple[tuple[str, float, tuple[RetrievedMemoryCandidate, ...]], ...],
    *,
    limit: int,
    rrf_k: int,
    provider_name: str,
) -> tuple[RetrievedMemoryCandidate, ...]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    refs: dict[str, list[str]] = {}
    for arm_name, weight, hits in ranked_lists:
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if not isinstance(hit, RetrievedMemoryCandidate) or hit.memory_id in seen:
                continue
            seen.add(hit.memory_id)
            scores[hit.memory_id] = scores.get(hit.memory_id, 0.0) + weight / (rrf_k + rank)
            best_rank[hit.memory_id] = min(best_rank.get(hit.memory_id, rank), rank)
            refs.setdefault(hit.memory_id, []).append(
                f"{arm_name}:{hit.provider_ref or hit.memory_id}"
            )
    ranked = sorted(
        scores,
        key=lambda memory_id: (-scores[memory_id], best_rank[memory_id], memory_id),
    )
    return tuple(
        RetrievedMemoryCandidate(
            memory_id=memory_id,
            provider=provider_name,
            provider_ref="|".join(refs[memory_id]),
            score=scores[memory_id],
        )
        for memory_id in ranked[:limit]
    )


class HybridRRFProvider:
    """Fuse independent lexical/semantic ranks without comparing raw score scales."""

    def __init__(
        self,
        arms: tuple[RetrievalArm, ...],
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if len(arms) < 2:
            raise ValueError("hybrid retrieval requires at least two arms")
        names = [arm.name for arm in arms]
        if len(set(names)) != len(names):
            raise ValueError("retrieval arm names must be unique")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if type(candidate_multiplier) is not int or not 1 <= candidate_multiplier <= 20:
            raise ValueError("candidate_multiplier must be an integer in [1, 20]")
        self._arms = arms
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        if query.limit == 0:
            return ()
        expanded_limit = _candidate_limit(query.limit, self._candidate_multiplier)
        expanded_query = replace(query, limit=expanded_limit)
        ranked_lists: list[
            tuple[str, float, tuple[RetrievedMemoryCandidate, ...]]
        ] = []
        for arm in self._arms:
            try:
                hits = arm.provider.search(expanded_query)
            except Exception as exc:
                raise RetrievalProviderUnavailable(
                    f"hybrid retrieval arm unavailable: {arm.name}"
                ) from exc
            ranked_lists.append((arm.name, arm.weight, hits))
        return _fuse(
            tuple(ranked_lists),
            limit=query.limit,
            rrf_k=self._rrf_k,
            provider_name="rrf:" + "+".join(arm.name for arm in self._arms),
        )


class QueryExpander(Protocol):
    def expand(self, query: str) -> str: ...


class CompletionFunction(Protocol):
    def __call__(self, prompt: str) -> str: ...


class PromptHyDEExpander:
    """Model-agnostic HyDE prompt adapter; the host supplies the completion call."""

    def __init__(self, complete: CompletionFunction, *, max_characters: int = 2048) -> None:
        if type(max_characters) is not int or not 128 <= max_characters <= 4096:
            raise ValueError("max_characters must be an integer in [128, 4096]")
        self._complete = complete
        self._max_characters = max_characters

    def expand(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("HyDE query must be nonempty")
        prompt = (
            "Write one short hypothetical memory passage that would directly answer "
            "or contextualize the user's query. Preserve concrete entities, numbers, "
            "versions and dates from the query. Do not answer the user and do not add "
            "commentary. Return only the hypothetical passage.\n\nQuery:\n"
            + query
        )
        result = self._complete(prompt)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("HyDE completion returned no usable text")
        return result.strip()[: self._max_characters]


class HyDEFallbackProvider:
    """Escalate to HyDE only when the primary retrieval result set is sparse.

    A successful primary search remains usable if optional HyDE expansion or
    its supplemental search fails. HyDE therefore adds recall capacity without
    turning every search into an LLM call or making the cheaper path less
    available.
    """

    def __init__(
        self,
        provider: RetrievalProvider,
        expander: QueryExpander,
        *,
        min_results: int = 3,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if type(min_results) is not int or not 1 <= min_results <= 100:
            raise ValueError("min_results must be an integer in [1, 100]")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if type(candidate_multiplier) is not int or not 1 <= candidate_multiplier <= 20:
            raise ValueError("candidate_multiplier must be an integer in [1, 20]")
        self._provider = provider
        self._expander = expander
        self._min_results = min_results
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        if query.limit == 0:
            return ()
        try:
            original = self._provider.search(query)
        except Exception as exc:
            raise RetrievalProviderUnavailable("primary retrieval unavailable before HyDE") from exc

        required = min(query.limit, self._min_results)
        unique_original = tuple(
            dict.fromkeys(
                hit.memory_id
                for hit in original
                if isinstance(hit, RetrievedMemoryCandidate)
            )
        )
        if len(unique_original) >= required:
            return original[: query.limit]

        try:
            expansion = self._expander.expand(query.text)
            if not isinstance(expansion, str) or not expansion.strip():
                raise ValueError("HyDE expander returned empty text")
            if expansion.strip() == query.text.strip():
                return original[: query.limit]
            expanded_limit = _candidate_limit(query.limit, self._candidate_multiplier)
            hypothetical = self._provider.search(
                MemoryRetrievalQuery(query.scope, expansion, expanded_limit)
            )
        except Exception:
            return original[: query.limit]

        return _fuse(
            (
                ("original", 1.0, original),
                ("hyde", 1.0, hypothetical),
            ),
            limit=query.limit,
            rrf_k=self._rrf_k,
            provider_name="hyde-fallback+rrf",
        )


class HyDEAugmentedProvider:
    """Fuse original-query and HyDE-query ranks from the same base provider."""

    def __init__(
        self,
        provider: RetrievalProvider,
        expander: QueryExpander,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        fallback_to_original: bool = False,
    ) -> None:
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if type(candidate_multiplier) is not int or not 1 <= candidate_multiplier <= 20:
            raise ValueError("candidate_multiplier must be an integer in [1, 20]")
        if type(fallback_to_original) is not bool:
            raise ValueError("fallback_to_original must be bool")
        self._provider = provider
        self._expander = expander
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier
        self._fallback_to_original = fallback_to_original

    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        if query.limit == 0:
            return ()
        expanded_limit = _candidate_limit(query.limit, self._candidate_multiplier)
        base_query = replace(query, limit=expanded_limit)
        try:
            original = self._provider.search(base_query)
        except Exception as exc:
            raise RetrievalProviderUnavailable("base retrieval unavailable for HyDE") from exc

        try:
            expansion = self._expander.expand(query.text)
            if not isinstance(expansion, str) or not expansion.strip():
                raise ValueError("HyDE expander returned empty text")
            if expansion.strip() == query.text.strip():
                return original[: query.limit]
            hypothetical = self._provider.search(
                MemoryRetrievalQuery(query.scope, expansion, expanded_limit)
            )
        except Exception as exc:
            if self._fallback_to_original:
                return original[: query.limit]
            raise RetrievalProviderUnavailable("HyDE expansion/retrieval unavailable") from exc

        return _fuse(
            (
                ("original", 1.0, original),
                ("hyde", 1.0, hypothetical),
            ),
            limit=query.limit,
            rrf_k=self._rrf_k,
            provider_name="hyde+rrf",
        )
