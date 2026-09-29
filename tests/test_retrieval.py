from dataclasses import replace

import pytest

from mr_mem import Scope, ScopeDomain
from mr_mem.memory.retrieval import (
    MemoryRetrievalQuery,
    MemoryRetrievalService,
    MemorySurfaceBudget,
    RetrievalProviderUnavailable,
    RetrievedMemoryCandidate,
)
from mr_mem.memory.store import CanonicalMemoryStore
from tests.conftest import memory


class Provider:
    def __init__(self, hits=(), error=None):
        self.hits = hits
        self.error = error
        self.calls = 0

    def search(self, query):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.hits


class ReverseReranker:
    available = True

    def rerank(self, *, query, candidates):
        return tuple(candidate.memory_id for candidate in reversed(candidates))


def test_retrieval_revalidates_canonical_content_scope_and_lifecycle(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = CanonicalMemoryStore(path)
    first = memory("m1", content="canonical one")
    other_scope = Scope(ScopeDomain.USER, user_id="other")
    second = replace(
        memory("m2", content="other"),
        scope=other_scope,
        sync=replace(memory("m2").sync, scope=other_scope),
    )
    store._commit((first, second))
    provider = Provider(
        (
            RetrievedMemoryCandidate("m2", "fixture", score=1.0),
            RetrievedMemoryCandidate("m1", "fixture", provider_text="STALE", score=0.5),
        )
    )
    result = MemoryRetrievalService(store=store, provider=provider).search(
        MemoryRetrievalQuery(first.scope, "canonical")
    )
    assert [item.memory.memory_id for item in result] == ["m1"]
    assert result[0].memory.content == "canonical one"


def test_retrieval_budget_and_provider_outage_are_explicit(tmp_path):
    store = CanonicalMemoryStore(tmp_path / "memory.sqlite")
    row = memory("m1", content="hello")
    store._commit((row,))
    hit = RetrievedMemoryCandidate("m1", "fixture")
    service = MemoryRetrievalService(store=store, provider=Provider((hit,)))
    query = MemoryRetrievalQuery(row.scope, "hello")
    assert service.search(
        query, budget=MemorySurfaceBudget(max_items=1, max_characters=4)
    ) == ()
    assert service.search(
        query, budget=MemorySurfaceBudget(max_items=1, max_characters=5)
    )[0].memory == row

    down = MemoryRetrievalService(store=store, provider=Provider(error=TimeoutError("down")))
    with pytest.raises(RetrievalProviderUnavailable):
        down.search(query)


def test_zero_limit_never_calls_provider(tmp_path):
    store = CanonicalMemoryStore(tmp_path / "memory.sqlite")
    provider = Provider(error=AssertionError("must not call"))
    service = MemoryRetrievalService(store=store, provider=provider)
    scope = Scope(ScopeDomain.USER, user_id="u")
    assert service.search(MemoryRetrievalQuery(scope, "q", limit=0)) == ()
    assert provider.calls == 0


def test_optional_reranker_changes_only_authorized_candidate_order(tmp_path):
    store = CanonicalMemoryStore(tmp_path / "memory.sqlite")
    rows = (memory("m1", content="one"), memory("m2", content="two"))
    store._commit(rows)
    provider = Provider(
        (
            RetrievedMemoryCandidate("m1", "fixture"),
            RetrievedMemoryCandidate("m2", "fixture"),
        )
    )
    result = MemoryRetrievalService(
        store=store, provider=provider, reranker=ReverseReranker()
    ).search(MemoryRetrievalQuery(rows[0].scope, "query"))
    assert [item.memory.memory_id for item in result] == ["m2", "m1"]
